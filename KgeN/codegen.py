from .utils import tensor_topo_sort_bottom_up, axis_topo_sort_top_down
from .tir import *
from .visitor import Visitor
from .te import *
from .schedule import *

# codegen
class CUDA_code_generator(Visitor):
    def __init__(self):
        self.scope = 0
    
    def enter_scope(self):
        self.scope += 1

    def exit_scope(self):
        self.scope -= 1
    
    def gen_signature(self, func_stmt):
        return "void kernel({})".format(", ".join([tensor.dtype + "* " + tensor.name for tensor in func_stmt.input_tensors + func_stmt.output_tensors]))
    
    def visit_func_stmt(self, stmt):
        res = ""
        for tensor in stmt.tensors:
            res += "// tensor: {0}\n".format(TensorSliceExpr(tensor, tensor.shape))
        res += self.gen_signature(stmt)
        res += "{\n"
        self.enter_scope()
        res += "".join([st.accept(self) for st in stmt.body])
        self.exit_scope()
        res += "}\n"
        return res

    def visit_assign_stmt(self, stmt):
        return self.scope * "    " + stmt.dest.accept(self) + " = " + stmt.source.accept(self) + ";\n"
    
    def visit_for_stmt(self, stmt):
        ret = ""
        var = stmt.iter_var
        if not var.range.is_single_point and not var.bind_type == IterVar.BIND:
            ret = self.scope * "    " + "for (int {0} = {1}; {0} < {2} ; {0} += {3};) {{\n".format(
                var.name, 
                var.range.start.accept(self),
                var.range.end.accept(self),
                1)
            self.enter_scope()
        
        for st in stmt.body:
            ret += st.accept(self)
        
        if not var.range.is_single_point and not var.bind_type == IterVar.BIND:
            self.exit_scope()
            ret += self.scope * "    " + "}\n"
        return ret
    
    def visit_binary_expr(self, expr):
        if expr.type > 9: # min, max
            return "({1}({0}, {2}))".format(expr.left.accept(self), Expr.mapping[expr.type], expr.right.accept(self))
        return "({0} {1} {2})".format(expr.left.accept(self), Expr.mapping[expr.type], expr.right.accept(self))
    
    def visit_unary_expr(self, expr):
        return "({0}({1}))".format(Expr.mapping[expr.type], expr.expr.accept(self))

    def visit_var_expr(self, expr):
        return expr.name

    def visit_const_expr(self, expr):
        return str(expr.val)

    def visit_iter_expr(self, expr):
        if expr.range.is_single_point:
            return expr.range.start.accept(self)
        elif expr.bind_type == IterVar.BIND:
            return expr.bind_name
        elif expr.type == IterVar.SPLIT:
            return "(({0} * {1}) + {2})".format(expr.outer.accept(self), expr.inner.range.end.accept(self), expr.inner.accept(self))
        elif expr.type == IterVar.FUSE:
            if expr is expr.fused.outer:
                return "({0} // {1})".format(expr.fused.accept(self), expr.fused.inner.range.end.accept(self))
            else:
                return "({0} % {1})".format(expr.fused.accept(self), expr.fused.inner.range.end.accept(self))
        else:
            return expr.name

    def visit_if_then_else_expr(self, expr):
        return "({0} ? {1} : {2})".format(expr.condition.accept(self), expr.then_expr.accept(self), expr.else_expr.accept(self))
    
    def visit_tensor_slice_expr(self, expr):
        return expr.tensor.name + "[" + ", ".join([index.accept(self) for index in expr.index]) + "]" 

def CUDA_codegen_pass(func):
    code_generator = CUDA_code_generator()
    return func.accept(code_generator)