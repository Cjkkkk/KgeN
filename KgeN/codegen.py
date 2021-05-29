from .utils import tensor_topo_sort_bottom_up, axis_topo_sort_top_down
from .tir import *
from .visitor import Visitor
from .te import *
from .schedule import *

# codegen
class CUDA_code_generator(Visitor):
    def __init__(self):
        self.scope = 0
        self.res = ""

    def emit(self, str):
        self.res += self.scope * "    " + str + "\n"
    
    def emit_newline(self):
        self.res += "\n"

    def enter_scope(self):
        self.scope += 1

    def exit_scope(self):
        self.scope -= 1
    
    def generate(self, func):
        func.accept(self)
        return self.res

    def generate_signature(self, func_stmt):
        self.emit("void kernel({}) {{".format(", ".join([tensor.dtype + "* " + tensor.name for tensor in func_stmt.input_tensors + func_stmt.output_tensors])))
    
    def generate_storage(self, func_stmt):
        for tensor in func_stmt.tensors:
            if tensor.scope == "global":
                continue
            elif tensor.scope == "local":
                self.emit("{0} {1};".format(tensor.dtype, TensorSliceExpr(tensor, [s for s in tensor.shape if not isinstance(s, ConstExpr) or s.val != 0])))
            elif tensor.scope == "shared":
                self.emit("__shared__ {0} {1};".format(tensor.dtype, TensorSliceExpr(tensor, [s for s in tensor.shape if not isinstance(s, ConstExpr) or s.val != 0])))
    
    def visit_func_stmt(self, stmt):
        for tensor in stmt.tensors:
            self.emit("// tensor: {0}".format(TensorSliceExpr(tensor, [s for s in tensor.shape if not isinstance(s, ConstExpr) or s.val != 0])))
        
        self.generate_signature(stmt)
        self.enter_scope()
        self.generate_storage(stmt)
        
        for st in stmt.body:
            st.accept(self)
        
        self.exit_scope()
        self.emit("}")

    def visit_assign_stmt(self, stmt):
        return self.emit(stmt.dest.accept(self) + " = " + stmt.source.accept(self) + ";")
    
    def visit_for_stmt(self, stmt):
        var = stmt.iter_var
        if not var.range.is_single_point and not var.bind_type == IterVar.BIND:
            self.emit("for (int {0} = {1}; {0} < {2} ; {0} += {3};) {{".format(
                var.name, 
                var.range.start.accept(self),
                var.range.end.accept(self),
                1))
            self.enter_scope()
        
        for st in stmt.body:
            st.accept(self)
        
        if not var.range.is_single_point and not var.bind_type == IterVar.BIND:
            self.exit_scope()
            self.emit("}")
    
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
            return "(({0} * {1}) + {2})".format(expr.splitted_outer.accept(self), expr.splitted_inner.range.end.accept(self), expr.splitted_inner.accept(self))
        elif expr.type == IterVar.FUSE:
            if expr is expr.fused.fused_outer:
                return "({0} // {1})".format(expr.fused.accept(self), expr.fused.fused_inner.range.end.accept(self))
            else:
                return "({0} % {1})".format(expr.fused.accept(self), expr.fused.fused_inner.range.end.accept(self))
        else:
            return expr.name

    def visit_if_then_else_expr(self, expr):
        return "({0} ? {1} : {2})".format(expr.condition.accept(self), expr.then_expr.accept(self), expr.else_expr.accept(self))
    
    def visit_tensor_slice_expr(self, expr):
        def f(a, b):
            return a * b
        def scan(f, state, l):
            res = []
            for e in l:
                state = f(state, e)
                res.append(state)
            return res
        
        prod = scan(f, ConstExpr(1), reversed(expr.tensor.shape[1:] + (ConstExpr(1),)))
        prod = reversed(prod)
        flatten_index = 0

        for index, prod in zip(expr.index, prod):
            flatten_index = flatten_index + index * prod
        # return expr.tensor.name + "[" + flatten_index.accept(self) + "]" 
        return expr.tensor.name + "[" + ", ".join([index.accept(self) for index in expr.index]) + "]" 

def CUDA_codegen_pass(func):
    code_generator = CUDA_code_generator()
    return code_generator.generate(func)