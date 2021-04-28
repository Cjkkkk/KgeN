from .utils import tensor_topo_sort_bottom_up
from .tir import *
from .visitor import Visitor


# codegen
class CUDA_code_generator(Visitor):
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
    
    def visit_reduce_expr(self, expr):
        raise NotImplemented

    def visit_tensor_expr(self, expr, indent=0):
        if expr.type == TensorExpr.PLACEHOLDER:
            return ""
        opening = ""
        closing = ""
        scope = indent
        # TODO: find out which axis to do reduce init
        # compose loop
        for i, axis in enumerate(expr.axis):
            if not axis.range.is_single_point and not axis.bind_type == IterVar.BIND:
                opening += "    " * scope + "for (int {0} = {1}; {0} < {2} ; {0} += {3};) {{\n".format(
                    axis.name, 
                    axis.range.start.accept(self),
                    axis.range.end.accept(self),
                    1)
                closing = "    " * scope + "}\n" + closing
                scope += 1
            
            for computation in axis.attached_computation:
                opening += computation.accept(self, scope)
        
        # TODO: should add boundary check if can not prove no out of index happens
        body = "    " * scope + TensorSliceExpr(expr, expr.root_axis).accept(self) + " = " + expr.expr.accept(self) + ";\n"
        return opening + body + closing

    def visit_tensor_slice_expr(self, expr):
        return expr.tensor.name + "[" + ", ".join([index.accept(self) for index in expr.index]) + "]" 

def CUDA_codegen_pass(tensor):
    tensors = tensor_topo_sort_bottom_up(tensor)
    code_generator = CUDA_code_generator()
    for tensor in tensors:
        print("buffer: {0}".format(TensorSliceExpr(tensor, tensor.shape)))
    res = ""
    for t in reversed(tensors):
        # skip codegen if it is attached to some axis
        if not t.attached:
            res += t.accept(code_generator)
    return res