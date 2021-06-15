from .tir import *
from .visitor import RewriteExprVisitor
from .expr_simplifier import expr_simplifier

class ExprExpander(RewriteExprVisitor):  
    def __init__(self):
        super().__init__()
    
    def visit_iter_expr(self, expr):
        if expr.range.is_single_point:
            return expr.range.start
        elif expr.relation == IterVar.SPLIT:
            # TODO: check this
            return expr.split_outer.accept(self) * expr.split_inner.range.end.accept(self) + expr.split_inner.accept(self)
        elif expr.relation == IterVar.FUSE:
            if expr is expr.fused.fused_outer:
                return expr.fused.accept(self) // expr.fused.fused_inner.range.end.accept(self)
            else:
                return expr.fused.accept(self) % expr.fused.fused_inner.range.end.accept(self)
        else:
            return expr

expr_expander = ExprExpander()

def expand_pass(tensors):
    for tensor in tensors:
        if tensor.type == TensorExpr.COMPUTE:
            tensor.expr = expr_expander.rewrite(tensor.expr)
            tensor.expr = expr_simplifier.rewrite(tensor.expr)