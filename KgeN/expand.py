from .tir import *
from .visitor import RewriteVisitor
from .expr_simplifier import expr_simplifier

class Expander(RewriteVisitor):  
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

expr_expander = Expander()

def expand_pass(func):
    func = expr_expander.rewrite(func)
    func = expr_simplifier.rewrite(func)
    return func