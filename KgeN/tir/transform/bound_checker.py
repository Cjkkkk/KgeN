from KgeN.tir.ir.visitor import RewriteVisitor
from KgeN.tir.ir.expr import ConstExpr


class BoundCheckerVisitor(RewriteVisitor):
    def __init__(self):
        super().__init__()
    
    def rewrite(self, func):
        return func.accept(self)

def bound_checker_pass(func):
    # TODO: 
    visitor = BoundCheckerVisitor()
    func = visitor.rewrite(func)
    return func