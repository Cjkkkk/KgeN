from .tir import TensorExpr
from .visitor import Visitor

class CollectInputVisitor(Visitor):
    def __init__(self):
        self.inputs = set()
        self.providers = {}

    def collect(self, expr):
        expr.accept(self)
        return list(self.inputs), self.providers
            
    def visit_binary_expr(self, expr):
        for subexpr in expr.subexprs:
            subexpr.accept(self)

    def visit_if_then_else_expr(self, expr):
        expr.condition.accept(self)
        expr.then_expr.accept(self)
        expr.else_expr.accept(self)
    
    def visit_reduce_expr(self, expr):
        expr.expr.accept(self)

    def visit_tensor_slice_expr(self, expr):
        if expr.tensor in self.providers:
            self.providers[expr.tensor].append(expr)
        else:
            self.providers[expr.tensor] = [expr]
        self.inputs.add(expr.tensor)
        for index in expr.index:
            index.accept(self)
    
    def visit_unary_expr(self, expr):
        pass

    def visit_var_expr(self, expr):
        pass

    def visit_const_expr(self, expr):
        pass

    def visit_iter_expr(self, expr):
        pass

    def visit_tensor_expr(self, expr):
        pass

def build_graph_pass(tensor):
    visited = {tensor}
    q = [tensor]
    visitor = CollectInputVisitor()

    while len(q) > 0:
        tensor = q.pop()
        if tensor.type == TensorExpr.COMPUTE:
            tensor.inputs, tensor.providers = visitor.collect(tensor.expr)

            for inp in tensor.inputs:
                if inp not in visited:
                    visited.add(inp)
                    q.append(inp)
                inp.outputs.append(tensor)
