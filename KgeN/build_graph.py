from .tir import TensorExpr
from .visitor import CollectVisitor

class CollectInputVisitor(CollectVisitor):
    def __init__(self):
        super().__init__()
        self.inputs = set()
        self.providers = {}

    def collect(self, expr):
        expr.accept(self)
        return list(self.inputs), self.providers
    
    def visit_tensor_slice_expr(self, expr):
        if expr.tensor in self.providers:
            self.providers[expr.tensor].append(expr)
        else:
            self.providers[expr.tensor] = [expr]
        self.inputs.add(expr.tensor)
        for index in expr.index:
            index.accept(self)

def build_graph(tensor):
    visited = {tensor}
    q = [tensor]

    while len(q) > 0:
        tensor = q.pop()
        if tensor.type == TensorExpr.COMPUTE:
            visitor = CollectInputVisitor()
            tensor.inputs, tensor.providers = visitor.collect(tensor.expr)
            for inp in tensor.inputs:
                if inp not in visited:
                    visited.add(inp)
                    q.append(inp)
    
    for tensor in visited:
        tensor.outputs = []
    
    for tensor in visited:
        for inp in tensor.inputs:
            inp.outputs.append(tensor)
