from KgeN.tir.ir.expr import *
from KgeN.te.build_graph import CollectInputVisitor
from KgeN.arith.constraint import constraint_evaluator
from KgeN.arith.expr_simplifier import expr_simplifier
import math


class Operation:
    def __init__(self):
        # Operation's inputs and outputs
        self.inputs = []
        self.outputs = []
        self.providers = {}

class PlaceholderOp(Operation):
    def __init__(self):
        super().__init__()


class ComputeOp(Operation):
    def __init__(self, shape, name, compute_func):
        super().__init__()
        self.compute_func = compute_func
        # leaf axis
        self.axis = tuple([IterVar(name + "_" + compute_func.__code__.co_varnames[i], v, IterVar.DEFAULT) for i, v in enumerate(shape)])
        self.reduce_axis = ()

        self.expr = wrap_number_as_const_expr(self.compute_func(*self.axis))
        if isinstance(self.expr, ReduceExpr):
            self.reduce_axis = self.expr.reduce_axis

# compute primitives
def var(name):
    return VarExpr(name)

def placeholder(shape, name):
    op = PlaceholderOp()
    output = TensorExpr(shape, name, op)
    op.outputs = [output]
    return output

def compute(shape, function, name, scope="local"):
    op = ComputeOp(shape, name, function)
    output = TensorExpr(shape, name, op, scope=scope)

    visitor = CollectInputVisitor()
    op.inputs, op.providers = visitor.collect(op.expr)
    op.outputs = [output]
    return output

def if_then_else(condition, then_expr, else_expr):
    expr = IfThenElseExpr(condition, then_expr, else_expr)
    expr.condition = expr_simplifier.rewrite(condition)
    
    then_constraint = constraint_evaluator.evaluator(expr.condition)
    else_constraint = constraint_evaluator.evaluator(Expr.not_(expr.condition))
    expr.then_expr.constraint = then_constraint
    expr.else_expr.constraint = else_constraint
    # TODO:
    # recursively add to sub expr
    return expr

def all(*condition):
    assert len(condition) > 1, "provide at least two condition, got {}".format(len(condition))
    expr = condition[0]
    for condition in condition[1:]:
        expr = Expr.and_(expr, condition)
    return expr

def reduce_sum(expr, axis):
    combinator = lambda x, y: x + y
    return ReduceExpr(combinator, 0, expr, axis)

def reduce_max(expr, axis):
    combinator = lambda x, y: Expr.max(x, y)
    return ReduceExpr(combinator, -math.inf, expr, axis)

def reduce_min(expr, axis):
    combinator = lambda x, y: Expr.min(x, y)
    return ReduceExpr(combinator, math.inf, expr, axis)

def reduce_axis(end, name):
    axis = IterVar(name, end, IterVar.REDUCE)
    return axis

def thread_axis(end=None, tag="", name=""):
    if isinstance(end, str) and tag == "":
        end, tag = math.inf, end
    assert tag in ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z", "vthread"], "illegal binding name {}".format(tag)
    
    if not name:
        name = tag
    axis = IterVar(name, end, IterVar.BIND, tag)
    return axis