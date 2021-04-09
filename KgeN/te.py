from .tir import *

# compute primitives
def var(name):
    return VarExpr(name)

def placeholder(shape, name):
    return TensorExpr(shape, name, TensorExpr.PLACEHOLDER)

def compute(shape, function, name):
    tensor = TensorExpr(shape, name, TensorExpr.COMPUTE, function)
    return tensor

def if_then_else(condition, then_expr, else_expr):
    return IfThenElseExpr(condition, then_expr, else_expr)
   
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
    axis = IterVar(name, 0, end)
    return axis