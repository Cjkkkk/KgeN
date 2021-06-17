from .tir import *
import math

# compute primitives
def var(name):
    return VarExpr(name)

def placeholder(shape, name):
    return TensorExpr(shape, name, TensorExpr.PLACEHOLDER)

def compute(shape, function, name, scope="local"):
    tensor = TensorExpr(shape, name, TensorExpr.COMPUTE, function, scope=scope)
    return tensor

def if_then_else(condition, then_expr, else_expr):
    return IfThenElseExpr(condition, then_expr, else_expr)

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
    axis = IterVar(name, 0, end)
    axis.type = IterVar.REDUCE
    return axis

def thread_axis(end=None, name=""):
    if isinstance(end, str) and name == "":
        end, name = None, end
    if end is None:
        end = math.inf
    assert name in ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z"], "illegal binding name {}".format(name)
    axis = IterVar(name, 0, end)
    axis.type = IterVar.BIND
    return axis