from .tir import *
from .visitor import Visitor
from .expr_simplifier import expr_simplifier
import math

def union_interval(a, b):
    return Interval(Expr.min(a.start, b.start), Expr.max(a.end, b.end))

def intersect_interval(a, b):
    return Interval(Expr.max(a.start, b.start), Expr.min(a.end, b.end))

class Interval:
    def __init__(self, start, end, stride=1):
        self.start = wrap_number_as_const_expr(start)
        self.end = wrap_number_as_const_expr(end)
        self.stride = wrap_number_as_const_expr(stride)

    @staticmethod
    def nothing():
        interval = Interval(math.inf, -math.inf)
        return interval

    @staticmethod
    def everything():
        interval = Interval(-math.inf, math.inf)
        return interval
    
    @property
    def is_nothing(self):
        return self.start.same_as(ConstExpr(math.inf)) and self.end.same_as(ConstExpr(-math.inf))
    
    @property
    def is_everything(self):
        return self.start.same_as(ConstExpr(-math.inf)) and self.end.same_as(ConstExpr(math.inf))
    
    def convert_to_range(self):
        return Range(self.start, self.end + 1, self.stride)

    def normalize(self):
        shift = ConstExpr(0)
        stride = ConstExpr(1)
        # TODO: fix this
        # if not self.start.same_as(ConstExpr(0)) and not self.is_single_point:
        if not self.start.same_as(ConstExpr(0)):
            shift = self.start
            self.end = self.end - self.start
            self.start = ConstExpr(0)
        if not self.stride.same_as(ConstExpr(1)):
            stride = self.stride
            self.stride = ConstExpr(1)
            self.end = self.end // stride
        return shift, stride

    def __str__(self):
        return "[{0}, {1}]".format(self.start, self.end)

# TODO: implement this
class IntervalSet:
    pass


class BoundEvaluator(Visitor):
    def __init__(self):
        super().__init__()
        self.rmap = None
        self.relax_set = None

    def evaluate(self, expr, rmap, constraint_map, relax_set):
        self.rmap = rmap
        self.constraint_map = constraint_map
        self.relax_set = relax_set
        return expr.accept(self)
    
    def visit_binary_expr(self, expr):
        left = expr.left.accept(self)
        right = expr.right.accept(self)
        if expr.type == Expr.ADD:
            interval = Interval(left.start + right.start, left.end + right.end)
        
        elif expr.type == Expr.SUB:
            interval = Interval(left.start - right.end, left.end - right.start)
        
        elif expr.type == Expr.MUL:
            ll = left.start * right.start
            lu = left.start * right.end
            ul = left.end * right.start
            uu = left.end * right.end
            # start and end could be negative
            interval = Interval(
                Expr.min(Expr.min(Expr.min(ll, lu), ul), uu), 
                Expr.max(Expr.max(Expr.max(ll, lu), ul), uu), 
                )
        
        elif expr.type == Expr.FLOOR_DIV: # TODO: fix this
            ll = left.start // right.start
            lu = left.start // right.end
            ul = left.end // right.start
            uu = left.end // right.end
            # start and end could be negative
            interval = Interval(
                Expr.min(Expr.min(Expr.min(ll, lu), ul), uu), 
                Expr.max(Expr.max(Expr.max(ll, lu), ul), uu), 
                )
        
        elif expr.type == Expr.MIN:
            interval = Interval(Expr.min(left.start, right.start), Expr.min(left.end, right.end))
        
        elif expr.type == Expr.MAX:
            interval = Interval(Expr.max(left.start, right.start), Expr.max(left.end, right.end))
        
        else:
            raise ValueError("Unsupported op type {}.".format(expr.type))
        return interval

    def visit_if_then_else_expr(self, expr):
        then_interval = expr.then_expr.accept(self)
        else_interval = expr.else_expr.accept(self)
        interval = union_interval(then_interval, else_interval)
        return interval
    
    def visit_unary_expr(self, expr):
        if expr.type == Expr.NEG:
            inner = expr.expr.accept(self)
            interval = Interval(- inner.end, - inner.start)
        else:
            raise ValueError("Unsupported op type {}.".format(expr.type))
        return interval

    def visit_var_expr(self, expr):
        return Interval(expr, expr)

    def visit_const_expr(self, expr):
        return Interval(expr, expr)

    def visit_iter_expr(self, expr):
        # convert to closed closed interval
        interval = self.rmap[expr].convert_to_interval()
        interval.end = expr_simplifier.rewrite(interval.end)
        if expr in self.constraint_map:
            interval = intersect_interval(interval, self.constraint_map[expr])
        if expr in self.relax_set:
            interval = Interval(interval.start.accept(self).start, interval.end.accept(self).end)
        return interval

bound_evaluator = BoundEvaluator()