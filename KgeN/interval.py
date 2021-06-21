from .tir import *
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