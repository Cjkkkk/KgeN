from KgeN.tir.ir import ConstExpr, Expr, RewriteVisitor
from KgeN.arith.interval import Interval, union_interval, intersect_interval
import math

class ConstraintEvaluator(RewriteVisitor):
    def __init__(self):
        super().__init__()
    
    def evaluator(self, expr):
        return expr.accept(self)
    
    def visit_binary_expr(self, expr):
        if expr.type == Expr.AND:
            left_map = expr.left.accept(self)
            right_map = expr.right.accept(self)
            for var in left_map:
                if var in right_map:
                    left_map[var] = intersect_interval(left_map[var], right_map[var])
            for var in right_map:
                if var not in left_map:
                    left_map[var] = right_map[var]
            return left_map
        elif expr.type == Expr.OR:
            left_map = expr.left.accept(self)
            right_map = expr.right.accept(self)
            for var in left_map:
                if var in right_map:
                    left_map[var] = union_interval(left_map[var], right_map[var])
            for var in right_map:
                if var not in left_map:
                    left_map[var] = right_map[var]
            return left_map
        elif expr.type == Expr.GT:
            return {expr.left: Interval(expr.right + 1, math.inf)}
        elif expr.type == Expr.GE:
            return {expr.left: Interval(expr.right, math.inf)}
        elif expr.type == Expr.LT:
            return {expr.left: Interval(-math.inf, expr.right - 1)}
        elif expr.type == Expr.GT:
            return {expr.left: Interval(-math.inf, expr.right)}
        else:
            raise ValueError("unsupported type.")
        
    def visit_unary_expr(self, expr):
        if expr.type == Expr.NOT:
            map = expr.expr.accept(self)
            for var in map:
                if map[var].is_everything:
                    map[var] = Interval.nothing()
                elif map[var].start.same_as(ConstExpr(-math.inf)):
                    map[var] = Interval(map[var].end + 1, math.inf)
                elif map[var].end.same_as(ConstExpr(math.inf)):
                    map[var] = Interval(-math.inf, map[var].start - 1)
                else:
                    # TODO: fix this
                    # right now not [5, 10] return (-inf, inf) instead of (-inf, 5) (10, inf)
                    # because we don't have interval set
                    map[var] = Interval.everything()
            return map
        else:
            raise ValueError("unsupported type.")

constraint_evaluator = ConstraintEvaluator()