from .tir import *
from .visitor import Visitor

class Pattern(Expr):
    def __init__(self, cls_):
        self.is_matched = False
        self.expr = None
        self.cls = cls_
    
    def match(self, other):
        if self.is_matched:
            return self.expr.same_as(other)
        else:      
            if isinstance(other, self.cls):      
                self.is_matched = True
                self.expr = other
                return True
            else:
                return False

    def reset(self):
        self.is_matched = False
        self.expr = None

    def same_as(self, other):
        return False

def real_match(expr, pattern):
    if isinstance(pattern, BinaryExpr):
        if isinstance(expr, BinaryExpr):
            if pattern.type != expr.type:
                return False
            ans_left = real_match(expr.left, pattern.left)
            ans_right = real_match(expr.right, pattern.right)
            return ans_left and ans_right
        else:
            return False
    else:
        return pattern.match(expr)

def eval_result(result):
    if isinstance(result, BinaryExpr):
        return Expr.function_mapping[result.type](
            eval_result(result.left), 
            eval_result(result.right))
    elif isinstance(result, Pattern):
        return result.expr
    else:
        return result

def rewrite_recursive(expr, pattern, result):
    while True:
        new_expr = rewrite(expr, pattern, result)
        if new_expr.same_as(expr):
            break
        expr = new_expr
    return expr

def rewrite_recursive_if(expr, pattern, result, arg, condition):
    while True:
        new_expr = rewrite_if(expr, pattern, result, arg, condition)
        if new_expr.same_as(expr):
            break
        expr = new_expr
    return expr

def rewrite_if(expr, pattern, result, arg, condition):
    ans = real_match(expr, pattern)
    if ans and condition(arg):
        expr = eval_result(result)
    reset_pattern(pattern)
    return expr

def rewrite(expr, pattern, result):
    ans = real_match(expr, pattern)
    if ans:
        expr = eval_result(result)
    reset_pattern(pattern)
    return expr

def reset_pattern(pattern):
    if isinstance(pattern, BinaryExpr):
        reset_pattern(pattern.left)
        reset_pattern(pattern.right)
    else:
        pattern.reset()


class Expr_Simpifier(Visitor):        
    def visit_binary_expr(self, expr):
        C1 = Pattern(ConstExpr)
        C2 = Pattern(ConstExpr)
        V1 = Pattern(Expr)
        V2 = Pattern(Expr)
        old_expr = expr
        while True:
            if isinstance(expr, BinaryExpr):
                expr.left = self.visit(expr.left)
                expr.right = self.visit(expr.right)
                if expr.type == Expr.ADD:
                    expr = rewrite(expr, (V1 + C1) + C2, V1 + (C1 + C2))
                    expr = rewrite(expr, (C1 + V1) + C2, V1 + (C1 + C2))
                    expr = rewrite(expr, (V1 - C1) + C2, V1 + (C2 - C1))
                    expr = rewrite(expr, (C1 - V1) + C2, V1 + (C1 - C2) - V1)
                    expr = rewrite(expr, C1 + C2, C1 + C2)
                elif expr.type == Expr.SUB:
                    expr = rewrite(expr, (V1 + V2) - V1, V2)
                    expr = rewrite(expr, (V2 + V1) - V1, V2)
                    expr = rewrite(expr, V1 - V1, ConstExpr(0))
                elif expr.type == Expr.MIN:
                    expr = rewrite_if(expr, Expr.min(V1 + C1, V1), V1, C1, lambda x: x.expr.val > 0)
                    expr = rewrite_if(expr, Expr.min(V1 - C1, V1), V1 - C1, C1, lambda x: x.expr.val > 0)
                elif expr.type == Expr.MAX:
                    expr = rewrite_if(expr, Expr.max(V1 + C1, V1), V1 + C1, C1, lambda x: x.expr.val > 0)
                    expr = rewrite_if(expr, Expr.max(V1 - C1, V1), V1, C1, lambda x: x.expr.val > 0)
                if old_expr.same_as(expr):
                    break
                else:
                    old_expr = expr
            else:
                break
        return expr
    
    def visit_unary_expr(self, expr):
        raise NotImplemented

    def visit_var_expr(self, expr):
        return expr

    def visit_const_expr(self, expr):
        return expr

    def visit_iter_expr(self, expr):
        return expr

    def visit_if_then_else_expr(self, expr):
        expr.condition = self.visit(expr.condition)
        expr.then_expr = self.visit(expr.then_expr)
        expr.else_expr = self.visit(expr.else_expr)
        return expr
    
    def visit_reduce_expr(self, expr):
        raise NotImplemented

    def visit_tensor_expr(self, expr):
        return expr

    def visit_tensor_slice_expr(self, expr):
        new_idx = []
        for index in expr.index:
            new_idx.append(self.visit(index))
            expr.index = tuple(new_idx)
        return expr

expr_simpifier = Expr_Simpifier()
        
if __name__ == "__main__":
    C1 = Pattern(ConstExpr)
    C2 = Pattern(ConstExpr)
    V1 = Pattern(Expr)
    V2 = Pattern(Expr)
    expr = IterVar("x") + ConstExpr(1) + ConstExpr(10) + ConstExpr(100)
    expr = rewrite_recursive(expr, (V1 + C1) + C2, V1 + (C1 + C2))
    print(expr)

    expr = ((IterVar("x") + ConstExpr(1)) + ConstExpr(10)) - (IterVar("x") + ConstExpr(1))
    expr = rewrite_recursive(expr, (V1 + V2) - V1, V2)
    print(expr)

    expr = Expr.min(IterVar("x") + ConstExpr(1), IterVar("x"))
    expr = rewrite_recursive_if(expr, Expr.min(V1 + C1, V1), V1, C1, lambda x: x.expr.val > 0)
    print(expr)