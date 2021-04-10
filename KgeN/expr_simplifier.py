from .tir import *

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
    else:
        return result.expr

def rewrite_recursive(expr, pattern, result):
    while True:
        new_expr = rewrite(expr, pattern, result)
        if new_expr.same_as(expr):
            break
        expr = new_expr

    if isinstance(expr, BinaryExpr):
        expr.left = rewrite_recursive(expr.left, pattern, result)
        expr.right = rewrite_recursive(expr.right, pattern, result)
    return expr

def reset_pattern(pattern):
    if isinstance(pattern, BinaryExpr):
        reset_pattern(pattern.left)
        reset_pattern(pattern.right)
    else:
        pattern.reset()

def rewrite(expr, pattern, result):
    ans = real_match(expr, pattern)
    if ans:
        expr = eval_result(result)
    reset_pattern(pattern)
    return expr

if __name__ == "__main__":
    C1 = Pattern(ConstExpr)
    C2 = Pattern(ConstExpr)
    V1 = Pattern(Expr)
    V2 = Pattern(Expr)
    expr = IterVar("x", 0, 0) + ConstExpr(1) + ConstExpr(10) + ConstExpr(100)
    expr = rewrite_recursive(expr, (V1 + C1) + C2, V1 + (C1 + C2))
    print(expr)


    expr = ((IterVar("x", 0, 0) + ConstExpr(1)) + ConstExpr(10)) - (IterVar("x", 0, 0) + ConstExpr(1))
    expr = rewrite_recursive(expr, (V1 + V2) - V1, V2)
    print(expr)