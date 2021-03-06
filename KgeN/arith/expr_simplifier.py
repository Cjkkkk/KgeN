from KgeN.tir.ir import *

class Pattern(Expr):
    def __init__(self, cls_):
        self.is_matched = False
        self.expr = None
        self.cls = cls_
    
    def match(self, other):
        if self.is_matched:
            return self.expr.same_as(other)
        elif isinstance(other, self.cls):      
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
    if isinstance(expr, IterVar):
        # TODO: fix this
        if expr.range.is_single_point:
            expr = expr.range.start
    if isinstance(pattern, BinaryExpr):
        if isinstance(expr, BinaryExpr):
            if pattern.type != expr.type:
                return False
            ans_left = real_match(expr.left, pattern.left)
            ans_right = real_match(expr.right, pattern.right)
            
            # automatically try 
            # if not (ans_left and ans_right) and Expr.is_commutative[expr.type]:
            #     reset_pattern(pattern)
            #     ans_left = real_match(expr.left, pattern.right)
            #     ans_right = real_match(expr.right, pattern.left)
            return ans_left and ans_right
        else:
            return False
    elif isinstance(pattern, UnaryExpr):
        if isinstance(expr, UnaryExpr):
            if pattern.type != expr.type:
                return False
            ans = real_match(expr.expr, pattern.expr)
            return ans
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

def rewrite_recursive_if(expr, pattern, result, condition):
    while True:
        new_expr = rewrite_if(expr, pattern, result, condition)
        if new_expr.same_as(expr):
            break
        expr = new_expr
    return expr

def rewrite_if(expr, pattern, then, condition):
    ans = real_match(expr, pattern)
    if ans and condition():
        expr = eval_result(then)
    reset_pattern(pattern)
    return expr

def rewrite_if_else(expr, pattern, then, else_, condition):
    ans = real_match(expr, pattern)
    if ans:
        if condition():
            expr = eval_result(then)
        else:
            expr = eval_result(else_)
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
    elif isinstance(pattern, UnaryExpr):
        reset_pattern(pattern.expr)
    else:
        pattern.reset()


class ExprSimplifier(RewriteVisitor):  
    def __init__(self):
        super().__init__()
    
    def rewrite(self, expr):
        expr = expr.accept(self)
        return expr
    
    def visit_binary_expr(self, expr):
        # TODO: move this to __init__ ?
        C1 = Pattern(ConstExpr)
        C2 = Pattern(ConstExpr)
        V1 = Pattern(Expr)
        V2 = Pattern(Expr)
        V3 = Pattern(Expr)
        old_expr = expr
        
        expr.left = expr.left.accept(self)
        expr.right = expr.right.accept(self)
        if expr.type == Expr.ADD:
            expr = rewrite(expr, C1 + V1, V1 + C1)
            expr = rewrite(expr, V1 + C1 + V2, V1 + V2 + C1)
            # expr = rewrite(expr, V1 + (V2 + V3), V1 + V2 + V3)

            # const folding
            expr = rewrite(expr, C1 + C2, C1 + C2)
            expr = rewrite_if(expr, V1 + C1, V1, lambda: C1.expr.val == 0)

            expr = rewrite(expr, V1 + C1 + C2, V1 + (C1 + C2))
            expr = rewrite(expr, (C1 - V1) + C2, (C1 + C2) - V1)
            expr = rewrite(expr, (V1 - C1) + C2, V1 - (C1 - C2))

        elif expr.type == Expr.SUB:
            # const folding
            expr = rewrite(expr, C1 - C2, C1 - C2)
            expr = rewrite_if(expr, V1 - C1, V1, lambda: C1.expr.val == 0)
            
            expr = rewrite(expr, V1 - V1, ConstExpr(0))
            expr = rewrite(expr, (V1 + V2) - (V1 + V3), V2 - V3)
            # expr = rewrite(expr, V1 + (V2 + V3) - V2, V1 + V3)
            expr = rewrite(expr, V1 + V2 + V3 - V1, V2 + V3)
            expr = rewrite(expr, (V1 + V2) - V1, V2)
            expr = rewrite(expr, (V1 + V2) - V2, V1)
            expr = rewrite(expr, (V1 + C1) - C2, V1 + (C1 - C2))
            expr = rewrite(expr, (V1 - C1) - C2, V1 + (C1 + C2))

        elif expr.type == Expr.MUL:
            expr = rewrite(expr, C1 * V1, V1 * C1)
            expr = rewrite(expr, V1 * C1 * V2, V1 * V2 * C1)

            # const folding
            expr = rewrite(expr, C1 * C2, C1 * C2)
            expr = rewrite_if(expr, V1 * C1, V1, lambda: C1.expr.val == 1)
            expr = rewrite_if(expr, V1 * C1, ConstExpr(0), lambda: C1.expr.val == 0)

            expr = rewrite(expr, (V1 + V2) * C1, V1 * C1 + V2 * C1)
            expr = rewrite(expr, (V1 - V2) * C1, V1 * C1 - V2 * C1)
            expr = rewrite(expr, (V1 * C1) * C2, V1 * (C1 * C2))
        
        elif expr.type == Expr.MIN:
            expr = rewrite(expr, Expr.min(V1, V1), V1)
            expr = rewrite_if(expr, Expr.min(V1 + C1, V1), V1, lambda: C1.expr.val > 0)
            expr = rewrite_if(expr, Expr.min(V1, V1 + C1), V1, lambda: C1.expr.val > 0)
            expr = rewrite_if(expr, Expr.min(V1 - C1, V1), V1 - C1, lambda: C1.expr.val > 0)
            expr = rewrite_if(expr, Expr.min(V1, V1 - C1), V1 - C1, lambda: C1.expr.val > 0)
        
        elif expr.type == Expr.MAX:
            expr = rewrite(expr, Expr.max(V1, V1), V1)
            expr = rewrite_if(expr, Expr.max(V1 + C1, V1), V1 + C1, lambda: C1.expr.val > 0)
            expr = rewrite_if(expr, Expr.max(V1, V1 + C1), V1 + C1, lambda: C1.expr.val > 0)
            expr = rewrite_if(expr, Expr.max(V1 - C1, V1), V1, lambda: C1.expr.val > 0)
            expr = rewrite_if(expr, Expr.max(V1, V1 - C1), V1, lambda: C1.expr.val > 0)
        
        elif expr.is_compare():
            expr = rewrite(expr, Expr.function_mapping[expr.type](V1 + C1, C2), Expr.function_mapping[expr.type](V1, C2 - C1))
            expr = rewrite(expr, Expr.function_mapping[expr.type](V1 - C1, C2), Expr.function_mapping[expr.type](V1, C2 + C1))
            expr = rewrite(expr, Expr.function_mapping[expr.type](V1 * C1, C2), Expr.function_mapping[expr.type](V1, C2 / C1))
            expr = rewrite(expr, Expr.function_mapping[expr.type](V1 / C1, C2), Expr.function_mapping[expr.type](V1, C2 * C1))
            
        if old_expr.same_as(expr):
            return expr
        else:
            return expr.accept(self)
    
    def visit_unary_expr(self, expr):
        V1 = Pattern(Expr)
        old_expr = expr
        
        expr.expr = expr.expr.accept(self)
        if expr.type == Expr.NEG:
            expr = rewrite(expr, -(-V1), V1)
        
        if old_expr.same_as(expr):
            return expr
        else:
            return expr.accept(self)

expr_simplifier = ExprSimplifier()
        
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
    expr = rewrite_recursive_if(expr, Expr.min(V1 + C1, V1), V1, lambda: C1.expr.val > 0)
    print(expr)