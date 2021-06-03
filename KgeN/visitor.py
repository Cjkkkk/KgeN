from .tir import *


class Visitor:
    def visit_func_stmt(self, stmt):
        raise NotImplementedError

    def visit_for_stmt(self, stmt):
        raise NotImplementedError
    
    def visit_assign_stmt(self, stmt):
        raise NotImplementedError
            
    def visit_binary_expr(self, expr):
        raise NotImplementedError
    
    def visit_unary_expr(self, expr):
        raise NotImplementedError

    def visit_var_expr(self, expr):
        raise NotImplementedError

    def visit_const_expr(self, expr):
        raise NotImplementedError

    def visit_iter_expr(self, expr):
        raise NotImplementedError

    def visit_if_then_else_expr(self, expr):
        raise NotImplementedError
    
    def visit_reduce_expr(self, expr):
        raise NotImplementedError

    def visit_tensor_expr(self, expr):
        raise NotImplementedError

    def visit_tensor_slice_expr(self, expr):
        raise NotImplementedError


class RewriteExprVisitor(Visitor):
    def __init__(self):
        super().__init__()
    
    def rewrite(self, expr):
        expr = expr.accept(self)
        return expr
    
    def visit_binary_expr(self, expr):
        expr.left = expr.left.accept(self)
        expr.right = expr.right.accept(self)
        return expr

    def visit_if_then_else_expr(self, expr):
        expr.condition = expr.condition.accept(self)
        expr.then_expr = expr.then_expr.accept(self)
        expr.else_expr = expr.else_expr.accept(self)
        return expr
    
    def visit_reduce_expr(self, expr):
        expr.expr = expr.expr.accept(self)
        return expr

    def visit_tensor_slice_expr(self, expr):
        expr.tensor = expr.tensor.accept(self)
        new_idx = []
        for index in expr.index:
            new_idx.append(index.accept(self))
        expr.index = tuple(new_idx)
        return expr
    
    def visit_unary_expr(self, expr):
        expr.expr = expr.expr.accept(self)
        return expr

    def visit_var_expr(self, expr):
        return expr

    def visit_const_expr(self, expr):
        return expr

    def visit_iter_expr(self, expr):
        return expr

    def visit_tensor_expr(self, expr):
        return expr


class CollectExprVisitor(Visitor):
    def __init__(self):
        super().__init__()

    def collect(self, expr):
        expr.accept(self)
            
    def visit_binary_expr(self, expr):
        expr.left.accept(self)
        expr.right.accept(self)

    def visit_if_then_else_expr(self, expr):
        expr.condition.accept(self)
        expr.then_expr.accept(self)
        expr.else_expr.accept(self)
    
    def visit_reduce_expr(self, expr):
        expr.expr.accept(self)

    def visit_tensor_slice_expr(self, expr):
        expr.tensor.accept(self)
        for index in expr.index:
            index.accept(self)
    
    def visit_unary_expr(self, expr):
        expr.expr.accept(self)

    def visit_var_expr(self, expr):
        pass

    def visit_const_expr(self, expr):
        pass

    def visit_iter_expr(self, expr):
        pass

    def visit_tensor_expr(self, expr):
        pass