from .tir import *


class Visitor:
    def visit(self, expr, *args, **kargs):
        if isinstance(expr, BinaryExpr):
            return self.visit_binary_expr(expr, *args, **kargs)
        elif isinstance(expr, UnaryExpr):
            return self.visit_unary_expr(expr, *args, **kargs)
        elif isinstance(expr, VarExpr):
            return self.visit_var_expr(expr, *args, **kargs)
        elif isinstance(expr, ConstExpr):
            return self.visit_const_expr(expr, *args, **kargs)
        elif isinstance(expr, IterVar):
            return self.visit_iter_expr(expr, *args, **kargs)
        elif isinstance(expr, IfThenElseExpr):
            return self.visit_if_then_else_expr(expr, *args, **kargs)
        elif isinstance(expr, ReduceExpr):
            return self.visit_reduce_expr(expr, *args, **kargs)
        elif isinstance(expr, TensorExpr):
            return self.visit_tensor_expr(expr, *args, **kargs)
        elif isinstance(expr, TensorSliceExpr):
            return self.visit_tensor_slice_expr(expr, *args, **kargs)
        else:
            raise ValueError("unsupported expr type: " + type(expr))
        
    def visit_binary_expr(self, expr, *args, **kargs):
        raise NotImplemented
    
    def visit_unary_expr(self, expr, *args, **kargs):
        raise NotImplemented

    def visit_var_expr(self, expr, *args, **kargs):
        raise NotImplemented

    def visit_const_expr(self, expr, *args, **kargs):
        raise NotImplemented

    def visit_iter_expr(self, expr, *args, **kargs):
        raise NotImplemented

    def visit_if_then_else_expr(self, expr, *args, **kargs):
        raise NotImplemented
    
    def visit_reduce_expr(self, expr, *args, **kargs):
        raise NotImplemented

    def visit_tensor_expr(self, expr, *args, **kargs):
        raise NotImplemented

    def visit_tensor_slice_expr(self, expr, *args, **kargs):
        raise NotImplemented