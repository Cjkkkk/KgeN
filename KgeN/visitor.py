from .tir import *


class Visitor:        
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