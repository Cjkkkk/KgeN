from .tir import *


class Visitor:
    def visit_func_stmt(self, stmt):
        raise NotImplemented

    def visit_for_stmt(self, stmt):
        raise NotImplemented
    
    def visit_assign_stmt(self, stmt):
        raise NotImplemented
            
    def visit_binary_expr(self, expr):
        raise NotImplemented
    
    def visit_unary_expr(self, expr):
        raise NotImplemented

    def visit_var_expr(self, expr):
        raise NotImplemented

    def visit_const_expr(self, expr):
        raise NotImplemented

    def visit_iter_expr(self, expr):
        raise NotImplemented

    def visit_if_then_else_expr(self, expr):
        raise NotImplemented
    
    def visit_reduce_expr(self, expr):
        raise NotImplemented

    def visit_tensor_expr(self, expr):
        raise NotImplemented

    def visit_tensor_slice_expr(self, expr):
        raise NotImplemented