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