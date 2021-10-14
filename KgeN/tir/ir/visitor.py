from .expr import *
from .stmt import *

class Visitor:
    def visit_func_stmt(self, stmt):
        raise NotImplementedError

    def visit_if_stmt(self, stmt):
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


class RewriteVisitor(Visitor):
    def __init__(self):
        super().__init__()
    
    def visit_func_stmt(self, stmt):
        for i in range(len(stmt.body)):
            stmt.body[i] = stmt.body[i].accept(self)
        return stmt

    def visit_if_stmt(self, stmt):
        stmt.condition = stmt.condition.accept(self)
        stmt.then_stmt = stmt.then_stmt.accept(self)
        if stmt.else_stmt:
            stmt.else_stmt = stmt.else_stmt.accept(self)
        return stmt

    def visit_assign_stmt(self, stmt):
        stmt.dest = stmt.dest.accept(self)
        stmt.source = stmt.source.accept(self)
        return stmt

    def visit_for_stmt(self, stmt):
        for i in range(len(stmt.body)):
            stmt.body[i] = stmt.body[i].accept(self)
        return stmt
    
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


class CollectVisitor(Visitor):
    def __init__(self):
        super().__init__()
    
    def visit_func_stmt(self, stmt):
        for st in stmt.body:
            st.accept(self)

    def visit_if_stmt(self, stmt):
        stmt.condition.accept(self)
        stmt.then_stmt.accept(self)
        if stmt.else_stmt:
            stmt.else_stmt.accept(self)
    
    def visit_assign_stmt(self, stmt):
        stmt.dest.accept(self)
        stmt.source.accept(self)

    def visit_for_stmt(self, stmt):
        for st in stmt.body:
            st.accept(self)
    
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