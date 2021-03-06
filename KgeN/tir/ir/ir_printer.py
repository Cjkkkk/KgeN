from .expr import *
from .visitor import CollectVisitor
from KgeN.te import *

# codegen
class IR_Printer(CollectVisitor):
    def __init__(self):
        super().__init__()
        self.scope = 0
        self.res = ""
        self.op_mapping = ["+", "*", "//", "/", "-", "%", ">", ">=", "<", "<=", "min", "max", "ceildiv", "&&", "||", "!", "-"]
    
    def emit(self, str):
        self.res += self.scope * "    " + str + "\n"
    
    def emit_newline(self):
        self.res += "\n"

    def enter_scope(self):
        self.scope += 1

    def exit_scope(self):
        self.scope -= 1
    
    def generate(self, func):
        func.accept(self)
        return self.res

    def generate_signature(self, func_stmt):
        self.emit("func kernel({}) {{".format(", ".join([tensor.dtype + "* " + tensor.name for tensor in func_stmt.input_tensors + func_stmt.output_tensors])))
    
    def generate_tensor_shape(self, func_stmt):
        for stage in func_stmt.schedule.stages:
            tensor = stage.op.outputs[0]
            self.emit("// tensor: {0}".format(TensorSliceExpr(tensor, tensor.shape)))
    
    def visit_func_stmt(self, stmt):
        self.generate_tensor_shape(stmt)
        self.generate_signature(stmt)
        self.enter_scope()
        
        for st in stmt.body:
            st.accept(self)
        
        self.exit_scope()
        self.emit("}")

    def visit_if_stmt(self, stmt):
        self.emit("if ({0}) {{".format(stmt.condition.accept(self)))
        self.enter_scope()
        stmt.then_stmt.accept(self)
        self.exit_scope()
        if stmt.else_stmt:
            self.emit("} else {")
            self.enter_scope()
            stmt.else_stmt.accept(self)
            self.exit_scope()
        self.emit("}")
    
    def visit_assign_stmt(self, stmt):
        self.emit(stmt.dest.accept(self) + " = " + stmt.source.accept(self) + ";")
    
    def visit_for_stmt(self, stmt):
        var = stmt.iter_var
        if not var.range.is_single_point and var.bind_to is None:
            self.emit("for ({0}: {1}, {2}) {{".format(
                var.name, 
                var.range.start.accept(self),
                var.range.end.accept(self)))
            self.enter_scope()
            
        for st in stmt.body:
            st.accept(self)
        
        if not var.range.is_single_point and var.bind_to is None:
            self.exit_scope()
            self.emit("}")

    def visit_binary_expr(self, expr):
        if expr.type == Expr.MIN or expr.type == Expr.MAX or expr.type == Expr.CEIL_DIV: # min, max, ceil_div
            return "({1}({0}, {2}))".format(expr.left.accept(self), self.op_mapping[expr.type], expr.right.accept(self))
        return "({0} {1} {2})".format(expr.left.accept(self), self.op_mapping[expr.type], expr.right.accept(self))
    
    def visit_unary_expr(self, expr):
        return "({0}({1}))".format(self.op_mapping[expr.type], expr.expr.accept(self))

    def visit_var_expr(self, expr):
        return expr.name

    def visit_const_expr(self, expr):
        return str(expr.val)

    def visit_iter_expr(self, expr):
        if expr.range.is_single_point:
            return expr.range.start.accept(self)
        elif expr.bind_to is not None:
            return expr.bind_to.name
        elif expr.relation == IterVar.SPLIT:
            return "(({0} * {1}) + {2})".format(expr.split_outer.accept(self), expr.split_inner.range.end.accept(self), expr.split_inner.accept(self))
        elif expr.relation == IterVar.FUSE:
            if expr is expr.fused.fused_outer:
                return "({0} // {1})".format(expr.fused.accept(self), expr.fused.fused_inner.range.end.accept(self))
            else:
                return "({0} % {1})".format(expr.fused.accept(self), expr.fused.fused_inner.range.end.accept(self))
        else:
            return expr.name

    def visit_if_then_else_expr(self, expr):
        return "({0} ? {1} : {2})".format(expr.condition.accept(self), expr.then_expr.accept(self), expr.else_expr.accept(self))
    
    def visit_tensor_slice_expr(self, expr): 
        return expr.tensor.name + "[" + ", ".join([index.accept(self) for index in expr.index]) + "]" 

    def visit_tensor_expr(self, expr):
        return expr.name