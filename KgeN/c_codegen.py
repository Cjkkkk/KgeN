from .utils import index_flatten
from .tir import *
from .ir_printer import IR_Printer
from .te import *
from .schedule import *

# codegen
class C_code_generator(IR_Printer):
    def __init__(self):
        super().__init__()
        self.scope = 0
        self.res = ""
        self.op_mapping = ["+", "*", "/", "/", "-", "%", ">", ">=", "<", "<=", "min", "max", "ceildiv", "&&", "||", "!", "-"]

    def generate_signature(self, func_stmt):
        self.emit("void kernel({}) {{".format(", ".join([tensor.dtype + "* " + tensor.name for tensor in func_stmt.input_tensors + func_stmt.output_tensors])))

    def generate_storage(self, func_stmt):
        from functools import reduce
        for tensor in func_stmt.storage:
            if tensor.scope == "local":
                self.emit("{0} {1}[{2}];".format(tensor.dtype, tensor.name, reduce(lambda x, y: x * y, tensor.shape).accept(self)))
    
    def visit_func_stmt(self, stmt):
        self.generate_tensor_shape(stmt)
        self.generate_signature(stmt)
        self.enter_scope()
        self.generate_storage(stmt)
        
        for st in stmt.body:
            st.accept(self)
        
        self.exit_scope()
        self.emit("}")
    
    def visit_for_stmt(self, stmt):
        var = stmt.iter_var
        if not var.range.is_single_point and var.bind_to is None:
            if var.type == IterVar.UNROLL:
                self.emit("#pragma unroll")
            self.emit("for (int {0} = {1}; {0} < {2} ; {0} += {3}) {{".format(
                var.name, 
                var.range.start.accept(self),
                var.range.end.accept(self),
                var.range.stride.accept(self)))
            self.enter_scope()
            
        for st in stmt.body:
            st.accept(self)
        
        if not var.range.is_single_point and var.bind_to is None:
            self.exit_scope()
            self.emit("}")
    
    def visit_tensor_slice_expr(self, expr):
        flatten_index = index_flatten(expr.index, expr.tensor.shape)
        return expr.tensor.name + "[" + flatten_index.accept(self) + "]" 

def C_codegen_pass(func):
    code_generator = C_code_generator()
    return code_generator.generate(func)