from .utils import index_flatten
from .tir import *
from .ir_printer import IR_Printer
from .te import *
from .schedule import *

# codegen
class CUDA_code_generator(IR_Printer):
    def __init__(self):
        super().__init__()
        self.scope = 0
        self.res = ""
        self.op_mapping = ["+", "*", "/", "/", "-", "%", ">", ">=", "<", "<=", "min", "max", "ceildiv", "&&", "||", "!", "-"]

    def generate_signature(self, func_stmt):
        self.emit("__global__ void kernel({}) {{".format(", ".join([tensor.dtype + "* " + tensor.name for tensor in func_stmt.input_tensors + func_stmt.output_tensors])))
    
    def generate_lanuch_config(self, func_stmt):
        # TODO: fix this: assume only one output
        grid_dim = {
            "blockIdx.x": 1,
            "blockIdx.y": 1, 
            "blockIdx.z": 1
        }
        block_dim = {
            "threadIdx.x": 1, 
            "threadIdx.y": 1,
            "threadIdx.z": 1
        }
        output = func_stmt.schedule[func_stmt.output_tensors[0]]
        for axis in output.leaf_axis:
            if axis.bind_to is not None:
                if axis.bind_to.name in block_dim:
                    block_dim[axis.bind_to.name] = axis.range.end.val
                if axis.bind_to.name in grid_dim:
                    grid_dim[axis.bind_to.name] = axis.range.end.val
        
        self.emit("// gridDim: [{0}, {1}, {2}]".format(*grid_dim.values()))
        self.emit("// blockDim: [{0}, {1}, {2}]".format(*block_dim.values()))

    def generate_storage(self, func_stmt):
        from functools import reduce
        for tensor in func_stmt.storage:
            if tensor.scope == "local":
                self.emit("{0} {1}[{2}];".format(tensor.dtype, tensor.name, reduce(lambda x, y: x * y, tensor.shape).accept(self)))
            elif tensor.scope == "shared":
                self.emit("__shared__ {0} {1}[{2}];".format(tensor.dtype, tensor.name, reduce(lambda x, y: x * y, tensor.shape).accept(self)))
    
    def visit_func_stmt(self, stmt):
        self.generate_tensor_shape(stmt)
        self.generate_lanuch_config(stmt)
        self.generate_signature(stmt)
        self.enter_scope()
        self.generate_storage(stmt)
        
        for st in stmt.body:
            st.accept(self)
        
        self.exit_scope()
        self.emit("}")
    
    def visit_for_stmt(self, stmt):
        var = stmt.iter_var
        if stmt.need_sync_before:
            self.emit("__syncthreads();")
        if not var.range.is_single_point and var.bind_to is None:
            if var.type == IterVar.UNROLL:
                self.emit("#pragma unroll")
            self.emit("for (int {0} = {1}; {0} < {2} ; {0} += 1) {{".format(
                var.name, 
                var.range.start.accept(self),
                var.range.end.accept(self)))
            self.enter_scope()
            
        for st in stmt.body:
            st.accept(self)
        
        if not var.range.is_single_point and var.bind_to is None:
            self.exit_scope()
            self.emit("}")
        if stmt.need_sync_after:
            self.emit("__syncthreads();")
    
    def visit_tensor_slice_expr(self, expr):
        flatten_index = index_flatten(expr.index, expr.tensor.shape)
        return expr.tensor.name + "[" + flatten_index.accept(self) + "]" 
        # return expr.tensor.name + "[" + ", ".join([index.accept(self) for index in expr.index]) + "]" 

def CUDA_codegen_pass(func):
    code_generator = CUDA_code_generator()
    return code_generator.generate(func)