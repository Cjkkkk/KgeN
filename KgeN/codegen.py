from .bound import infer_bound_pass, check_bound_pass, tensor_topo_sort
from .tir import TensorSliceExpr

# codegen
def CUDA_codegen_pass(tensor):
    tensors = tensor_topo_sort(tensor)
    for tensor in tensors:
        print("buffer: {0}".format(TensorSliceExpr(tensor, tensor.shape)))
    res = ""
    for t in reversed(tensors):
        # skip codegen if it is attached to some axis
        if not t.attached:
            res += t.CUDA_codegen()
    return res

def lower(tensor):
    infer_bound_pass(tensor)
    check_bound_pass(tensor)
    return CUDA_codegen_pass(tensor)