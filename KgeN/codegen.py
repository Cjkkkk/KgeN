from .utils import tensor_topo_sort_bottom_up
from .tir import TensorSliceExpr

# codegen
def CUDA_codegen_pass(tensor):
    tensors = tensor_topo_sort_bottom_up(tensor)
    for tensor in tensors:
        print("buffer: {0}".format(TensorSliceExpr(tensor, tensor.shape)))
    res = ""
    for t in reversed(tensors):
        # skip codegen if it is attached to some axis
        if not t.attached:
            res += t.CUDA_codegen()
    return res