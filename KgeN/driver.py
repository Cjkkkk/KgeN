from .bound import infer_bound_pass, check_bound_pass
from .gen_func import gen_func_pass
from .codegen import CUDA_codegen_pass
from .utils import *

def lower(tensor):
    tensors = tensor_topo_sort_bottom_up(tensor)
    infer_bound_pass(tensors)
    check_bound_pass(tensors)
    func = gen_func_pass(tensors)
    return CUDA_codegen_pass(func)