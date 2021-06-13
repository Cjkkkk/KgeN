from .bound import infer_bound_pass, check_bound_pass
from .gen_func import gen_func_pass
from .codegen import CUDA_codegen_pass
from .build_graph import build_graph_pass
from .sync_analysis import sync_analysis_pass
from .utils import tensor_topo_sort_bottom_up
from .inline_injection import inline_injection_pass

def lower(tensor):
    build_graph_pass(tensor)
    tensors = tensor_topo_sort_bottom_up(tensor)
    
    inline_injection_pass(tensors)
    infer_bound_pass(tensors)
    check_bound_pass(tensors)

    func = gen_func_pass(tensors)
    func = sync_analysis_pass(func)
    return func

def build(func):
    return CUDA_codegen_pass(func)