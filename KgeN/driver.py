from .te import placeholder
from .bound import infer_bound_pass, check_bound_pass
from .gen_func import gen_func_pass
from .cuda_codegen import CUDA_codegen_pass
from .build_graph import build_graph_pass
from .sync_analysis import sync_analysis_pass
from .utils import tensor_topo_sort_bottom_up
from .inline_injection import inline_injection_pass
from .expand import expand_pass
from .tir import TensorExpr

def lower(bufs):
    inputs = [t for t in bufs if t.type == TensorExpr.PLACEHOLDER]
    outputs = [t for t in bufs if t.type == TensorExpr.COMPUTE]

    assert(len(outputs) == 1, "only support one output.")
    for output in outputs:
        output.scope = "global"
    
    build_graph_pass(outputs[0])
    tensors = tensor_topo_sort_bottom_up(outputs[0])

    inline_injection_pass(tensors)
    infer_bound_pass(tensors)
    check_bound_pass(tensors)

    func = gen_func_pass(inputs, outputs, tensors)
    func = expand_pass(func)
    func = sync_analysis_pass(func)
    return func

def build(func):
    return CUDA_codegen_pass(func)