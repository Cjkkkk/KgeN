from .bound import infer_bound_pass, check_bound_pass
from .gen_func import gen_func_pass
from .cuda_codegen import CUDA_codegen_pass
from .c_codegen import C_codegen_pass
from .sync_analysis import sync_analysis_pass
from .inline_injection import inline_injection_pass
from .expand import expand_pass
from .vthread_injection import vthread_injection_pass
from .tir import TensorExpr
from .target import Target

def lower(schdule, bufs):
    inputs = [t for t in bufs if t.type == TensorExpr.PLACEHOLDER]
    outputs = [t for t in bufs if t.type == TensorExpr.COMPUTE]

    assert len(outputs) == 1, "only support one output."
    for output in outputs:
        output.scope = "global"

    inline_injection_pass(schdule)
    infer_bound_pass(schdule)
    check_bound_pass(schdule)
    
    func = gen_func_pass(schdule, inputs, outputs)
    func = expand_pass(func)
    func = vthread_injection_pass(func)
    return func

def build(func, target=Target.CUDA):
    if target == Target.CUDA:
        func = sync_analysis_pass(func)
        return CUDA_codegen_pass(func)
    elif target == Target.C:
        return C_codegen_pass(func)
    else:
        raise ValueError("Unsupported target: {0}".format(target))