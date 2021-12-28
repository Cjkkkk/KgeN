from KgeN.te.bound import infer_bound_pass, set_tensor_shape_pass
from KgeN.te.gen_func import gen_func_pass
from KgeN.target.cuda_codegen import CUDA_codegen_pass
from KgeN.target.c_codegen import C_codegen_pass
from KgeN.target.target import Target
from KgeN.tir.transform import sync_analysis_pass, inline_injection_pass, expand_pass, vthread_injection_pass, bound_checker_pass
from KgeN.te.operation import ComputeOp, PlaceholderOp
from KgeN.te.build_graph import create_feed_graph, create_attach_path


def lower(schdule, bufs):
    inputs = [t for t in bufs if isinstance(t.op, PlaceholderOp)]
    outputs = [t for t in bufs if isinstance(t.op, ComputeOp)]

    assert len(outputs) == 1, "only support one output."
    for output in outputs:
        output.scope = "global"

    create_feed_graph(schdule)
    inline_injection_pass(schdule)
    create_attach_path(schdule)
    infer_bound_pass(schdule)
    set_tensor_shape_pass(schdule)
    
    func = gen_func_pass(schdule, inputs, outputs)
    func = expand_pass(func)
    func = vthread_injection_pass(func)
    func = bound_checker_pass(func)
    return func

def build(func, target=Target.CUDA):
    if target == Target.CUDA:
        func = sync_analysis_pass(func)
        return CUDA_codegen_pass(func)
    elif target == Target.C:
        return C_codegen_pass(func)
    else:
        raise ValueError("Unsupported target: {0}".format(target))