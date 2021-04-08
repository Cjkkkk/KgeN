from .bound import infer_bound_pass, tensor_topo_sort

# codegen
def CUDA_codegen_pass(tensor):
    tensors = tensor_topo_sort(tensor)
    res = ""
    for t in reversed(tensors):
        # skip codegen if it is attached to some axis
        if not t.attached:
            res += t.CUDA_codegen()
    return res

def lower(tensor):
    infer_bound_pass(tensor)
    return CUDA_codegen_pass(tensor)