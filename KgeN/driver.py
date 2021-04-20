from .bound import infer_bound_pass, check_bound_pass
from .codegen import CUDA_codegen_pass

def lower(tensor):
    infer_bound_pass(tensor)
    check_bound_pass(tensor)
    return CUDA_codegen_pass(tensor)