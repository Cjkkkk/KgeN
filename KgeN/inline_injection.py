from .schedule import compute_at

def inline_injection_pass(tensors):
    for tensor in tensors:
        if tensor.is_inline:
            for output in tensor.outputs:
                compute_at(tensor, output, output.axis[-1])