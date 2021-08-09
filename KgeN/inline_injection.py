def inline_injection_pass(schdule):
    tensors = schdule.tensors
    for tensor in tensors:
        stage = schdule[tensor]
        if stage.is_inline:
            for output in tensor.outputs:
                stage.compute_at(output, schdule[output].leaf_axis[-1])