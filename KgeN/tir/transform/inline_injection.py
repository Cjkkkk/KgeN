def inline_injection_pass(schdule):
    for stage in schdule.stages:
        if stage.is_inline:
            for output in stage.outputs:
                stage.compute_at(output, output.leaf_axis[-1])