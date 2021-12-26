def inline_injection_pass(schedule):
    for stage in schedule.stages:
        if stage.is_inline:
            for output in schedule.feed_graph[stage]:
                stage.compute_at(output, output.leaf_axis[-1])