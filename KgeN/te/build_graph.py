from KgeN.tir.ir import CollectVisitor


class CollectInputVisitor(CollectVisitor):
    def __init__(self):
        super().__init__()
        self.inputs = set()
        self.providers = {}

    def collect(self, expr):
        expr.accept(self)
        return list(self.inputs), self.providers
    
    def visit_tensor_slice_expr(self, expr):
        if expr.tensor in self.providers:
            self.providers[expr.tensor].append(expr)
        else:
            self.providers[expr.tensor] = [expr]
        self.inputs.add(expr.tensor)
        for index in expr.index:
            index.accept(self)


def create_feed_graph(schedule):
    # stage: [stage]
    feed_graph = {}
    for stage in schedule.stages:
        for provider in stage.op.providers:
            if provider in feed_graph:
                feed_graph[schedule[provider]].append(stage)
            else:
                feed_graph[schedule[provider]] = [stage]
    schedule.feed_graph = feed_graph


def create_attach_path(schedule):
    # stage: [axis]
    attach_path = {}
    for stage in schedule.stages:
        cur_stage = stage
        path = []
        while cur_stage.attached:
            cur_attach_path = []
            for axis in cur_stage.attach_at.leaf_axis:
                cur_attach_path.append(axis)
                if axis is cur_stage.attach_axis:
                    path += reversed(cur_attach_path)
                    break
            cur_stage = cur_stage.attach_at
        attach_path[stage] = tuple(path)
    schedule.attach_path = attach_path


def build_context(schedule):
    create_feed_graph(schedule)
    create_attach_path(schedule)


# def build_graph(op):
#     visited = {op}
#     q = [op]

#     while len(q) > 0:
#         op = q.pop()
#         if isinstance(op, ComputeOp):
#             visitor = CollectInputVisitor()
#             op.inputs, op.providers = visitor.collect(op.expr)
#             for inp in op.inputs:
#                 if inp.op not in visited:
#                     visited.add(inp.op)
#                     q.append(inp.op)
