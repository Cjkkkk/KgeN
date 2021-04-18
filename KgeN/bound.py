from .tir import *


# bound inference
def topo_sort(iterable, get_output):
    # Kahn's algorithm
    # calculate indegree map
    indegree_map = {}
    for node in iterable:
        indegree_map[node] = 0
    nodes_list = [*iterable]
    for node in nodes_list:
        out_nodes = get_output(node)
        for out_node in out_nodes:
            nodes_list.append(out_node)
            if out_node not in indegree_map:
                indegree_map[out_node] = 1
            else:
                indegree_map[out_node] += 1
    
    # dequeue node with indegree == 0 into solutions
    nodes_queue = [*iterable]
    solutions = []
    while len(nodes_queue) > 0:
        node = nodes_queue.pop()
        solutions.append(node)
        out_nodes = get_output(node)
        for out_node in out_nodes:
            indegree_map[out_node] -= 1
            if indegree_map[out_node] == 0:
                nodes_queue.append(out_node)
    return solutions

def tensor_topo_sort_bottom_up(tensor):
    def get_output(tensor):
        return tensor.inputs
    res = topo_sort([tensor], get_output)
    return res

def axis_topo_sort_top_down(axis_tuple):
    def get_output(axis):
        if axis.type == IterVar.SPLIT:
            return [axis.outer, axis.inner]
        elif axis.type == IterVar.FUSE:
            return [axis.fused]
        else:
            return []
    res = topo_sort(axis_tuple, get_output)
    return res

def axis_topo_sort_bottom_up(axis_tuple):
    def get_output(axis):
        if axis.type == IterVar.NORMAL and hasattr(axis, "parent"):
            return [axis.parent]
        elif axis.type == IterVar.NORMAL and hasattr(axis, "outer"):
            return [axis.outer, axis.inner]
        else:
            return []
    res = topo_sort(axis_tuple, get_output)
    return res

def infer_root_iter_bound(tensor, rmap):
    if len(tensor.outputs) > 0:
        bounds = None
        # step 1: do pass up for compute_at
        if len(tensor.attach_path) > 0: # not necessary but just for clearity
            for axis in tensor.attach_path:
                rmap[axis] = Range.single_point(axis)
            pass_up(rmap, axis_topo_sort_bottom_up(tensor.attach_path))
        # step 2: calculate bound of producer
        for output in tensor.outputs:
            for consumer in output.consumers[tensor]:
                relax_set = set()
                # TODO: fix this
                if tensor.attach_at is not output and len(output.attach_path) > 0:
                    # tensor is not attached at current consumer
                    for axis in output.root_axis:
                        relax_set.add(axis)
                new_bounds = [evaluate_expr_bound(index, rmap, relax_set) for index in consumer.index]
                
                if bounds is None:
                    bounds = new_bounds
                else:
                    # TODO: Test consolidate bounds
                    for i, new_bound in enumerate(new_bounds):
                        bounds[i] = Range(
                                Expr.min(bounds[i].start, new_bound.start), 
                                Expr.max(bounds[i].end, new_bound.end),
                                type_=RangeType.CLOSED_CLOSED
                                )
        
        # step 3: normalize bounds
        # TODO: fix this with attach.py
        # shift = [bound.normalize() for bound in bounds]
        # # change consumer index according to bound normalizatoin since index must start from 0
        # # for example: [-3, 125) is normalized to [0, 128)
        ## TODO: fix this: may substract variable, for example [m, m + 5] -> [0, 5]
        # for output in tensor.outputs:
        #     for consumer in output.consumers[tensor]:
        #         consumer.index = tuple([idx - shift[i] for i, idx in enumerate(consumer.index)])

        # step 4: set range of root axis so later it can be propagated to leaf
        for i, root_axis in enumerate(tensor.root_axis):
            # convert back to closed_open interval
            bounds[i].end = bounds[i].end + 1
            rmap[root_axis] = bounds[i]
            root_axis.range = rmap[root_axis]
        
        # step 5: recover pass_up side effect
        if len(tensor.attach_path) > 0:
            affected_axis = axis_topo_sort_bottom_up(tensor.attach_path)
            for axis in affected_axis:
                rmap[axis] = axis.range
    else:
        # is output tensor, therefore no consumers
        for root_axis in tensor.root_axis:
            rmap[root_axis] = root_axis.range

# TODO: fix pass up and pass down, take care of single point case
def pass_down(rmap, axis_tuple):
    for axis in axis_tuple:
        if axis.type == IterVar.SPLIT:
            if rmap[axis].is_single_point:
                rmap[axis.outer] = Range.single_point(axis.outer)
                rmap[axis.inner] = Range.single_point(axis.inner)
            else:
                rmap[axis.outer] = Range(0, Expr.ceildiv(rmap[axis].end, axis.factor))
                rmap[axis.inner] = Range(0, axis.factor)
            axis.outer.range = rmap[axis.outer]
            axis.inner.range = rmap[axis.inner]
        elif axis.type == IterVar.FUSE and axis is axis.fused.outer:
            if rmap[axis].is_single_point and rmap[axis.fused.inner].is_single_point:
                rmap[axis.fused] = Range.single_point(axis.fused)
            else:
                rmap[axis.fused] = Range(rmap[axis.fused.outer].start * axis.factor + rmap[axis.inner].start, 
                                            rmap[axis.fused.outer].end * axis.factor + rmap[axis.inner].end)
            axis.fused.range = rmap[axis.fused]
        else:
            # we already know root_axis's range
            pass

def pass_up(rmap, axis_tuple):
    for axis in axis_tuple:
        if axis.type == IterVar.SPLIT:
            if rmap[axis.outer].is_single_point and rmap[axis.inner].is_single_point:
                rmap[axis] = Range.single_point(axis)
            else:
                rmap[axis] = Range(rmap[axis.outer].start * axis.factor + rmap[axis.inner].start, 
                                    rmap[axis.outer].end * axis.factor + rmap[axis.inner].end)
        elif axis.type == IterVar.FUSE and axis is axis.fused.outer:
            if rmap[axis.fused].is_single_point:
                rmap[axis.fused.outer] = Range.single_point(axis.fused.outer)
                rmap[axis.fused.inner] = Range.single_point(axis.fused.inner)
            else:
                rmap[axis.fused.outer] = Range(rmap[axis.fused].start // axis.factor, Expr.ceilDiv(rmap[axis.fused].end, axis.factor))
                rmap[axis.fused.inner] = Range(0, axis.factor)
        else:
            pass

def set_rmap(rmap, axis_sort):
    for axis in axis_sort:
        rmap[axis] = axis.range

def create_attach_path(tensor):
    cur_tensor = tensor
    attach_path = []
    while cur_tensor.attached:
        cur_attach_path = []
        for axis in cur_tensor.attach_at.axis:
            cur_attach_path.append(axis)
            if axis is cur_tensor.attach_axis:
                attach_path += reversed(cur_attach_path)
                break
        cur_tensor = cur_tensor.attach_at
    tensor.attach_path = tuple(attach_path)

def evaluate_expr_bound(expr, rmap, relax_set):
    if isinstance(expr, IterVar):
        if rmap[expr].is_single_point:
            interval = rmap[expr]
        else:
            # convert to closed closed interval
            interval = Range(rmap[expr].start, rmap[expr].end - 1, type_= RangeType.CLOSED_CLOSED)
        if expr in relax_set:
            return Range(evaluate_expr_bound(interval.start, rmap, relax_set).start, evaluate_expr_bound(interval.end, rmap, relax_set).end, type_= RangeType.CLOSED_CLOSED)
    
    elif isinstance(expr, BinaryExpr):
        left = evaluate_expr_bound(expr.left, rmap, relax_set)
        right = evaluate_expr_bound(expr.right, rmap, relax_set)
        if expr.type == Expr.ADD:
            interval = Range(left.start + right.start, left.end + right.end, type_= RangeType.CLOSED_CLOSED)
        elif expr.type == Expr.SUB:
            interval = Range(left.start - right.start, left.end - right.end, type_= RangeType.CLOSED_CLOSED)
        elif expr.type == Expr.MUL:
            interval = Range(left.start * right.start, left.end * right.end, type_= RangeType.CLOSED_CLOSED)
        elif expr.type == Expr.FLOOR_DIV:
            interval = Range(left.start // right.start, left.end // right.end, type_= RangeType.CLOSED_CLOSED)
        elif expr.type == Expr.MOD:
            interval = Range(left.start % right.start, left.end % right.end, type_= RangeType.CLOSED_CLOSED)
        elif expr.type == Expr.MIN:
            interval = Range(Expr.min(left.start, right.start), Expr.min(left.end, right.end), type_= RangeType.CLOSED_CLOSED)
        elif expr.type == Expr.MAX:
            interval = Range(Expr.max(left.start, right.start), Expr.max(left.end, right.end), type_= RangeType.CLOSED_CLOSED)
        else:
            raise ValueError("Unsupported type {}.".format(expr.type))
    elif isinstance(expr, UnaryExpr):
        if expr.type == Expr.NEG:
            inner = evaluate_expr_bound(expr.expr, rmap, relax_set)
            interval = Range(- inner.end, - inner.start, type_= RangeType.CLOSED_CLOSED)
        else:
            raise ValueError("Unsupported type {}.".format(expr.type))
    elif isinstance(expr, IfThenElseExpr):
        # TODO: fix ifThenElseExpr
        then_interval = evaluate_expr_bound(expr.then_expr, rmap, relax_set)
        else_interval = evaluate_expr_bound(expr.else_expr, rmap, relax_set)
    elif isinstance(expr, TensorSliceExpr):
        # TODO: fix TensorSliceExpr
        pass
    elif isinstance(expr, ConstExpr) or isinstance(expr, VarExpr):
        interval = Range.single_point(expr)
    else:
        print(expr)
        raise ValueError("Unsupported expr type {}".format(type(expr)))
    return interval

def infer_bound_pass(tensor):
    tensors = tensor_topo_sort_bottom_up(tensor)
    rmap = {}
    for tensor in tensors:
        axis_sort = axis_topo_sort_top_down(tensor.root_axis + tensor.reduce_axis)
        set_rmap(rmap, axis_sort)
        create_attach_path(tensor)
        infer_root_iter_bound(tensor, rmap)
        pass_down(rmap, axis_sort)


def check_bound_pass(tensor):
    tensors = tensor_topo_sort_bottom_up(tensor)
    for tensor in tensors:
        is_safe = True
        for idx, root_axis in enumerate(tensor.root_axis):
            # TODO: add boundary test, can prove?
            res = isinstance(root_axis.range.end, ConstExpr) and isinstance(tensor.shape[idx], ConstExpr) and root_axis.range.end.val <= tensor.shape[idx].val
            if res:
                # can decrease tensor size to save memory
                tensor.shape[idx].val = root_axis.range.end.val
            is_safe = is_safe and res
        tensor.is_safe = is_safe
                    
                