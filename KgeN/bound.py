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

def tensor_topo_sort(tensor):
    def get_output(tensor):
        return tensor.inputs
    res = topo_sort([tensor], get_output)
    return res

def axis_topo_sort(axis_tuple):
    def get_output(axis):
        if axis.type == IterVar.SPLIT:
            return [axis.outer, axis.inner]
        elif axis.type == IterVar.FUSE:
            return [axis.fused]
        else:
            return []
    res = topo_sort(axis_tuple, get_output)
    return res

def infer_root_iter_bound(tensor, rmap):
    if len(tensor.consumers) > 0:
        bounds = None
        # step 1: do pass up for compute_at
        if len(tensor.fixed_axis) > 0: # not necessary but just for clearity
            for axis in tensor.fixed_axis:
                rmap[axis] = Range.single_point(axis)
            pass_up(rmap, reversed(tensor.attach_at.axis_sort))
        # step 2: calculate bound of producer
        for consumer in tensor.consumers:
            new_bounds = [evaluate_expr_bound(index, rmap) for index in consumer.index]
            for bound in new_bounds:
                if not bound.is_single_point:
                    # convert back to closed_open interval
                    bound.end = bound.end + 1
            if bounds is None:
                bounds = new_bounds
            else:
                # TODO: Test consolidate bounds
                for i, new_bound in enumerate(new_bounds):
                    bounds[i] = Range(
                            Expr.min(bounds[i].start, new_bound.start), 
                            Expr.max(bounds[i].end, new_bound.end)
                        )
        
        # step 3: normalize bounds
        shift = [bound.normalize() for bound in bounds]
        # change consumer index according to bound normalizatoin since index must start from 0
        # for example: [-3, 125) is normalized to [0, 128)
        for consumer in tensor.consumers:
            consumer.index = tuple([idx - shift[i] for i, idx in enumerate(consumer.index)])

        # step 4: set range of root axis so later it can be propagated to leaf
        for i, root_axis in enumerate(tensor.root_axis):
            rmap[root_axis] = bounds[i]
            root_axis.range = rmap[root_axis]
        
        # step 5: recover pass_up side effect
        if len(tensor.fixed_axis) > 0:
            for axis in tensor.attach_at.axis_sort:
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
                rmap[axis.outer] = Range.single_point(0)
                rmap[axis.inner] = Range.single_point(0)
            else:
                rmap[axis.outer] = Range(0, Expr.ceilDiv(rmap[axis].end, axis.factor))
                rmap[axis.inner] = Range(0, axis.factor)
            axis.outer.range = rmap[axis.outer]
            axis.inner.range = rmap[axis.inner]
        elif axis.type == IterVar.FUSE and axis is axis.fused.outer:
            if rmap[axis].is_single_point and rmap[axis.fused.inner].is_single_point:
                rmap[axis.fused] = Range.single_point(0)
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

def get_rmap(tensors):
    rmap = {}
    for tensor in tensors:
        create_axis_sort(tensor) # cache axis sort results
        for axis in tensor.axis_sort:
            rmap[axis] = axis.range
    return rmap

def create_axis_sort(tensor):
    tensor.axis_sort = axis_topo_sort(tensor.root_axis + tensor.reduce_axis)
    
def evaluate_expr_bound(expr, rmap):
    if isinstance(expr, IterVar):
        if rmap[expr].is_single_point:
            return rmap[expr]
        else:
            # convert to closed closed interval
            return Range(rmap[expr].start, rmap[expr].end - 1, type_= RangeType.CLOSED_OPEN)
    elif isinstance(expr, BinaryExpr):
        # TODO: fix corner cases
        left = evaluate_expr_bound(expr.left, rmap)
        right = evaluate_expr_bound(expr.right, rmap)
        if expr.type == Expr.ADD:
            interval = Range(left.start + right.start, left.end + right.end, type_= RangeType.CLOSED_OPEN)
        elif expr.type == Expr.SUB:
            interval = Range(left.start - right.start, left.end - right.end, type_= RangeType.CLOSED_OPEN)
        elif expr.type == Expr.MUL:
            interval = Range(left.start * right.start, left.end * right.end, type_= RangeType.CLOSED_OPEN)
        elif expr.type == Expr.FLOOR_DIV:
            interval = Range(left.start // right.start, left.end // right.end, type_= RangeType.CLOSED_OPEN)
        elif expr.type == Expr.MOD:
            interval = Range(left.start % right.start, left.end % right.end, type_= RangeType.CLOSED_OPEN)
        else:
            raise ValueError("Unsupported type {}.".format(expr.type))
    elif isinstance(expr, UnaryExpr):
        if expr.type == Expr.NEG:
            inner = evaluate_expr_bound(expr.expr, rmap)
            interval = Range(- inner.end, - inner.start, type_= RangeType.CLOSED_OPEN)
        else:
            raise ValueError("Unsupported type {}.".format(expr.type))
    elif isinstance(expr, IfThenElseExpr):
        # TODO: fix ifThenElseExpr
        then_interval = evaluate_expr_bound(expr.then_expr, rmap)
        else_interval = evaluate_expr_bound(expr.else_expr, rmap)
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
    tensors = tensor_topo_sort(tensor)
    rmap = get_rmap(tensors)
    for tensor in tensors:
        infer_root_iter_bound(tensor, rmap)
        pass_down(rmap, tensor.axis_sort)


def check_bound_pass(tensor):
    tensors = tensor_topo_sort(tensor)
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
                    
                