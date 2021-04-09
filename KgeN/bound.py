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
        if len(tensor.fixed_axis) > 0: # not necessary but just for clearity
            for axis in tensor.fixed_axis:
                rmap[axis] = Range.single_point(axis)
            pass_up(rmap, reversed(tensor.attach_at.axis_sort))
        for consumer in tensor.consumers:
            new_bounds = [evaluate_expr_bound(index, rmap) for index in consumer.index]
            if bounds is None:
                bounds = new_bounds
            else:
                # TODO: Test consolidate bounds
                for i, new_bound in enumerate(new_bounds):
                    bounds[i] = Range(
                            Expr.min(bounds[i].start, new_bound.start), 
                            Expr.max(bounds[i].end, new_bound.end)
                        )
        # normalize bounds
        shift = [bound.normalize() for bound in bounds]
        # change consumer index according to bound normalizatoin since index must start from 0
        for consumer in tensor.consumers:
            consumer.index = tuple([idx - shift[i] for i, idx in enumerate(consumer.index)])

        for i, root_axis in enumerate(tensor.root_axis):
            rmap[root_axis] = bounds[i]
            root_axis.range = rmap[root_axis]
        
        # recover pass_up effect
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
            if axis.range.is_single_point:
                pass
            else:
                # TODO: fix this: should be ceil div
                rmap[axis.outer] = Range(0, rmap[axis].end // axis.factor)
                rmap[axis.inner] = Range(0, axis.factor)
                axis.outer.range = rmap[axis.outer]
                axis.inner.range = rmap[axis.inner]
        elif axis.type == IterVar.FUSE:
            rmap[axis.fused] = Range(0, rmap[axis.fused.outer].end * rmap[axis.fused.inner].end)
            axis.fused.range = rmap[axis.fused]
        else:
            # we already know root_axis's range
            pass

def pass_up(rmap, axis_tuple):
    for axis in axis_tuple:
        if axis.type == IterVar.SPLIT:
            rmap[axis] = Range(0, rmap[axis.outer].end * rmap[axis.inner].end)
        elif axis.type == IterVar.FUSE:
            if axis is axis.fused.outer:
                # TODO: fix this: should be ceil div
                rmap[axis] = Range(0, rmap[axis.fused].end // axis.factor)
            else:
                rmap[axis] = Range(0, axis.factor)
        else:
            pass

def get_rmap(tensors):
    rmap = {}
    for tensor in tensors:
        if tensor.type != TensorExpr.PLACEHOLDER: # we don't care about placeholder's bound
            create_axis_sort(tensor) # cache axis sort results
            for axis in tensor.axis_sort:
                rmap[axis] = axis.range
    return rmap

def create_axis_sort(tensor):
    tensor.axis_sort = axis_topo_sort(tensor.root_axis + tensor.reduce_axis)

def evaluate_expr_bound(expr, rmap):
    if isinstance(expr, IterVar):
        return rmap[expr]
    elif isinstance(expr, BinaryExpr):
        # TODO: fix corner cases
        left = evaluate_expr_bound(expr.left, rmap)
        right = evaluate_expr_bound(expr.right, rmap)
        if expr.type == Expr.ADD:
            if left.is_single_point and right.is_single_point:
                interval = Range.single_point(left.start + right.start)
            else:
                interval = Range(left.start + right.start, left.end + right.end)
        elif expr.type == Expr.SUB:
            if left.is_single_point and right.is_single_point:
                interval = Range.single_point(left.start - right.start)
            else:
                interval = Range(left.start - right.start, left.end - right.end)
        elif expr.type == Expr.MUL:
            if left.is_single_point and right.is_single_point:
                interval = Range.single_point(left.start * right.start)
            elif not left.is_single_point and not right.is_single_point:
                interval = Range(left.start * right.start, (left.end - 1) * (right.end - 1))
            elif left.is_single_point and not right.is_single_point:
                interval = Range(left.start * right.start, left.end * (right.end - 1))
            else:
                interval = Range(left.start * right.start, (left.end -1 ) * right.end)
        elif expr.type == Expr.FLOOR_DIV:
            if left.is_single_point and right.is_single_point:
                interval = Range.single_point(left.start // right.start)
            elif not left.is_single_point and not right.is_single_point:
                interval = Range(left.start // right.start, (left.end - 1) // (right.end - 1))
            else:
                interval = Range(left.start // right.start, left.end // right.end)
        elif expr.type == Expr.MOD:
            if left.is_single_point and right.is_single_point:
                interval = Range.single_point(left.start % right.start)
            elif not left.is_single_point and right.is_single_point:
                interval = Range(left.start % right.start, left.end % right.end)
            else:
                raise ValueError("Should not be here.")
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
        if tensor.type != TensorExpr.PLACEHOLDER: # we don't care about placeholder's bound
            infer_root_iter_bound(tensor, rmap)
            pass_down(rmap, tensor.axis_sort)