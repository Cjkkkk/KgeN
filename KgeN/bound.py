from .utils import *
from .expr_simplifier import expr_simpifier

# bound inference
def rewrite_expr(expr, shift_map):
    if isinstance(expr, TensorSliceExpr):
        new_idx = []
        for index in expr.index:
            new_idx.append(rewrite_expr(index, shift_map))
            expr.index = tuple(new_idx)
    elif isinstance(expr, BinaryExpr):
        expr.left = rewrite_expr(expr.left, shift_map)
        expr.right = rewrite_expr(expr.right, shift_map)
    elif isinstance(expr, IfThenElseExpr):
        expr.condition = rewrite_expr(expr.condition, shift_map)
        expr.then_expr = rewrite_expr(expr.then_expr, shift_map)
        expr.else_expr = rewrite_expr(expr.else_expr, shift_map)
    elif isinstance(expr, IterVar):
        if expr in shift_map:
            expr = expr + shift_map[expr]
            expr = expr_simpifier.simpify(expr)
    return expr

def normalize_bound_and_rewrite_expr(tensor, bounds):
    shift = [bound.normalize() for bound in bounds]
    # change consumer index according to bound normalizatoin since index must start from 0
    # for example: [-3, 125) is normalized to [0, 128)
    root_axis_to_shift = {}
    for i, axis in enumerate(tensor.root_axis):
        root_axis_to_shift[axis] = shift[i]
    
    for output in tensor.outputs:
        for consumer in output.consumers[tensor]:
            consumer.index = tuple([expr_simpifier.simpify(idx - shift[i]) for i, idx in enumerate(consumer.index)])
    
    if tensor.type != TensorExpr.PLACEHOLDER:
        tensor.expr = rewrite_expr(tensor.expr, root_axis_to_shift)

def consolidate_range(a, b):
    return Range(Expr.min(a.start, b.start), Expr.max(a.end, b.end), type_=Range.CLOSED_CLOSED)

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
                        bounds[i] = consolidate_range(bounds[i], new_bound)
        
        # step 3: normalize bounds
        normalize_bound_and_rewrite_expr(tensor, bounds)

        # step 4: set range of root axis so later it can be propagated to leaf
        for i, root_axis in enumerate(tensor.root_axis):
            # convert back to closed_open interval
            bounds[i].as_closed_open()
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


def pass_down(rmap, axis_tuple):
    for axis in axis_tuple:
        if axis.type == IterVar.SPLIT:
            if rmap[axis].is_single_point:
                rmap[axis.outer] = Range.single_point(axis.outer)
                rmap[axis.inner] = Range.single_point(axis.inner)
            else:
                rmap[axis.outer] = Range(0, Expr.ceildiv(rmap[axis].end, axis.factor))
                rmap[axis.inner] = Range(0, axis.factor)
        elif axis.type == IterVar.FUSE and axis is axis.fused.outer:
            if rmap[axis].is_single_point and rmap[axis.fused.inner].is_single_point:
                rmap[axis.fused] = Range.single_point(axis.fused)
            else:
                rmap[axis.fused] = Range(rmap[axis.fused.outer].start * axis.factor + rmap[axis.inner].start, 
                                            rmap[axis.fused.outer].end * axis.factor + rmap[axis.inner].end)
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
            interval = Range(rmap[expr].start, rmap[expr].end - 1, type_= Range.CLOSED_CLOSED)
        if expr in relax_set:
            interval = Range(evaluate_expr_bound(interval.start, rmap, relax_set).start, evaluate_expr_bound(interval.end, rmap, relax_set).end, type_= Range.CLOSED_CLOSED)
    
    elif isinstance(expr, BinaryExpr):
        left = evaluate_expr_bound(expr.left, rmap, relax_set)
        right = evaluate_expr_bound(expr.right, rmap, relax_set)
        if expr.type == Expr.ADD:
            interval = Range(left.start + right.start, left.end + right.end, type_= Range.CLOSED_CLOSED)
        elif expr.type == Expr.SUB:
            interval = Range(left.start - right.start, left.end - right.end, type_= Range.CLOSED_CLOSED)
        elif expr.type == Expr.MUL:
            interval = Range(left.start * right.start, left.end * right.end, type_= Range.CLOSED_CLOSED)
        elif expr.type == Expr.FLOOR_DIV:
            interval = Range(left.start // right.start, left.end // right.end, type_= Range.CLOSED_CLOSED)
        elif expr.type == Expr.MOD:
            interval = Range(left.start % right.start, left.end % right.end, type_= Range.CLOSED_CLOSED)
        elif expr.type == Expr.MIN:
            interval = Range(Expr.min(left.start, right.start), Expr.min(left.end, right.end), type_= Range.CLOSED_CLOSED)
        elif expr.type == Expr.MAX:
            interval = Range(Expr.max(left.start, right.start), Expr.max(left.end, right.end), type_= Range.CLOSED_CLOSED)
        else:
            raise ValueError("Unsupported op type {}.".format(expr.type))
    elif isinstance(expr, UnaryExpr):
        if expr.type == Expr.NEG:
            inner = evaluate_expr_bound(expr.expr, rmap, relax_set)
            interval = Range(- inner.end, - inner.start, type_= Range.CLOSED_CLOSED)
        else:
            raise ValueError("Unsupported op type {}.".format(expr.type))
    elif isinstance(expr, IfThenElseExpr):
        # TODO: fix ifThenElseExpr
        then_interval = evaluate_expr_bound(expr.then_expr, rmap, relax_set)
        else_interval = evaluate_expr_bound(expr.else_expr, rmap, relax_set)
        interval = consolidate_range(then_interval, else_interval)
    elif isinstance(expr, TensorSliceExpr):
        # TODO: fix TensorSliceExpr
        pass
    elif isinstance(expr, ConstExpr) or isinstance(expr, VarExpr):
        interval = Range.single_point(expr)
    else:
        raise ValueError("Unsupported expr type {}".format(type(expr)))
    return interval

def bound_simplify_and_bind(rmap, axis_sort):
    for axis in axis_sort:
        rmap[axis].start = expr_simpifier.simpify(rmap[axis].start)
        rmap[axis].end = expr_simpifier.simpify(rmap[axis].end)
        axis.range = rmap[axis]

def infer_bound_pass(tensors):
    rmap = {}
    for tensor in tensors:
        axis_sort = axis_topo_sort_top_down(tensor.root_axis + tensor.reduce_axis)
        set_rmap(rmap, axis_sort)
        create_attach_path(tensor)
        infer_root_iter_bound(tensor, rmap)
        pass_down(rmap, axis_sort)
        bound_simplify_and_bind(rmap, axis_sort)


def check_bound_pass(tensors):
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
                    
                