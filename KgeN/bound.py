from .utils import *
from .visitor import RewriteVisitor
from .expr_simplifier import expr_simplifier
from .tir import Range

# bound inference
class RewriteIterVarVisitor(RewriteVisitor):
    def __init__(self, map):
        super().__init__()
        self.map = map

    def visit_iter_expr(self, expr):
        if expr in self.map:
            expr = self.map[expr]
            expr = expr_simplifier.rewrite(expr)
        return expr

def normalize_bound_and_rewrite_expr(tensor, bounds):
    res = [bound.normalize() for bound in bounds]
    for bound in bounds:
        bound.end = expr_simplifier.rewrite(bound.end)
    # change provider index according to bound normalizatoin since index must start from 0
    # for example: [-3, 125) is normalized to [0, 128)
    root_axis_to_shift = {}
    for i, axis in enumerate(tensor.root_axis):
        shift,stride = res[i]
        root_axis_to_shift[axis] = (axis + shift) * stride
        root_axis_to_shift[axis] = expr_simplifier.rewrite(root_axis_to_shift[axis])
        
    for output in tensor.outputs:
        for provider in output.providers[tensor]:
            provider.index = tuple([expr_simplifier.rewrite((idx - res[i][0]) // res[i][1]) for i, idx in enumerate(provider.index)])
    
    if tensor.type != TensorExpr.PLACEHOLDER:
        visitor = RewriteIterVarVisitor(root_axis_to_shift)
        tensor.expr = visitor.rewrite(tensor.expr)
    return bounds

def consolidate_range(a, b):
    return Range(Expr.min(a.start, b.start), Expr.max(a.end, b.end), type=Range.CLOSED_CLOSED)

def infer_root_iter_bound(tensor, rmap):
    if len(tensor.outputs) > 0:
        bounds = None
        # step 1: do pass up for compute_at
        if len(tensor.attach_path) > 0: # not necessary but just for clearity
            for axis in tensor.attach_path:
                # Do not treat certain axis as single point axis
                # TODO: should this be necessary?
                if tensor.scope == "global" and axis.bind_name in ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z"]:
                    continue
                if tensor.scope == "shared" and axis.bind_name in ["threadIdx.x", "threadIdx.y", "threadIdx.z"]:
                    continue
                rmap[axis] = Range.single_point(axis)
            pass_up(rmap, axis_topo_sort_bottom_up(tensor.attach_path))
        # for axis in rmap:
        #     print(axis, rmap[axis])
        # step 2: calculate bound of producer
        for output in tensor.outputs:
            for provider in output.providers[tensor]:
                relax_set = set()
                # TODO: fix this
                if tensor.attach_at is not output and len(output.attach_path) > 0:
                    # tensor is not attached at current provider
                    for axis in output.root_axis:
                        relax_set.add(axis)
                new_bounds = [evaluate_expr_bound(index, rmap, relax_set) for index in provider.index]
                
                if bounds is None:
                    bounds = new_bounds
                else:
                    # TODO: Test consolidate bounds
                    for i, new_bound in enumerate(new_bounds):
                        bounds[i] = consolidate_range(bounds[i], new_bound)
        
        # step 3: normalize bounds
        bounds = normalize_bound_and_rewrite_expr(tensor, bounds)
        # step 4: set range of root axis so later it can be propagated to leaf
        for i, root_axis in enumerate(tensor.root_axis):
            # convert back to closed_open interval if it is not single
            bounds[i].as_closed_open()
            rmap[root_axis] = bounds[i]
            root_axis.range = rmap[root_axis]
        
        # step 5: recover pass_up side effect
        if len(tensor.attach_path) > 0:
            affected_axis = axis_topo_sort_bottom_up(tensor.attach_path)
            for axis in affected_axis:
                rmap[axis] = axis.range
    else:
        # is output tensor, therefore no providers
        for root_axis in tensor.root_axis:
            rmap[root_axis] = root_axis.range

def pass_down(rmap, axis_tuple):
    for axis in axis_tuple:
        if axis.relation == IterVar.SPLIT:
            if axis.factor != -1:
                rmap[axis.split_outer] = Range(0, Expr.ceildiv(rmap[axis].end, axis.factor))
                rmap[axis.split_inner] = Range(0, Expr.min(rmap[axis].end, axis.factor))
            else:
                rmap[axis.split_outer] = Range(0, Expr.min(rmap[axis].end, axis.nparts))
                rmap[axis.split_inner] = Range(0, Expr.ceildiv(rmap[axis].end, axis.nparts))
        elif axis.relation == IterVar.FUSE and axis is axis.fused.fused_outer:
            rmap[axis.fused] = Range(0, rmap[axis.fused.fused_outer].end * rmap[axis.fused.fused_inner].end)
        else:
            # we already know root_axis's range
            pass

# TODO: check this...
def pass_up(rmap, axis_tuple):
    for axis in axis_tuple:
        if axis.relation == IterVar.SPLIT:
            if rmap[axis.split_outer].is_single_point and rmap[axis.split_inner].is_single_point:
                rmap[axis] = Range.single_point(axis)
            else:
                rmap[axis] = evaluate_expr_bound(axis.split_outer * axis.split_inner.range.end + axis.split_inner, rmap, {})
                rmap[axis].as_closed_open()
        elif axis.relation == IterVar.FUSE and axis is axis.fused.fused_outer:
            if rmap[axis.fused].is_single_point:
                rmap[axis.fused.fused_outer] = Range.single_point(axis.fused.fused_outer)
                rmap[axis.fused.fused_inner] = Range.single_point(axis.fused.fused_inner)
            else:
                rmap[axis.fused.fused_outer] = Range(Expr.ceildiv(rmap[axis.fused].start, axis.fused.fused_inner.range.end), Expr.ceildiv(rmap[axis.fused].end, axis.fused.fused_inner.range.end))
                rmap[axis.fused.fused_inner] = Range(0, axis.fused.fused_inner.range.end)
        else:
            # we already set leaf_axis's range
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
        # convert to closed closed interval
        interval = Range(rmap[expr].start, expr_simplifier.rewrite(rmap[expr].end - 1), type=Range.CLOSED_CLOSED)
        if expr in relax_set:
            interval = Range(evaluate_expr_bound(interval.start, rmap, relax_set).start, evaluate_expr_bound(interval.end, rmap, relax_set).end, type=Range.CLOSED_CLOSED)
    
    elif isinstance(expr, BinaryExpr):
        left = evaluate_expr_bound(expr.left, rmap, relax_set)
        right = evaluate_expr_bound(expr.right, rmap, relax_set)
        if expr.type == Expr.ADD:
            interval = Range(left.start + right.start, left.end + right.end, type=Range.CLOSED_CLOSED)
        elif expr.type == Expr.SUB:
            interval = Range(left.start - right.start, left.end - right.end, type=Range.CLOSED_CLOSED)
        elif expr.type == Expr.MUL:
            interval = Range(left.start * right.start, left.end * right.end, type=Range.CLOSED_CLOSED)
        elif expr.type == Expr.FLOOR_DIV or expr.type == Expr.CEIL_DIV: # TODO: fix this
            interval = Range(left.start // right.start, left.end // right.end, type=Range.CLOSED_CLOSED)
        elif expr.type == Expr.MOD:
            interval = Range(left.start % right.start, left.end % right.end, type=Range.CLOSED_CLOSED)
        elif expr.type == Expr.MIN:
            interval = Range(Expr.min(left.start, right.start), Expr.min(left.end, right.end), type=Range.CLOSED_CLOSED)
        elif expr.type == Expr.MAX:
            interval = Range(Expr.max(left.start, right.start), Expr.max(left.end, right.end), type=Range.CLOSED_CLOSED)
        else:
            raise ValueError("Unsupported op type {}.".format(expr.type))
    elif isinstance(expr, UnaryExpr):
        if expr.type == Expr.NEG:
            inner = evaluate_expr_bound(expr.expr, rmap, relax_set)
            interval = Range(- inner.end, - inner.start, type=Range.CLOSED_CLOSED)
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
        interval = Range(expr, expr, type=Range.CLOSED_CLOSED)
    else:
        raise ValueError("Unsupported expr type {}".format(type(expr)))
    return interval

def bind_to_axis(rmap, axis_sort):
    for axis in axis_sort:
        axis.range = rmap[axis]

def infer_bound_pass(tensors):
    rmap = {}
    for tensor in tensors:
        axis_sort = axis_topo_sort_top_down(tensor.root_axis + tensor.reduce_axis)
        set_rmap(rmap, axis_sort)
        create_attach_path(tensor)
        infer_root_iter_bound(tensor, rmap)
        pass_down(rmap, axis_sort)
        bind_to_axis(rmap, axis_sort)


def check_bound_pass(tensors):
    for tensor in tensors:
        is_safe = True
        new_shape = []
        for idx, root_axis in enumerate(tensor.root_axis):
            # TODO: add boundary test, can prove?
            res = isinstance(root_axis.range.end, ConstExpr) and isinstance(tensor.shape[idx], ConstExpr) and root_axis.range.end.val <= tensor.shape[idx].val
            if res:
                # can decrease tensor size to save memory
                new_shape.append(root_axis.range.end + 1 if root_axis.range.type == Range.CLOSED_CLOSED else root_axis.range.end)
            else:
                new_shape.append(tensor.shape[idx])
            is_safe = is_safe and res
        tensor.shape = tuple(new_shape)
        tensor.is_safe = is_safe
                    
                