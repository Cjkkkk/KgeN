from .utils import *
from .visitor import RewriteVisitor
from .expr_simplifier import expr_simplifier
from .tir import Range
from .interval import Interval, union_interval, bound_evaluator

# bound inference
class RewriteIterVarVisitor(RewriteVisitor):
    def __init__(self, map):
        super().__init__()
        self.map = map

    def rewrite(self, expr):
        expr = expr.accept(self)
        return expr

    def visit_iter_expr(self, expr):
        if expr in self.map:
            expr = self.map[expr]
            expr = expr_simplifier.rewrite(expr)
        return expr

def normalize_bound_and_rewrite_expr(tensor, bounds):
    print(tensor.name)
    for bound in bounds:
        print(bound)
    res = [bound.normalize() for bound in bounds]
    for bound in bounds:
        bound.end = expr_simplifier.rewrite(bound.end)
    # change provider index according to bound normalizatoin since index must start from 0
    # for example: [-3, 125) is normalized to [0, 128)
    root_axis_to_shift = {}
    for i, axis in enumerate(tensor.root_axis):
        shift, stride = res[i]
        root_axis_to_shift[axis] = axis * stride + shift
        root_axis_to_shift[axis] = expr_simplifier.rewrite(root_axis_to_shift[axis])
        
    for output in tensor.outputs:
        for provider in output.providers[tensor]:
            provider.index = tuple([expr_simplifier.rewrite((idx - res[i][0]) // res[i][1]) for i, idx in enumerate(provider.index)])
    
    if tensor.type != TensorExpr.PLACEHOLDER:
        visitor = RewriteIterVarVisitor(root_axis_to_shift)
        tensor.expr = visitor.rewrite(tensor.expr)
    return bounds

def infer_root_iter_bound(tensor, rmap):
    if not tensor.is_output():
        # nothing union a = a
        bounds = [Interval.nothing() for _ in tensor.root_axis] 
        affected_axis = []
        # step 1: do pass up for compute_at
        for axis in tensor.attach_path:
            # Do not treat certain axis as single point axis
            if tensor.scope == "global" and axis.bind_to is not None and axis.bind_to.name in ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z"]:
                continue
            elif tensor.scope == "shared" and axis.bind_to is not None and axis.bind_to.name in ["threadIdx.x", "threadIdx.y", "threadIdx.z"]:
                continue
            rmap[axis] = Range.single_point(axis)
        affected_axis += axis_topo_sort_bottom_up(tensor.attach_path)
        pass_up(rmap, affected_axis)

        # step 2: calculate bound of producer
        for output in tensor.outputs:
            for provider in output.providers[tensor]:
                relax_set = set()
                # TODO: fix this
                if tensor.attach_at is not output and len(output.attach_path) > 0:
                    # tensor is not attached at current provider
                    for axis in output.root_axis:
                        relax_set.add(axis)

                constraint_map = {}
                if hasattr(provider, "constraint"):
                    constraint_map = provider.constraint
                        
                new_bounds = [bound_evaluator.evaluate(index, rmap, constraint_map, relax_set) for index in provider.index]
                for i, new_bound in enumerate(new_bounds):
                    bounds[i] = union_interval(bounds[i], new_bound)

        # step 3: normalize bounds
        bounds = normalize_bound_and_rewrite_expr(tensor, bounds)
        # step 4: set range of root axis so later it can be propagated to leaf
        for i, root_axis in enumerate(tensor.root_axis):
            # convert back to closed_open interval if it is not single
            rmap[root_axis] = bounds[i].convert_to_range()
            rmap[root_axis].end = expr_simplifier.rewrite(rmap[root_axis].end)
        # step 5: recover pass_up side effect
        set_rmap(rmap, affected_axis)
    else:
        # is output tensor, therefore no providers
        pass

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
                interval = bound_evaluator.evaluate(axis.split_outer * axis.split_inner.range.end + axis.split_inner, rmap, {}, {})
                rmap[axis] = interval.convert_to_range()
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


def bind_to_axis(rmap, axis_sort):
    for axis in axis_sort:
        axis.range = rmap[axis]

def infer_bound_pass(tensors):
    # rmap: {itervar: Range}
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
                new_shape.append(root_axis.range.end)
            else:
                new_shape.append(tensor.shape[idx])
            is_safe = is_safe and res
        tensor.shape = tuple(new_shape)
        tensor.is_safe = is_safe
                    
                