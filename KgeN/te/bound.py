from KgeN.tir.ir import Expr, TensorExpr, Range, RewriteVisitor
from KgeN.te.utils import *
from KgeN.arith.expr_simplifier import expr_simplifier
from KgeN.arith.interval import Interval, union_interval, bound_evaluator
import math

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
            # expr = expr_simplifier.rewrite(expr)
        return expr

def normalize_bound_and_rewrite_expr(tensor, bounds):
    shift = [bound.normalize() for bound in bounds]
    # for bound in bounds:
    #     bound.end = expr_simplifier.rewrite(bound.end)
    
    # change provider index according to bound normalizatoin since index must start from 0
    # for example: [-3, 125) is normalized to [0, 128)
    
    root_axis_to_shift = {}
    for i, axis in enumerate(tensor.axis):
        root_axis_to_shift[axis] = axis + shift[i]
    
    for output in tensor.outputs:
        for provider in output.providers[tensor]:
            provider.index = tuple([idx - shift[i] for i, idx in enumerate(provider.index)])
    
    # must do this after loop normalization: 
    # example: for(i: 3, 5) A[i] = B[i] => for(i: 0, 2) A[i] = B[i + 3]
    if tensor.type != TensorExpr.PLACEHOLDER:
        visitor = RewriteIterVarVisitor(root_axis_to_shift)
        tensor.expr = visitor.rewrite(tensor.expr)
    return bounds

def infer_root_iter_bound(stage, rmap):
    if not stage.tensor.is_output():
        # nothing union a = a
        bounds = [Interval.nothing() for _ in stage.tensor.axis] 
        affected_axis = []
        # step 1: do pass up for compute_at
        for axis in stage.attach_path:
            # Do not treat certain axis as single point axis
            if stage.tensor.scope == "global" and axis.bind_to is not None and axis.bind_to.thread_tag in ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z", "vthread"]:
                continue
            elif stage.tensor.scope == "shared" and axis.bind_to is not None and axis.bind_to.thread_tag in ["threadIdx.x", "threadIdx.y", "threadIdx.z", "vthread"]:
                continue
            rmap[axis] = Range.single_point(axis)
        affected_axis += axis_topo_sort_bottom_up(stage.attach_path)
        pass_up(rmap, affected_axis)

        # step 2: calculate bound of producer
        for output in stage.outputs:
            for provider in output.tensor.providers[stage.tensor]:
                relax_set = set()
                # TODO: fix this
                if stage.attach_at is not output and len(output.attach_path) > 0:
                    # tensor is not attached at current provider
                    for axis in output.tensor.axis:
                        relax_set.add(axis)

                constraint_map = {}
                if hasattr(provider, "constraint"):
                    constraint_map = provider.constraint
                        
                new_bounds = [bound_evaluator.evaluate(index, rmap, constraint_map, relax_set) for index in provider.index]
                for i, new_bound in enumerate(new_bounds):
                    bounds[i] = union_interval(bounds[i], new_bound)

        # step 3: normalize bounds
        bounds = normalize_bound_and_rewrite_expr(stage.tensor, bounds)
        # step 4: set range of root axis so later it can be propagated to leaf
        for i, axis in enumerate(stage.tensor.axis):
            # convert back to closed_open interval if it is not single
            rmap[axis] = bounds[i].convert_to_range()
            # Important: only simplify bound here
            rmap[axis].end = expr_simplifier.rewrite(rmap[axis].end)
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


def create_attach_path(stage):
    cur_stage = stage
    attach_path = []
    while cur_stage.attached:
        cur_attach_path = []
        for axis in cur_stage.attach_at.leaf_axis:
            cur_attach_path.append(axis)
            if axis is cur_stage.attach_axis:
                attach_path += reversed(cur_attach_path)
                break
        cur_stage = cur_stage.attach_at
    stage.attach_path = tuple(attach_path)


def bind_to_axis(rmap, axis_sort):
    for axis in axis_sort:
        axis.range = rmap[axis]

def infer_bound_pass(schdule):
    # rmap: {itervar: Range}
    rmap = {}
    for stage in schdule.stages:
        axis_sort = axis_topo_sort_top_down(stage.tensor.axis + stage.tensor.reduce_axis)
        set_rmap(rmap, axis_sort)
        create_attach_path(stage)
        infer_root_iter_bound(stage, rmap)
        pass_down(rmap, axis_sort)
        # set axis's range from rmap
        bind_to_axis(rmap, axis_sort)
        # check if axis's range equals to the thread axis's range that it binds to
        check_thread_axis_bound(axis_sort)

def check_thread_axis_bound(axis_sort):
    for axis in axis_sort:
        if axis.bind_to is not None and isinstance(axis.range.end, ConstExpr) and isinstance(axis.bind_to.range.end, ConstExpr) and not axis.bind_to.range.end.same_as(ConstExpr(math.inf)):
            assert axis.range.end.same_as(axis.bind_to.range.end), "range of axis {0} should equal to range of thread axis {1}, got {2} and {3} respectively.".format(axis.name, axis.bind_to.name, axis.range, axis.bind_to.range)


def check_bound_pass(schdule):
    for stage in schdule.stages:
        tensor = stage.tensor
        is_safe = True
        new_shape = []
        for idx, root_axis in enumerate(tensor.axis):
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
                    
                