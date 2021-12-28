from KgeN.te import schedule
from KgeN.te.operation import ComputeOp
from KgeN.tir.ir import Expr, TensorExpr, Range, RewriteVisitor
from KgeN.te.utils import *
from KgeN.arith.expr_simplifier import expr_simplifier
from KgeN.arith.interval import Interval, union_interval, BoundEvaluator
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
        return expr

def normalize_bound_and_rewrite_expr(schedule, stage, bounds):
    tensor = stage.op.outputs[0]
    # loop normalization: 
    # change provider index according to bound normalizatoin since index must start from 0
    # for example: [-3, 125) is normalized to [0, 128)
    shift = [bound.normalize() for bound in bounds]

    # storage folding:
    for feed_stage in schedule.feed_graph[stage]:
        for provider in feed_stage.op.providers[tensor]:
            provider.index = tuple([idx - shift[i] for i, idx in enumerate(provider.index)])
    
    # must do this after loop normalization: 
    # example: for(i: 3, 5) A[i] = B[i] => for(i: 0, 2) A[i] = B[i + 3]
    if isinstance(stage.op, ComputeOp):
        root_axis_to_shift = {}
        for i, axis in enumerate(stage.op.axis):
            root_axis_to_shift[axis] = shift[i] + axis
        
        visitor = RewriteIterVarVisitor(root_axis_to_shift)
        tensor.expr = visitor.rewrite(stage.op.expr)
    return bounds

def infer_root_iter_bound(schedule, stage, rmap):
    if not stage.is_output:
        # nothing union a = a
        bounds = [Interval.nothing() for _ in stage.op.axis] 
        affected_axis = []
        # step 1: do pass up for compute_at
        attach_path = schedule.attach_path[stage]
        for axis in attach_path:
            tensor = stage.op.outputs[0]
            # Do not treat certain axis as single point axis
            if tensor.scope == "global" and axis.bind_to is not None and axis.bind_to.thread_tag in ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z", "vthread"]:
                continue
            elif tensor.scope == "shared" and axis.bind_to is not None and axis.bind_to.thread_tag in ["threadIdx.x", "threadIdx.y", "threadIdx.z", "vthread"]:
                continue
            rmap[axis] = Range.single_point(axis)
        affected_axis += axis_topo_sort_bottom_up(attach_path)
        pass_up(rmap, affected_axis)

        # step 2: calculate bound of producer
        for feed_stage in schedule.feed_graph[stage]:
            for provider in feed_stage.op.providers[stage.op.outputs[0]]:
                relax_set = set()
                # TODO: fix this
                if stage.attach_at is not feed_stage and len(schedule.attach_path[feed_stage]) > 0:
                    # tensor is not attached at current provider
                    for axis in feed_stage.op.axis:
                        relax_set.add(axis)

                constraint_map = {}
                if hasattr(provider, "constraint"):
                    constraint_map = provider.constraint
                
                bound_evaluator = BoundEvaluator(rmap, constraint_map, relax_set)
                new_bounds = [bound_evaluator.evaluate(index) for index in provider.index]
                for i, new_bound in enumerate(new_bounds):
                    bounds[i] = union_interval(bounds[i], new_bound)

        # step 3: normalize bounds
        bounds = normalize_bound_and_rewrite_expr(schedule, stage, bounds)
        # step 4: set range of root axis so later it can be propagated to leaf
        for i, axis in enumerate(stage.op.axis):
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
                bound_evaluator = BoundEvaluator(rmap, {}, {})
                interval = bound_evaluator.evaluate(axis.split_outer * axis.split_inner.range.end + axis.split_inner)
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


def bind_to_axis(rmap, axis_sort):
    for axis in axis_sort:
        axis.range = rmap[axis]


def infer_bound_pass(schedule):
    # rmap: {itervar: Range}
    rmap = {}
    for stage in schedule.stages:
        if isinstance(stage.op, ComputeOp):
            axis_sort = axis_topo_sort_top_down(stage.op.axis + stage.op.reduce_axis)
            set_rmap(rmap, axis_sort)
            
            infer_root_iter_bound(schedule, stage, rmap)
            pass_down(rmap, axis_sort)
            
            # set axis's range from rmap
            bind_to_axis(rmap, axis_sort)
            # check if axis's range equals to the thread axis's range that it binds to
            check_thread_axis_bound(axis_sort)
    return rmap


def check_thread_axis_bound(axis_sort):
    for axis in axis_sort:
        if axis.bind_to is not None and isinstance(axis.range.end, ConstExpr) and isinstance(axis.bind_to.range.end, ConstExpr) and not axis.bind_to.range.end.same_as(ConstExpr(math.inf)):
            assert axis.range.end.same_as(axis.bind_to.range.end), "range of axis {0} should equal to range of thread axis {1}, got {2} and {3} respectively.".format(axis.name, axis.bind_to.name, axis.range, axis.bind_to.range)


def set_tensor_shape_pass(schedule):
    for stage in schedule.stages:
        if isinstance(stage.op, ComputeOp):
            tensor = stage.op.outputs[0]
            new_shape = []
            for idx, root_axis in enumerate(stage.op.axis):
                # TODO: add boundary test, can prove?
                if isinstance(root_axis.range.end, ConstExpr) and isinstance(tensor.shape[idx], ConstExpr) and root_axis.range.end.val <= tensor.shape[idx].val:
                    # can decrease tensor size to save memory
                    new_shape.append(root_axis.range.end)
                else:
                    new_shape.append(tensor.shape[idx])
            tensor.shape = tuple(new_shape)