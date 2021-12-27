from KgeN.tir.ir.expr import *
from KgeN.tir.ir.visitor import RewriteVisitor
from .build_graph import CollectInputVisitor
from .operation import ComputeOp, compute
from .utils import axis_topo_sort_top_down, op_topo_sort_bottom_up
import math


# schedule primitives
def create_schedule(op):
    s = Schedule(op)
    return s

class Schedule:
    def __init__(self, output_op):
        ops = op_topo_sort_bottom_up(output_op)
        self.output_op = output_op
        self.ops = ops
        self.stages = []
        self.stage_map = {}

        self.feed_graph = {}
        # tensor's attach_path, only used when compute_at
        # for example: A.compute_at(B, B.axis[1]), then A.attach_path = (B.axis[1], B.axis[0])
        self.attach_path = {}
        
        for op in ops:
            stage = Stage(op)
            self.stages.append(stage)
            self.stage_map[op] = stage

        self.stage_map[output_op].is_output = True

    def __getitem__(self, tensor):
        return self.stage_map[tensor.op]

    def cache_read(self, tensor, scope, readers):
        cache_tensor_name = tensor.name + "_" + scope
        lambda_str = "def _ ({0}): return tensor[{0}]".format(", ".join([tensor.op.compute_func.__code__.co_varnames[i] if hasattr(tensor.op, "compute_func") else 'i' + str(i) for i in range(len(tensor.shape))]))
        local_vars = {}
        exec(lambda_str, {"tensor": tensor}, local_vars)
        compiled = local_vars["_"]
        cache_tensor = compute(tensor.shape, compiled, cache_tensor_name, scope)
        dataflow_rewriter.add_mapping(tensor, cache_tensor)
        
        # rewrite dataflow from tensor -> readers to tensor -> cache_tensor -> readers
        for reader in readers:
            reader.op.expr = dataflow_rewriter.rewrite(reader.op.expr)
            visitor = CollectInputVisitor()
            reader.op.inputs, reader.op.providers = visitor.collect(reader.op.expr)
        
        self.__init__(self.output_op)
        return cache_tensor

    def cache_write(self, tensor, scope):
        # TODO: fix compute, use local
        cache_tensor_name = tensor.name + "_" + scope
        cache_tensor = TensorExpr(tensor.shape, cache_tensor_name, tensor.op, scope=scope)
        cache_tensor.op.outputs = [cache_tensor]
        for axis in cache_tensor.op.axis:
            axis.name = axis.name.replace(tensor.name, cache_tensor.name, 1)

        # change tensor's compute_func
        lambda_str = "def _ ({0}): return cache_tensor[{0}]".format(", ".join([tensor.op.compute_func.__code__.co_varnames[i] if hasattr(tensor.op, "compute_func") else 'i' + str(i) for i in range(len(tensor.shape))]))
        local_vars = {}
        exec(lambda_str, {"cache_tensor": cache_tensor}, local_vars)
        compiled = local_vars["_"]

        fake = compute(tensor.shape, compiled, tensor.name, tensor.scope)
        tensor.op = fake.op
        tensor.op.outputs = [tensor]

        # all_reduce_axis = set(axis_topo_sort_top_down(tensor.op.reduce_axis))
        # new_axis = tuple([axis for axis in tensor.op.axis if axis not in all_reduce_axis])
        # tensor.op.reduce_axis = ()
        # tensor.op.axis = new_axis

        if cache_tensor.op is self.output_op:
            self.__init__(tensor.op)
        else:
            self.__init__(self.output_op)
        # TODO: what to do with attach information?
        return cache_tensor

class Stage:
    def __init__(self, op):
        self.op = op
        # is output stage
        self.is_output = False
        # compute at
        self.attached = False
        self.attach_at = None
        self.is_inline = False

        if isinstance(op, ComputeOp):
            self.leaf_axis = list(self.op.axis + self.op.reduce_axis)

    def check_axis(self, *axis):
        assert isinstance(self.op, ComputeOp), "Can not schedule placeholder operation."
        for ax in axis:
            assert ax in self.leaf_axis, "{0} is not {1}'s axis".format(ax.name, self.op.outputs[0].name)


    def bind(self, ax, thread_axis):
        assert thread_axis.type == IterVar.BIND, "Should provide thread_axis."
        assert ax.bind_to is None, "Already bind to another thread axis {}".format(ax.bind_to.name)
        assert ax.type == IterVar.DEFAULT or ax.type == IterVar.REDUCE, "Can not bind axis of {} type".format(ax.type)
        ax.bind_to = thread_axis

    def split(self, ax, factor=-1, nparts=-1):
        self.check_axis(ax)
        
        new_axis = []
        for axis in self.leaf_axis:
            if ax is axis:
                outer = IterVar(axis.name + "_outer", math.inf, IterVar.DEFAULT)
                inner = IterVar(axis.name + "_inner", math.inf, IterVar.DEFAULT)
                
                axis.split_outer = outer
                axis.split_inner = inner
                outer.split = axis
                inner.split = axis

                axis.factor = factor
                axis.nparts = nparts
                
                axis.relation = IterVar.SPLIT
                new_axis.append(outer)
                new_axis.append(inner)
            else:
                new_axis.append(axis)
        self.leaf_axis = new_axis
        return outer, inner

    def tile(self, ax1, ax2, factor1, factor2):
        self.check_axis(ax1, ax2)
        ax1_outer, ax1_inner = self.split(ax1, factor1)
        ax2_outer, ax2_inner = self.split(ax2, factor2)
        return ax1_outer, ax1_inner, ax2_outer, ax2_inner

    def reorder(self, *axis):
        self.check_axis(*axis)
        new_axis = []
        cur = 0
        axis_set = set(axis)
        for ax in self.leaf_axis:
            if ax in axis_set:
                new_axis.append(axis[cur])
                cur += 1
            else:
                new_axis.append(ax)
        self.leaf_axis = new_axis

    def fuse(self, ax1, ax2):
        self.check_axis(ax1, ax2)
        new_axis = []
        # set axis to fuse
        fused = IterVar(ax1.name + "_" + ax2.name + "_fused", math.inf, IterVar.DEFAULT)
        
        ax1.relation = IterVar.FUSE
        ax2.relation = IterVar.FUSE
        
        ax1.fused = fused
        ax2.fused = fused

        fused.fused_outer = ax1
        fused.fused_inner = ax2

        for axis in self.leaf_axis:
            if axis is ax1:
                new_axis.append(fused)
            elif axis is ax2:
                continue
            else:
                new_axis.append(axis)
        self.leaf_axis = new_axis
        return fused

    def vectorize(self, axis):
        self.check_axis(axis)
        axis.type = IterVar.VECTORIZED

    def unroll(self, axis):
        self.check_axis(axis)
        axis.type = IterVar.UNROLL

    def compute_at(self, attach_at, axis):
        attach_at.check_axis(axis)
        self.attached = True
        self.attach_at = attach_at
        self.attach_axis = axis
        axis.attached_computation.append(self)

    def compute_inline(self):
        self.is_inline = True

# rewrite dataflow for cache_read
class RewriteDataFlowVisitor(RewriteVisitor):
    def __init__(self):
        super().__init__()
        self.map = {}

    def add_mapping(self, tensor, cache_tensor):
        self.map[tensor] = cache_tensor
        for k, v in self.map.items():
            if tensor is v:
                self.map[k] = cache_tensor

    def rewrite(self, expr):
        expr = expr.accept(self)
        return expr
    
    def visit_tensor_expr(self, expr):
        if expr in self.map:
            expr = self.map[expr]
        return expr

dataflow_rewriter = RewriteDataFlowVisitor()