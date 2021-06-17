from tvm import te
from tvm.tir.expr import Reduce
from .te import compute
from .tir import *
from .visitor import RewriteVisitor
from .utils import axis_topo_sort_top_down
import math


# schedule primitives
def check_tensor_and_axis(tensor, *axis):
    assert isinstance(tensor, TensorExpr), "Expect TensorExpr not {0}".format(type(tensor))

    for ax in axis:
        assert ax in tensor.axis, "{0} is not {1}'s axis".format(ax.name, tensor.name)

def bind(ax, thread_axis):
    assert thread_axis.type == IterVar.BIND, "should provide thread_axis."
    assert ax.bind_to is None, "already bind to another thread axis {}".format(ax.bind_to.name)
    assert ax.type == IterVar.DEFAULT or ax.type == IterVar.REDUCE, "can not bind axis of {} type".format(ax.type)
    ax.bind_to = thread_axis

def split(tensor, ax, factor=-1, nparts=-1):
    check_tensor_and_axis(tensor, ax)
    
    new_axis = []
    for axis in tensor.axis:
        if ax is axis:
            outer = IterVar(axis.name + "_outer", -math.inf, math.inf)
            inner = IterVar(axis.name + "_inner", -math.inf, math.inf)
            
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
    tensor.axis = new_axis
    return outer, inner

def tile(tensor, ax1, ax2, factor1, factor2):
    check_tensor_and_axis(tensor, ax1, ax2)
    ax1_outer, ax1_inner = split(tensor, ax1, factor1)
    ax2_outer, ax2_inner = split(tensor, ax2, factor2)
    return ax1_outer, ax1_inner, ax2_outer, ax2_inner

def reorder(tensor, *axis):
    check_tensor_and_axis(tensor, *axis)
    new_axis = []
    cur = 0
    axis_set = set(axis)
    for ax in tensor.axis:
        if ax in axis_set:
            new_axis.append(axis[cur])
            cur += 1
        else:
            new_axis.append(ax)
    tensor.axis = new_axis

def fuse(tensor, ax1, ax2):
    check_tensor_and_axis(tensor, ax1, ax2)
    new_axis = []
    # set axis to fuse
    fused = IterVar(ax1.name + "_" + ax2.name + "_fused", -math.inf, math.inf)
    
    ax1.relation = IterVar.FUSE
    ax2.relation = IterVar.FUSE
    
    ax1.fused = fused
    ax2.fused = fused

    fused.fused_outer = ax1
    fused.fused_inner = ax2

    for axis in tensor.axis:
        if axis is ax1:
            new_axis.append(fused)
        elif axis is ax2:
            continue
        else:
            new_axis.append(axis)
    tensor.axis = new_axis
    return fused

def vectorize(tensor, axis):
    check_tensor_and_axis(tensor, axis)
    axis.type = IterVar.VECTORIZED

def unroll(tensor, axis):
    check_tensor_and_axis(tensor, axis)
    axis.type = IterVar.UNROLL

def compute_at(tensor, attach_at, axis):
    check_tensor_and_axis(attach_at, axis)
    tensor.attached = True
    tensor.attach_at = attach_at
    tensor.attach_axis = axis
    axis.attached_computation.append(tensor)

def compute_inline(tensor):
    tensor.is_inline = True

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

def cache_read(tensor, scope, readers):
    cache_tensor_name = tensor.name + "_" + scope
    lambda_str = "def _ ({0}): return tensor[{0}]".format(", ".join([tensor.compute_func.__code__.co_varnames[i] if tensor.compute_func is not None else 'i' + str(i) for i in range(len(tensor.shape))]))
    local_vars = {}
    exec(lambda_str, {"tensor": tensor}, local_vars)
    compiled = local_vars["_"]
    cache_tensor = compute(tensor.shape, compiled, cache_tensor_name, scope)
    dataflow_rewriter.add_mapping(tensor, cache_tensor)
    
    # rewrite dataflow from tensor -> readers to tensor -> cache_tensor -> readers
    for reader in readers:
        reader.expr = dataflow_rewriter.rewrite(reader.expr)
    return cache_tensor

def cache_write(tensor, scope):
    # TODO: fix compute, use local
    cache_tensor_name = tensor.name + "_" + scope
    cache_tensor = compute(tensor.shape, tensor.compute_func, cache_tensor_name, scope)
    cache_tensor.expr = dataflow_rewriter.rewrite(cache_tensor.expr)
    
    # change tensor's compute_func
    lambda_str = "def _ ({0}): return cache_tensor[{0}]".format(", ".join([tensor.compute_func.__code__.co_varnames[i] if tensor.compute_func is not None else 'i' + str(i) for i in range(len(tensor.shape))]))
    local_vars = {}
    exec(lambda_str, {"cache_tensor": cache_tensor}, local_vars)
    compiled = local_vars["_"]

    tensor.compute_func = compiled
    tensor.expr = tensor.compute_func(*tensor.root_axis)

    all_reduce_axis = set(axis_topo_sort_top_down(tensor.reduce_axis))
    new_axis = [axis for axis in tensor.axis if axis not in all_reduce_axis]
    tensor.reduce_axis = ()
    tensor.axis = new_axis

    # TODO: what to do with attach information?
    return cache_tensor