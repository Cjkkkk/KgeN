from .te import compute
from .tir import *
from .visitor import RewriteExprVisitor
import math

# schedule primitives
def bind(ax, name):
    if name not in ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z"]:
        raise ValueError("illegal binding name {}".format(name))
    ax.bind_type = IterVar.BIND
    ax.bind_name = name

def split(tensor, ax, factor):
    if not isinstance(tensor, TensorExpr):
        raise ValueError("Expect TensorExpr not {0}".format(type(tensor)))
    
    new_axis = []
    for axis in tensor.axis:
        if ax is axis:
            outer = IterVar(axis.name + "_outer", -math.inf, math.inf)
            inner = IterVar(axis.name + "_inner", -math.inf, math.inf)
            
            axis.splitted_outer = outer
            axis.splitted_inner = inner
            outer.splitted = axis
            inner.splitted = axis

            axis.factor = factor
            axis.type = IterVar.SPLIT
            new_axis.append(outer)
            new_axis.append(inner)
        else:
            new_axis.append(axis)
    tensor.axis = new_axis
    return outer, inner

def tile(tensor, ax1, ax2, factor1, factor2):
    ax1_outer, ax1_inner = split(tensor, ax1, factor1)
    ax2_outer, ax2_inner = split(tensor, ax2, factor2)
    return ax1_outer, ax1_inner, ax2_outer, ax2_inner

def reorder(tensor, axis_tuple):
    new_axis_list = []
    cur = 0
    axis_set = {*axis_tuple}
    for axis in tensor.axis:
        if axis in axis_set:
            new_axis_list.append(axis_tuple[cur])
            cur += 1
        else:
            new_axis_list.append(axis)
    tensor.axis = new_axis_list

def fuse(tensor, axis_tuple):
    new_axis = []
    # set axis to fuse
    fused = IterVar(axis_tuple[0].name + "_" + axis_tuple[1].name + "_fused", -math.inf, math.inf)
    
    axis_tuple[0].type = IterVar.FUSE
    axis_tuple[1].type = IterVar.FUSE
    
    axis_tuple[0].fused = fused
    axis_tuple[1].fused = fused

    fused.fused_outer = axis_tuple[0]
    fused.fused_inner = axis_tuple[1]

    for axis in tensor.axis:
        if axis is axis_tuple[0]:
            new_axis.append(fused)
        elif axis is axis_tuple[1]:
            continue
        else:
            new_axis.append(axis)
    tensor.axis = new_axis
    return fused
    
def compute_at(tensor, attach_at, axis):
    tensor.attached = True
    tensor.attach_at = attach_at
    tensor.attach_axis = axis
    axis.attached_computation.append(tensor)

# rewrite dataflow for cache_read
class RewriteDataFlowVisitor(RewriteExprVisitor):
    def __init__(self, map):
        super().__init__()
        self.map = map

    def visit_tensor_expr(self, expr):
        if expr in self.map:
            expr = self.map[expr]
        return expr

def cache_read(tensor, scope, readers):
    cache_tensor_name = tensor.name + "_" + scope
    lambda_str = "def _ ({0}): return tensor[{0}]".format(", ".join([tensor.compute_func.__code__.co_varnames[i] if tensor.compute_func is not None else 'i' + str(i) for i in range(len(tensor.shape))]))
    local_vars = {}
    exec(lambda_str, {"tensor": tensor}, local_vars)
    compiled = local_vars["_"]
    cache_tensor = compute(tensor.shape, compiled, cache_tensor_name)
    cache_tensor.scope = scope

    # rewrite dataflow from tensor -> readers to tensor -> cache_tensor -> readers
    for reader in readers:
        visitor = RewriteDataFlowVisitor({tensor: cache_tensor})
        reader.expr = visitor.rewrite(reader.expr)
    return cache_tensor

def cache_write(tensor, scope):
    pass