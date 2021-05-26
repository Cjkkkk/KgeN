from .te import compute
from .tir import *
import math

# schedule primitives
def bind(ax, name):
    if name not in ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z"]:
        raise ValueError("illegal binding name {}".format(name))
    ax.bind_type = IterVar.BIND
    ax.name = name

def split(tensor, ax, factor):
    if not isinstance(tensor, TensorExpr):
        raise ValueError("Expect TensorExpr not {0}".format(type(tensor)))
    
    new_axis = []
    for axis in tensor.axis:
        if ax is axis:
            outer = IterVar(axis.name + "_outer", -math.inf, math.inf)
            inner = IterVar(axis.name + "_inner", -math.inf, math.inf)
            
            axis.outer = outer
            axis.inner = inner
            outer.parent = axis
            inner.parent = axis

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

    fused.outer = axis_tuple[0]
    fused.inner = axis_tuple[1]

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

def cache_read(tensor, scope, readers):
    cache_tensor_name = tensor.name + "_" + scope
    # TODO: fix lambda or directly insert expr to cache_tensor or copy?
    cache_tensor = TensorExpr(tensor.shape, cache_tensor_name, TensorExpr.COMPUTE)
    cache_tensor.expr = tensor[cache_tensor.root_axis]
    # call collect_input to update dataflow
    cache_tensor.collect_input()

    # rewrite tensor's outputs
    tensor.outputs = [output for output in tensor.outputs if output not in readers]

    # rewrite dataflow from tensor -> readers to tensor -> cache_tensor -> readers
    for reader in readers:
        # # rewrite inputs
        # for idx, inp in enumerate(reader.inputs):
        #     if inp is tensor:
        #         reader.inputs[idx] = cache_tensor
        # reader.providers[cache_tensor] = []
        # # copy tensor's providers to cache_tensor
        # for tensor_slice in reader.providers[tensor]:
        #     reader.providers[cache_tensor].append(cache_tensor[tensor_slice.index])
        # reader.providers.pop(tensor)
        # TODO: rewrite expr of reader and call collect_input to update dataflow
        reader.collect_input()



def cache_write(tensor, scope):
    pass