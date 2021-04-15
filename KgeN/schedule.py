import math
from .tir import *

# schedule primitives
def bind(tensor, ax, name):
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
    # fixed_axis = []
    # for ax in consumer.axis:
    #     fixed_axis.append(ax)
    #     if ax is axis:
    #         break
    tensor.attached = True
    tensor.attach_at = attach_at
    tensor.attach_axis = axis
    # producer.fixed_axis = tuple(fixed_axis)
    axis.attached_computation.append(tensor)
