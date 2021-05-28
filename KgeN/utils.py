from .tir import *


def topo_sort(iterable, get_output):
    # Kahn's algorithm
    # calculate indegree map
    indegree_map = {}
    for node in iterable:
        indegree_map[node] = 0
    nodes_list = [*iterable]
    for node in nodes_list:
        out_nodes = get_output(node)
        for out_node in out_nodes:
            nodes_list.append(out_node)
            if out_node not in indegree_map:
                indegree_map[out_node] = 1
            else:
                indegree_map[out_node] += 1
    
    # dequeue node with indegree == 0 into solutions
    nodes_queue = [*iterable]
    solutions = []
    while len(nodes_queue) > 0:
        node = nodes_queue.pop()
        solutions.append(node)
        out_nodes = get_output(node)
        for out_node in out_nodes:
            indegree_map[out_node] -= 1
            if indegree_map[out_node] == 0:
                nodes_queue.append(out_node)
    return solutions

def tensor_topo_sort_bottom_up(tensor):
    def get_output(tensor):
        return tensor.inputs
    res = topo_sort([tensor], get_output)
    return res

def axis_topo_sort_top_down(axis_tuple):
    def get_output(axis):
        if axis.type == IterVar.SPLIT:
            return [axis.splitted_outer, axis.splitted_inner]
        elif axis.type == IterVar.FUSE:
            return [axis.fused]
        else:
            return []
    res = topo_sort(axis_tuple, get_output)
    return res

def axis_topo_sort_bottom_up(axis_tuple):
    def get_output(axis):
        if hasattr(axis, "splitted"):
            return [axis.splitted]
        elif hasattr(axis, "fused_outer"):
            return [axis.fused_outer, axis.fused_inner]
        else:
            return []
    res = topo_sort(axis_tuple, get_output)
    return res