from .tir import *

def topo_sort(iterable, get_output):
    # Kahn's algorithm
    # calculate indegree map
    indegree_map = {}
    visited = set()
    for node in iterable:
        indegree_map[node] = 0
        visited.add(node)
    node_q = [*iterable]
    
    while len(node_q) > 0:
        node = node_q.pop()
        out_nodes = get_output(node)
        for out_node in out_nodes:
            if out_node not in visited:
                visited.add(out_node)
                node_q.append(out_node)
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
        if axis.relation == IterVar.SPLIT:
            return [axis.splitted_outer, axis.splitted_inner]
        elif axis.relation == IterVar.FUSE:
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

def index_flatten(index, shape):
    def f(a, b):
        return a * b
    def scan(f, state, l):
        res = []
        for e in l:
            state = f(state, e)
            res.append(state)
        return res
    
    prod = scan(f, ConstExpr(1), reversed(shape[1:] + (ConstExpr(1),)))
    prod = reversed(prod)
    flatten_index = 0

    for index, prod in zip(index, prod):
        flatten_index = flatten_index + index * prod
    return flatten_index