import math

def wrap_number_as_const_expr(v):
    if isinstance(v, int) or isinstance(v, float):
        return ConstExpr(v)
    else:
        return v

# Expr IR
class Expr:
    ADD = 0
    MUL = 1
    TRUE_DIV = 2
    FLOOR_DIV = 3
    SUB = 4
    MOD = 5
    GT = 6
    GE = 7
    LT = 8
    LE = 9
    MIN = 10
    MAX = 11

    mapping = ["+", "*", "/", "//", "-", "%", ">", ">=", "<", "<=", "min", "max"]
    def __init__(self, *subexprs):
        self.subexprs = subexprs

    def __add__(self, other):
        other = wrap_number_as_const_expr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val + other.val)
        elif isinstance(self, ConstExpr) and self.val == 0:
            return other
        elif isinstance(other, ConstExpr) and other.val == 0:
            return self
        else:
            return BinaryExpr(self, other, Expr.ADD)

    def __sub__(self, other):
        other = wrap_number_as_const_expr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val - other.val)
        elif isinstance(other, ConstExpr) and other.val == 0:
            raise self
        else:
            return BinaryExpr(self, other, Expr.SUB)

    def __mul__(self, other):
        other = wrap_number_as_const_expr(other)
        # folding
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val * other.val)
        elif isinstance(self, ConstExpr) and self.val == 0:
            return ConstExpr(0)
        elif isinstance(other, ConstExpr) and other.val == 0:
            return ConstExpr(0)
        else:
            return BinaryExpr(self, other, Expr.MUL)

    def __truediv__(self, other):
        other = wrap_number_as_const_expr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val / other.val)
        elif isinstance(self, ConstExpr) and self.val == 0:
            return ConstExpr(0)
        elif isinstance(other, ConstExpr) and other.val == 0:
            raise ValueError("Expr divided by 0.")
        else:
            return BinaryExpr(self, other, Expr.TRUE_DIV)

    def __floordiv__(self, other):
        other = wrap_number_as_const_expr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val // other.val)
        elif isinstance(self, ConstExpr) and self.val == 0:
            return ConstExpr(0)
        elif isinstance(other, ConstExpr) and other.val == 0:
            raise ValueError("Expr divided by 0.")
        else:
            return BinaryExpr(self, other, Expr.FLOOR_DIV)

    def __mod__(self, other):
        other = wrap_number_as_const_expr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val % other.val)
        return BinaryExpr(self, other, Expr.MOD)

    def __gt__(self, other):
        other = wrap_number_as_const_expr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(int(self.val > other.val))
        return BinaryExpr(self, other, Expr.GT)
    
    def __ge__(self, other):
        other = wrap_number_as_const_expr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(int(self.val >= other.val))
        return BinaryExpr(self, other, Expr.GE)
    
    def __lt__(self, other):
        other = wrap_number_as_const_expr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(int(self.val < other.val))
        return BinaryExpr(self, other, Expr.LT)

    def __le__(self, other):
        other = wrap_number_as_const_expr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(int(self.val <= other.val))
        return BinaryExpr(self, other, Expr.LE)
     
    __radd__ = __add__
    __rmul__ = __mul__

    @staticmethod
    def min(a, b):
        a = wrap_number_as_const_expr(a)
        b = wrap_number_as_const_expr(b)
        if isinstance(a, ConstExpr) and isinstance(b, ConstExpr):
            return ConstExpr(min(a.val, b.val))
        elif isinstance(self, ConstExpr) and self.val == math.inf:
            return other
        elif isinstance(self, ConstExpr) and self.val == -math.inf:
            return self
        elif isinstance(other, ConstExpr) and other.val == math.inf:
            return self
        elif isinstance(other, ConstExpr) and other.val == -math.inf:
            return other
        else:
            return BinaryExpr(a, b, Expr.MIN)
    
    @staticmethod
    def max(a, b):
        a = wrap_number_as_const_expr(a)
        b = wrap_number_as_const_expr(b)
        if isinstance(a, ConstExpr) and isinstance(b, ConstExpr):
            return ConstExpr(max(a.val, b.val))
        elif isinstance(self, ConstExpr) and self.val == math.inf:
            return self
        elif isinstance(self, ConstExpr) and self.val == -math.inf:
            return other
        elif isinstance(other, ConstExpr) and other.val == math.inf:
            return other
        elif isinstance(other, ConstExpr) and other.val == -math.inf:
            return self
        else:
            return BinaryExpr(a, b, Expr.MAX)

    def CUDA_codegen(self):
        raise NotImplementedError

class BinaryExpr(Expr):
    def __init__(self, left, right, type):
        super().__init__(left, right)
        self.left = left
        self.right = right
        self.type = type

    def __str__(self):
        if self.type > 9: # min, max
            return "({1}({0}, {2}))".format(self.left, Expr.mapping[self.type], self.right)
        return "({0} {1} {2})".format(self.left, Expr.mapping[self.type], self.right)

    def CUDA_codegen(self):
        if self.type > 9: # min, max
            return "({1}({0}, {2}))".format(self.left.CUDA_codegen(), Expr.mapping[self.type], self.right.CUDA_codegen())
        return "({0} {1} {2})".format(self.left.CUDA_codegen(), Expr.mapping[self.type], self.right.CUDA_codegen())

class VarExpr(Expr):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __str__(self):
        return self.name

    def CUDA_codegen(self):
        return self.name

class ConstExpr(Expr):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def __str__(self):
        return str(self.val)

    def CUDA_codegen(self):
        return str(self.val)

class RangeType:
    CLOSED_OPEN = 0
    CLOSED_CLOSED = 1

class Range:
    def __init__(self, start, end, type_= RangeType.CLOSED_OPEN):
        self.start = wrap_number_as_const_expr(start)
        self.end = wrap_number_as_const_expr(end)
        self.is_single_point = False
        self.type = type_

    @staticmethod
    def single_point(expr):
        interval = Range(expr, expr, RangeType.CLOSED_CLOSED)
        interval.is_single_point = True
        return interval

class IterVar(Expr):
    NORMAL = 0
    SPLIT = 1
    FUSE = 2
    REDUCE = 3

    NONE = 3
    BIND = 4
    def __init__(self, name, start, end):
        super().__init__()
        self.name = name
        self.range = Range(start, end)
        self.attached_computation = []
        self.type = IterVar.NORMAL
        self.bind_type = IterVar.NONE
        self.bind_name = ""

    def __str__(self):
        return "{0}: [{1}, {2} {3}".format(self.name, self.range.start, self.range.end, "]" if self.range.type == RangeType.CLOSED_CLOSED else ")")

    def CUDA_codegen(self):
        if self.type == IterVar.SPLIT:
            return "(({0} * {1}) + {2})".format(self.outer.CUDA_codegen(), self.inner.range.end.CUDA_codegen(), self.inner.CUDA_codegen())
        elif self.type == IterVar.FUSE:
            if self is self.fused.outer:
                return "({0} // {1})".format(self.fused.CUDA_codegen(), self.fused.inner.range.end.CUDA_codegen())
            else:
                return "({0} % {1})".format(self.fused.CUDA_codegen(), self.fused.inner.range.end.CUDA_codegen())
        else:
            if self.range.is_single_point:
                return self.range.start.CUDA_codegen()
            elif self.bind_type == IterVar.BIND:
                return self.bind_name
            else:
                return self.name

def reduce_axis(end, name):
    axis = IterVar(name, 0, end)
    axis.type = IterVar.REDUCE
    return axis

class ReduceExpr(Expr):
    def __init__(self, combinator, init, expr, axis):
        super().__init__()
        self.combinator = wrap_number_as_const_expr(combinator)
        self.init = wrap_number_as_const_expr(init)
        self.expr = wrap_number_as_const_expr(expr)
        self.reduce_axis = axis if isinstance(axis, tuple) else (axis, )
    
    def __str__(self):
        raise NotImplementedError

    def CUDA_codegen(self):
        raise NotImplementedError
 
class IfThenElseExpr(Expr):
    def __init__(self, condition, then_expr, else_expr):
        super().__init__()
        self.condition = wrap_number_as_const_expr(condition)
        self.then_expr = wrap_number_as_const_expr(then_expr)
        self.else_expr = wrap_number_as_const_expr(else_expr)

    def __str__(self):
        return "({0} ? {1} : {2})".format(str(self.condition), str(self.then_expr), str(self.else_expr))
    
    def CUDA_codegen(self):
        return "({0} ? {1} : {2})".format(self.condition.CUDA_codegen(), self.then_expr.CUDA_codegen(), self.else_expr.CUDA_codegen())

class TensorSliceExpr(Expr):
    def __init__(self, tensor, index):
        super().__init__()
        self.tensor = tensor
        self.index = index

    def __getitem__(self, index):
        raise NotImplementedError
        # return self.tensor[index]

    def __setitem__(self, index, item):
        raise NotImplementedError
    
    def __str__(self):
        return self.tensor.name + "[" + ", ".join([str(index) for index in self.index]) + "]" 

    def CUDA_codegen(self):
        return self.tensor.name + "[" + ", ".join([index.CUDA_codegen() for index in self.index]) + "]" 

class TensorExpr(Expr):
    PLACEHOLDER = 0
    COMPUTE = 1
    def __init__(self, shape, name, tensor_type, compute_func=None):
        super().__init__()
        self.shape = tuple([ConstExpr(s) if isinstance(s, int) else s for s in shape])
        self.name = name
        self.type = tensor_type
        self.compute_func = compute_func
        self.attached = False
        self.inputs = []
        self.consumers = []
        self.axis = ()
        self.root_axis = ()
        self.fixed_axis = ()
        
        if tensor_type == TensorExpr.COMPUTE:
            self.axis = tuple([IterVar(self.name + "_" + compute_func.__code__.co_varnames[i], 0, v) for i, v in enumerate(self.shape)])
            self.root_axis = self.axis
            self.expr = compute_func(*self.axis)
            self.inputs = collect_inputs(self.expr)

            if isinstance(self.expr, ReduceExpr):
                self.axis = tuple(self.axis + self.expr.reduce_axis)
                self.expr = self.expr.combinator(TensorSliceExpr(self, self.root_axis), self.expr.expr)
    
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index, )
        tensor_slice = TensorSliceExpr(self, index)
        self.consumers.append(tensor_slice)
        return tensor_slice

    def __setitem__(self, index, item):
        raise NotImplementedError
    
    def __str__(self):
        return self.name

    def CUDA_codegen(self, indent=0):
        if self.type == TensorExpr.PLACEHOLDER:
            return ""
        opening = ""
        closing = ""
        scope = indent
        # TODO: find out which axis to do reduce init
        # compose loop
        for i, axis in enumerate(self.axis):
            if not axis.range.is_single_point and not axis.bind_type == IterVar.BIND:
                opening += "    " * scope + "for (int {0} = {1}; {0} < {2} ; {0} += {3};) {{\n".format(
                    axis.name, 
                    axis.range.start.CUDA_codegen(),
                    axis.range.end.CUDA_codegen(),
                    1)
                closing = "    " * scope + "}\n" + closing
                scope += 1
            
            for computation in axis.attached_computation:
                opening += computation.CUDA_codegen(scope)
        
        body = "    " * scope + TensorSliceExpr(self, self.root_axis).CUDA_codegen() + " = " + self.expr.CUDA_codegen() + ";\n"
        return opening + body + closing


# compute primitives
def var(name):
    return VarExpr(name)

def placeholder(shape, name):
    return TensorExpr(shape, name, TensorExpr.PLACEHOLDER)

def collect_inputs(producer):
    inputs = set()
    q = [producer]
    while len(q) > 0:
        expr = q.pop()
        if isinstance(expr, TensorSliceExpr):
            inputs.add(expr.tensor)
            for index in expr.index:
                q.append(index)
        if isinstance(expr, BinaryExpr):
            for subexpr in expr.subexprs:
                q.append(subexpr)
        if isinstance(expr, IfThenElseExpr):
            q.append(expr.condition)
            q.append(expr.then_expr)
            q.append(expr.else_expr)
        if isinstance(expr, ReduceExpr):
            q.append(expr.expr)
    return list(inputs)

def compute(shape, function, name):
    tensor = TensorExpr(shape, name, TensorExpr.COMPUTE, function)
    return tensor

def if_then_else(condition, then_expr, else_expr):
    return IfThenElseExpr(condition, then_expr, else_expr)
   
def reduce_sum(expr, axis):
    combinator = lambda x, y: x + y
    return ReduceExpr(combinator, 0, expr, axis)

def reduce_max(expr, axis):
    combinator = lambda x, y: Expr.max(x, y)
    return ReduceExpr(combinator, -math.inf, expr, axis)

def reduce_min(expr, axis):
    combinator = lambda x, y: Expr.min(x, y)
    return ReduceExpr(combinator, math.inf, expr, axis)

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
            axis.factor = factor
            axis.type = IterVar.SPLIT
            new_axis.append(outer)
            new_axis.append(inner)
        else:
            new_axis.append(axis)
    tensor.axis = tuple(new_axis)
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
    tensor.axis = tuple(new_axis_list)

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
    tensor.axis = tuple(new_axis)
    return fused
    
def compute_at(producer, consumer, axis):
    fixed_axis = []
    for ax in consumer.axis:
        fixed_axis.append(ax)
        if ax is axis:
            break
    producer.attached = True
    producer.fixed_axis = tuple(fixed_axis)
    axis.attached_computation.append(producer)

# bound inference and codegen
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

def tensor_topo_sort(tensor):
    def get_output(tensor):
        return tensor.inputs
    res = topo_sort([tensor], get_output)
    return res

def axis_topo_sort(axis_tuple):
    def get_output(axis):
        if axis.type == IterVar.SPLIT:
            return [axis.outer, axis.inner]
        elif axis.type == IterVar.FUSE:
            return [axis.fused]
        else:
            return []
    res = topo_sort(axis_tuple, get_output)
    return res

def infer_root_iter_bound(tensor, rmap):
    if len(tensor.consumers) > 0:
        bounds = None
        for consumer in tensor.consumers:
            new_bounds = [evaluate_expr_bound(index, tensor.fixed_axis) for index in consumer.index]
            if bounds is None:
                bounds = new_bounds
            else:
                # TODO: Test consolidate bounds
                for i, new_bound in enumerate(new_bounds):
                    bounds[i] = Range(
                            Expr.min(bounds[i].start, new_bound.start), 
                            Expr.max(bounds[i].end, new_bound.end)
                        )
        for i, root_axis in enumerate(tensor.root_axis):
            root_axis.range = bounds[i]
            rmap[root_axis] = bounds[i]
    else:
        # is output tensor, therefore no consumers
        for root_axis in tensor.root_axis:
            rmap[root_axis] = root_axis.range

def pass_down(tensor, rmap):
    for axis in rmap:
        if axis.type == IterVar.SPLIT:
            # TODO: fix this: should be ceil div
            rmap[axis.outer] = Range(0, rmap[axis].end // axis.factor)
            rmap[axis.inner] = Range(0, axis.factor)
            axis.outer.range = rmap[axis.outer]
            axis.inner.range = rmap[axis.inner]
        elif axis.type == IterVar.FUSE:
            rmap[axis.fused] = Range(0, rmap[axis.fused.outer].end * rmap[axis.fused.inner].end)
            axis.fused.range = rmap[axis.fused]
        else:
            # we already know root_axis's range
            pass

def get_rmap(tensor):
    rmap = {}
    axis_res = axis_topo_sort(tensor.root_axis)
    for axis in axis_res:
        rmap[axis] = axis.range
    return rmap

def infer_bound_pass(tensor):
    tensors = tensor_topo_sort(tensor)
    for tensor in tensors:
        if tensor.type != TensorExpr.PLACEHOLDER: # we don't care about placeholder's bound
            rmap = get_rmap(tensor)
            infer_root_iter_bound(tensor, rmap)
            pass_down(tensor, rmap)

def evaluate_expr_bound(expr, fixed_axis):
    if isinstance(expr, IterVar):
        if expr.type == IterVar.SPLIT:
            interval = evaluate_expr_bound(expr.outer * expr.inner.range.end + expr.inner, fixed_axis)
        elif expr.type == IterVar.FUSE:
            if expr is expr.fused.outer:
                interval = evaluate_expr_bound(expr.fused // expr.fused.inner.range.end, fixed_axis)
            else:
                interval = evaluate_expr_bound(expr.fused % expr.fused.inner.range.end, fixed_axis)
        elif expr in fixed_axis:
            interval = Range.single_point(expr)
        else:
            interval = Range(expr.range.start, expr.range.end)
    elif isinstance(expr, BinaryExpr):
        # TODO: fix corner cases
        left = evaluate_expr_bound(expr.left, fixed_axis)
        right = evaluate_expr_bound(expr.right, fixed_axis)
        if expr.type == Expr.ADD:
            if left.is_single_point and right.is_single_point:
                interval = Range.single_point(left.start + right.start)
            else:
                interval = Range(left.start + right.start, left.end + right.end)
        elif expr.type == Expr.SUB:
            if left.is_single_point and right.is_single_point:
                interval = Range.single_point(left.start - right.start)
            else:
                interval = Range(left.start - right.start, left.end - right.end)
        elif expr.type == Expr.MUL:
            if left.is_single_point and right.is_single_point:
                interval = Range.single_point(left.start * right.start)
            elif not left.is_single_point and not right.is_single_point:
                interval = Range(left.start * right.start, (left.end - 1) * (right.end - 1))
            elif left.is_single_point and not right.is_single_point:
                interval = Range(left.start * right.start, left.end * (right.end - 1))
            else:
                interval = Range(left.start * right.start, (left.end -1 ) * right.end)
        elif expr.type == Expr.FLOOR_DIV:
            if left.is_single_point and right.is_single_point:
                interval = Range.single_point(left.start // right.start)
            elif not left.is_single_point and not right.is_single_point:
                interval = Range(left.start // right.start, (left.end - 1) // (right.end - 1))
            else:
                interval = Range(left.start // right.start, left.end // right.end)
        elif expr.type == Expr.MOD:
            if left.is_single_point and right.is_single_point:
                interval = Range.single_point(left.start % right.start)
            elif not left.is_single_point and right.is_single_point:
                interval = Range(left.start % right.start, left.end % right.end)
            else:
                raise ValueError("Should not be here.")
        else:
            raise ValueError("Unsupported type {}.".format(expr.type))
    elif isinstance(expr, IfThenElseExpr):
        # TODO: fix ifThenElseExpr
        then_interval = evaluate_expr_bound(expr.then_expr)
        else_interval = evaluate_expr_bound(expr.else_expr)
    elif isinstance(expr, TensorSliceExpr):
        # TODO: fix TensorSliceExpr
        pass
    elif isinstance(expr, ConstExpr) or isinstance(expr, VarExpr):
        interval = Range.single_point(expr)
    else:
        print(expr)
        raise ValueError("Unsupported expr type {}".format(type(expr)))
    return interval

def CUDA_codegen_pass(tensor):
    tensors = tensor_topo_sort(tensor)
    res = ""
    for t in reversed(tensors):
        # skip codegen if it is attached to some axis
        if not t.attached:
            res += t.CUDA_codegen()
    return res

def lower(tensor):
    infer_bound_pass(tensor)
    return CUDA_codegen_pass(tensor)