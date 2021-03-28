class Expr:
    ADD = 0
    MUL = 1
    DIV = 2
    SUB = 3
    MOD = 4
    mapping = ["+", "*", "/", "-", "%"]
    def __init__(self, *subexprs):
        self.subexprs = subexprs

    def __add__(self, other):
        if isinstance(other, int):
            other = ConstExpr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val + other.val)
        return BinaryExpr(self, other, Expr.ADD)

    def __mul__(self, other):
        if isinstance(other, int):
            other = ConstExpr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val * other.val)
        return BinaryExpr(self, other, Expr.MUL)

    def __floordiv__(self, other):
        if isinstance(other, int):
            other = ConstExpr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val // other.val)
        return BinaryExpr(self, other, Expr.DIV)
    
    def __sub__(self, other):
        if isinstance(other, int):
            other = ConstExpr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val - other.val)
        return BinaryExpr(self, other, Expr.SUB)

    def __mod__(self, other):
        if isinstance(other, int):
            other = ConstExpr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val % other.val)
        return BinaryExpr(self, other, Expr.MOD)
    __radd__ = __add__
    __rmul__ = __mul__

    def CUDA_codegen(self):
        raise NotImplementedError

class BinaryExpr(Expr):
    def __init__(self, left, right, type):
        super().__init__(left, right)
        self.left = left
        self.right = right
        self.type = type

    def __str__(self):
        return "({0} {1} {2})".format(self.left, Expr.mapping[self.type], self.right)

    def CUDA_codegen(self):
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

class Range:
    def __init__(self, start, end, stride=1):
        self.start = ConstExpr(start) if isinstance(start, int) else start
        self.end = ConstExpr(end) if isinstance(end, int) else end
        self.stride = ConstExpr(stride) if isinstance(stride, int) else stride

class IterVar(Expr):
    NORMAL = 0
    SPLIT = 1
    FUSE = 2
    def __init__(self, name, start, end, stride=1):
        super().__init__()
        self.name = name
        self.range = Range(start, end, stride)
        self.attached_computation = []
        self.type = IterVar.NORMAL

    def __str__(self):
        return "{0}: [{1}, {2})".format(self.name, self.range.start, self.range.end)

    def CUDA_codegen(self):
        if self.type == IterVar.SPLIT:
            return "(({0} * {1}) + {2})".format(self.outer.CUDA_codegen(), self.inner.range.end.CUDA_codegen(), self.inner.CUDA_codegen())
        if self.type == IterVar.FUSE:
            if self is self.fused.outer:
                return "({0} // {1})".format(self.fused.CUDA_codegen(), self.fused.inner.range.end.CUDA_codegen())
            else:
                return "({0} % {1})".format(self.fused.CUDA_codegen(), self.fused.outer.range.end.CUDA_codegen())
        else:
            return self.name

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
            self.index = self.axis
            self.expr = compute_func(self.axis)
            self.inputs = collect_inputs(self.expr)

    def __getitem__(self, index):
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
        # compose loop
        for i, axis in enumerate(self.axis):
            opening += "    " * (i + indent) + "for (int {0} = {1}; {0} < {2} ; {0} += {3};) {{\n".format(
                axis.name, 
                axis.range.start.CUDA_codegen(),
                axis.range.end.CUDA_codegen(),
                axis.range.stride.CUDA_codegen())
            
            for computation in axis.attached_computation:
                opening += computation.CUDA_codegen(i + 1)
            
            closing += "    " * (len(self.axis) - i - 1 + indent) + "}\n"
        body = "    " * (len(self.axis) + indent) + TensorSliceExpr(self, self.index).CUDA_codegen() + " = " + self.expr.CUDA_codegen() + ";\n"
        return opening + body + closing

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
    # for i in inputs:
    #     print(i)
    return list(inputs)

def compute(shape, function, name):
    tensor = TensorExpr(shape, name, TensorExpr.COMPUTE, function)
    return tensor

def split(tensor, ax, factor):
    new_axis = []
    if not isinstance(tensor, TensorExpr):
        raise ValueError("Expect TensorExpr not {0}".format(type(tensor)))
    for axis in tensor.axis:
        if ax is axis:
            outer = IterVar(axis.name + "_outer", 0, 0)
            inner = IterVar(axis.name + "_inner", 0, 0)
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

def reorder(tensor, axis_tuple):
    if len(axis_tuple) != len(tensor.axis):
        raise ValueError("should provide {0} axis".format(len(axis_tuple)))
    tensor.axis = tuple(axis_tuple)

def fuse(tensor, axis_tuple):
    new_axis = []
    # set axis to fuse
    fused = IterVar(axis_tuple[0].name + "_" + axis_tuple[1].name + "_fused", 0, 0)
    
    axis_tuple[0].type = IterVar.FUSE
    axis_tuple[1].type = IterVar.FUSE
    
    axis_tuple[0].fused = fused
    axis_tuple[1].fused = fused

    fused.outer = axis_tuple[0]
    fused.inner = axis_tuple[1]

    # TODO: fix range of fused
    for axis in tensor.axis:
        if not axis in axis_tuple:
            new_axis.append(axis)
        if axis is axis_tuple[0]:
            new_axis.append(fused)
    tensor.axis = tuple(new_axis)
    return new_axis
    
def compute_at(producer, consumer, axis):
    fixed_axis = []
    for ax in consumer.axis:
        fixed_axis.append(ax)
        if ax is axis:
            break
    producer.attached = True
    producer.fixed_axis = tuple(fixed_axis)
    axis.attached_computation.append(producer)


def topo_sort(tensor):
    res = [tensor]
    q = [tensor]
    visited = {tensor}
    while len(q) > 0:
        t = q.pop()
        for input_tensor in t.inputs:
            if input_tensor not in visited:
                visited.add(input_tensor)
                res.append(input_tensor)
                q.append(input_tensor)
    return res

def infer_root_iter_bound(tensor, rmap):
    if tensor.type != TensorExpr.PLACEHOLDER: # we don't care about placeholder's axis
        if len(tensor.consumers) > 0:
            bounds = None
            for consumer in tensor.consumers:
                new_bounds = [evaluate_expr_bound(index, tensor.fixed_axis) for index in consumer.index]
            if bounds is None:
                bounds =  new_bounds
            else:
                # TODO: consolidate bounds
                # for i, bound in enumerate(new_bounds):
                #     bounds[i] = range(min(), max(), 1)
                pass
            for i, root_axis in enumerate(tensor.root_axis):
                root_axis.range = bounds[i]
                rmap[root_axis] = bounds[i]
        else:
            # is output tensor, therefor no consumers
            for root_axis in tensor.root_axis:
                rmap[root_axis] = root_axis.range

def pass_down(tensor, rmap):
    if tensor.type != TensorExpr.PLACEHOLDER: # we don't care about placeholder's axis
        for axis in rmap:
            if axis.type == IterVar.SPLIT:
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
    if tensor.type != TensorExpr.PLACEHOLDER:
        for axis in tensor.root_axis:
            rmap[axis] = axis.range

    q = list(rmap.keys())
    while len(q) > 0:
        axis = q.pop()
        if axis.type == IterVar.SPLIT:
            rmap[axis.outer] = axis.outer.range
            rmap[axis.inner] = axis.inner.range
            q.append(axis.outer)
            q.append(axis.inner)
        elif axis.type == IterVar.FUSE:
            rmap[axis.fused] = axis.fused.range
            q.append(axis.fused)
        else:
            pass
    return rmap

def infer_bound_pass(tensor):
    tensors = topo_sort(tensor)
    for tensor in tensors:
        rmap = get_rmap(tensor)
        infer_root_iter_bound(tensor, rmap)
        pass_down(tensor, rmap)


def CUDA_codegen_pass(tensor):
    tensors = topo_sort(tensor)
    res = ""
    for t in reversed(tensors):
        # skip codegen if it is attached to some axis
        if not t.attached:
            res += t.CUDA_codegen()
    return res

def evaluate_expr_bound(expr, fixed_axis):
    if isinstance(expr, IterVar):
        if expr.type == IterVar.SPLIT:
            interval = evaluate_expr_bound(expr.outer * expr.inner.range.end + expr.inner, fixed_axis)
        elif expr in fixed_axis:
            interval = Range(expr, expr + 1)
        else:
            interval = Range(expr.range.start, expr.range.end)
    elif isinstance(expr, BinaryExpr):
        left = evaluate_expr_bound(expr.left, fixed_axis)
        right = evaluate_expr_bound(expr.right, fixed_axis)
        if expr.type == Expr.ADD:
            interval = Range(left.start + right.start, left.end + right.end)
        elif expr.type == Expr.SUB:
            interval = Range(left.start - right.start, left.end - right.end)
        elif expr.type == Expr.MUL:
            interval = Range(left.start * right.start, (left.end - 1) * (right.end - 1))
        else:
            raise ValueError("Unsupported type.")
    else:
        interval = Range(expr, expr + 1)
    return interval


def lower(tensor):
    infer_bound_pass(tensor)
    return CUDA_codegen_pass(tensor)

if __name__ == "__main__":
    # definition
    # m = var("m")
    m = 128
    A = placeholder((m, ), name = "A")
    B = compute((m, ), lambda i: 2 + A[i], name = "B")
    C = compute((m, ), lambda i: 3 + B[i], name = "C")
    # schedule
    outer, inner = split(C, C.axis[0], 32)
    # fused = fuse(C, (outer, inner))
    # reorder(C, (inner, outer))
    compute_at(B, C, outer)

    # lower
    print(lower(C))
