class Expr:
    ADD = 0
    MUL = 1
    DIV = 2
    SUB = 3
    mapping = ["+", "*", "/", "-"]
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
        return "{0} {1} {2}".format(self.left, Expr.mapping[self.type], self.right)

    def CUDA_codegen(self):
        return "{0} {1} {2}".format(self.left.CUDA_codegen(), Expr.mapping[self.type], self.right.CUDA_codegen())

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

class IterVar(Expr):
    NORMAL = 0
    SPLIT = 1
    def __init__(self, name, start, end, stride=1):
        super().__init__()
        self.name = name
        self.start = ConstExpr(start) if isinstance(start, int) else start
        self.end = ConstExpr(end) if isinstance(end, int) else end
        self.stride = ConstExpr(stride) if isinstance(stride, int) else stride
        self.attached_computation = []
        self.sub_axis = []
        self.type = IterVar.NORMAL

    def __str__(self):
        return "{0}: [{1}, {2})".format(self.name, self.start, self.end)

    def CUDA_codegen(self):
        if self.type == IterVar.SPLIT:
            return "{0} * {1} + {2}".format(self.sub_axis[0].CUDA_codegen(), self.sub_axis[1].end.CUDA_codegen(), self.sub_axis[1].CUDA_codegen())
        else:
            return self.name

class TensorSliceExpr(Expr):
    def __init__(self, tensor, index):
        super().__init__()
        self.tensor = tensor
        self.index = index

    def __getitem__(self, index):
        return self.tensor[index]

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
        self.tensor_type = tensor_type
        self.compute_func = compute_func
        if tensor_type == TensorExpr.COMPUTE:
            self.axis = tuple([IterVar(self.name + "_" + compute_func.__code__.co_varnames[i], 0, v) for i, v in enumerate(self.shape)])
            self.old_axis = self.axis
            self.index = self.axis
            self.producer = compute_func(self.axis)

    def __getitem__(self, index):
        return TensorSliceExpr(self, index)

    def __setitem__(self, index, item):
        raise NotImplementedError
    
    def __str__(self):
        return self.tensor.name

    def CUDA_codegen(self, indent=0):
        res = ""
        # TODO: fixed this
        closing = "".join(["    " * (len(self.axis) - 1 - i + indent) + "}\n" for i, axis in enumerate(self.axis)])
        # compose loop
        for i, axis in enumerate(self.axis):
            res += "    " * (i + indent) + "for (int {0} = {1}; {0} < {2} ; {0} += {3};) {{\n".format(
                axis.name, 
                axis.start.CUDA_codegen(),
                axis.end.CUDA_codegen(),
                axis.stride.CUDA_codegen())
            for computation in axis.attached_computation:
                res += computation.CUDA_codegen(i + 1)
        # compose computation
        return res + "    " * (len(self.axis) + indent) + TensorSliceExpr(self, self.index).CUDA_codegen() + " = " + self.producer.CUDA_codegen() + ";\n" + closing

def var(name):
    return VarExpr(name)

def placeholder(shape, name):
    return TensorExpr(shape, name, TensorExpr.PLACEHOLDER)

def collect_producer_tensor(producer):
    res = []
    q = [producer]
    while len(q) > 0:
        expr = q.pop()
        if isinstance(expr, TensorSliceExpr):
            res.append(expr)
        for subexpr in expr.subexprs:
            q.append(subexpr)
    return res

def compute(shape, function, name):
    tensor = TensorExpr(shape, name, TensorExpr.COMPUTE, function)
    tensor.producer_tensor = collect_producer_tensor(tensor.producer)
    return tensor

def split(tensor, axis_idx, factor):
    new_axis = []
    if not isinstance(tensor, TensorExpr):
        raise ValueError("Expect TensorExpr not {0}".format(type(tensor)))
    for i, axis in enumerate(tensor.axis):
        if axis_idx == i:
            outer = IterVar(axis.name + "_outer", 0, axis.end // factor)
            inner = IterVar(axis.name + "_inner", 0, factor)
            axis.sub_axis.append(outer)
            axis.sub_axis.append(inner)
            axis.type = IterVar.SPLIT
            new_axis.append(outer)
            new_axis.append(inner)
        else:
            new_axis.append(axis)
    tensor.axis = tuple(new_axis)
    return outer, inner

def reorder(tensor, axis):
    if len(axis) != len(tensor.axis):
        raise ValueError("should provide {0} axis".format(len(axis)))
    tensor.axis = tuple(axis)

def compute_at(producer, consumer, axis):
    axis.attached_computation.append(producer)

def infer_bound(tensor):
    fixed_axis = []
    for idx, axis in enumerate(tensor.axis):
        fixed_axis.append(axis)
        for computation in axis.attached_computation:
            for producer_tensor in tensor.producer_tensor:
                if producer_tensor.tensor is computation:
                    intervals = [evaluate_expr_range(index, fixed_axis) for index in producer_tensor.index]
                    # Adjust original axis interval
                    for p_i, p_axis in enumerate(producer_tensor.tensor.old_axis):
                        p_axis.start = intervals[p_i].start
                        p_axis.end = intervals[p_i].end

def evaluate_expr_range(expr, fixed_axis):
    if isinstance(expr, IterVar):
        if expr.type == IterVar.SPLIT:
            interval = evaluate_expr_range(expr.sub_axis[0] * expr.sub_axis[1].end + expr.sub_axis[1], fixed_axis)
        elif expr in fixed_axis:
            interval = IterVar("", expr, expr + 1)
        else:
            interval = IterVar("", expr.start, expr.end)
    elif isinstance(expr, BinaryExpr):
        left = evaluate_expr_range(expr.left, fixed_axis)
        right = evaluate_expr_range(expr.right, fixed_axis)
        if expr.type == Expr.ADD:
            interval = IterVar("", left.start + right.start, left.end + right.end)
        elif expr.type == Expr.SUB:
            interval = IterVar("", left.start - right.start, left.end - right.end)
        elif expr.type == Expr.MUL:
            interval = IterVar("", left.start * right.start, (left.end - 1) * (right.end - 1))
        else:
            raise ValueError("Unsupported type.")
    else:
        interval = IterVar("", expr, expr + 1)
    return interval


def lower(tensor):
    infer_bound(tensor)
    return tensor.CUDA_codegen()

if __name__ == "__main__":
    # definition
    # m = var("m")
    m = 128
    A = placeholder((m, ), name = "A")
    B = compute((m, ), lambda i: 2 + A[i], name = "B")
    C = compute((m, ), lambda i: 3 + B[i], name = "C")

    # schedule
    outer, inner = split(C, 0, 32)
    # reorder(C, (inner, outer))
    compute_at(B, C, outer)

    # lower
    print(lower(C))
