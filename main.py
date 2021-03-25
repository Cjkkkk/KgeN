ADD = 0
MUL = 1
DIV = 2
mapping = ["+", "*", "/"]

class Expr:
    def __init__(self, *subexprs):
        self.subexprs = subexprs

    def __add__(self, other):
        if isinstance(other, int):
            other = ConstExpr(other)
        return BinaryExpr(self, other, ADD)

    def __mul__(self, other):
        if isinstance(other, int):
            other = ConstExpr(other)
        return BinaryExpr(self, other, MUL)

    def __floordiv__(self, other):
        if isinstance(other, int):
            other = ConstExpr(other)
        return BinaryExpr(self, other, DIV)

    __radd__ = __add__

    def CUDA_codegen(self):
        raise NotImplementedError

class BinaryExpr(Expr):
    def __init__(self, left, right, type):
        super().__init__(left, right)
        self.left = left
        self.right = right
        self.type = type

    def CUDA_codegen(self):
        return "{0} {1} {2}".format(self.left.CUDA_codegen(), mapping[self.type], self.right.CUDA_codegen())

class VarExpr(Expr):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def CUDA_codegen(self):
        return self.name

class ConstExpr(Expr):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def CUDA_codegen(self):
        return str(self.val)

class IterVar(Expr):
    def __init__(self, name, start, length, stride=1):
        super().__init__()
        self.name = name
        self.start = ConstExpr(start) if isinstance(start, int) else start
        self.length = ConstExpr(length) if isinstance(length, int) else length
        self.stride = ConstExpr(stride) if isinstance(stride, int) else stride
        self.attached_computation = []

    def CUDA_codegen(self):
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
    
    def CUDA_codegen(self):
        return self.tensor.name + "[" + ", ".join([index.CUDA_codegen() for index in self.index]) + "]" 

class TensorExpr(Expr):
    def __init__(self, shape, name):
        super().__init__()
        self.shape = [ConstExpr(s) if isinstance(s, int) else s for s in shape]
        self.name = name
        self.axis = tuple([IterVar(name + "." + str(i), 0, v) for i, v in enumerate(self.shape)])
        self.index = self.axis
        self.producer = None
        self.producer_function = None

    def __getitem__(self, index):
        return TensorSliceExpr(self, index)

    def __setitem__(self, index, item):
        raise NotImplementedError
    
    def CUDA_codegen(self, indent=0):
        res = ""
        closing = "".join(["    " * (len(self.axis) - 1 - i + indent) + "}\n" for i in range(len(self.axis))])
        # compose loop
        for i, axis in enumerate(self.axis):
            res += "    " * (i + indent) + "for (int {0} = {1}; {0} < {2} ; {0} += {3};) {{\n".format(
                axis.name, 
                axis.start.CUDA_codegen(),
                axis.length.CUDA_codegen(), # TODO: start + length
                axis.stride.CUDA_codegen())
            for computation in axis.attached_computation:
                res += computation.CUDA_codegen(i + 1)
        # compose computation
        return res + "    " * (len(self.axis) + indent) + TensorSliceExpr(self, self.index).CUDA_codegen() + " = " + self.producer.CUDA_codegen() + ";\n" + closing

def var(name):
    return VarExpr(name)

def placeholder(shape, name):
    return TensorExpr(shape, name)

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
    tensor = TensorExpr(shape, name)
    tensor.producer = function(tensor.axis)
    tensor.producer_function = function
    tensor.producer_tensor = collect_producer_tensor(tensor.producer)
    return tensor

def split(tensor, axis_idx, factor):
    if not isinstance(tensor, TensorExpr):
        raise ValueError("Expect TensorExpr not {0}".format(type(tensor)))
    old_axis = tensor.axis
    new_axis = []
    new_index = []
    
    if axis_idx < 0 or axis_idx >= len(old_axis):
        raise ValueError("axis_idx should be in range(0, {0})".format(len(old_axis)))
    for i, axis in enumerate(old_axis):
        if i == axis_idx:
            # new axis
            outer = IterVar(axis.name + ".outer", 0, axis.length // factor if not isinstance(axis.length, ConstExpr) else ConstExpr(axis.length.val // factor))
            inner = IterVar(axis.name + ".inner", 0, ConstExpr(factor))
            new_axis.append(outer)
            new_axis.append(inner)
            new_index.append(outer * factor + inner)
        else:
            new_axis.append(axis)
    tensor.axis = tuple(new_axis)
    tensor.index = tuple(new_index)
    tensor.producer = tensor.producer_function(new_index)
    tensor.producer_tensor = collect_producer_tensor(tensor.producer)

def reorder(tensor, new_axis_idx):
    old_axis = tensor.axis
    new_axis = []
    for idx in new_axis_idx:
        new_axis.append(old_axis[idx])
    tensor.axis = tuple(new_axis)

def compute_at(producer, consumer, axis_idx):
    consumer.axis[axis_idx].attached_computation.append(producer)

def infer_bound(tensor, index):
    # TODO: only infer when there are const axis
    pass

def lower(tensor):
    # infer_bound(tensor)
    return tensor.CUDA_codegen()

if __name__ == "__main__":
    # definition
    # m = var("m")
    m = 128
    A = placeholder((m, ), name = "A")
    B = compute((m, ), lambda i: 2 + A[i], name = "B")
    C = compute((m, ), lambda i: 3 + B[i], name = "C")

    # schedule
    split(C, 0, 32)
    reorder(C, (1, 0))
    compute_at(B, C, 1)

    # lower
    print(lower(C))
