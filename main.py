ADD = 0
MUL = 1
DIV = 2

class Expr:
    def __init__(self):
        pass

    def __add__(self, other):
        return BinaryExpr(self, other, ADD)

    def __mul__(self, other):
        return BinaryExpr(self, other, MUL)

    def __floordiv__(self, other):
        return BinaryExpr(self, other, DIV)

    __radd__ = __add__

class BinaryExpr(Expr):
    def __init__(self, left, right, type):
        super().__init__()
        self.left = left
        self.right = right
        self.type = type

class VarExpr(Expr):
    def __init__(self, name="", start=0, length=-1, stride=1):
        super().__init__()
        self.name = name
        self.start = start
        self.length = length
        self.stride = stride

class TensorSliceExpr(Expr):
    def __init__(self, tensor, index):
        super().__init__()
        self.tensor = tensor
        self.index = index

class TensorExpr(Expr):
    def __init__(self, shape, name):
        super().__init__()
        self.shape = shape
        self.producer = None
        self.producer_function = None
        self.axis = self.shape

    def __getitem__(self, index):
        return TensorSliceExpr(self, index)

    def __setitem__(self, index, item):
        raise NotImplementedError

def var(name):
    return VarExpr(name)

def placeholder(shape, name):
    return TensorExpr(shape, name)

def compute(shape, function, name):
    tensor = TensorExpr(shape, name)
    tensor.producer = function(shape)
    tensor.producer_function = function
    return TensorSliceExpr(tensor, shape)

def split(tensor_slice, axis_idx, factor):
    if not isinstance(tensor_slice, TensorSliceExpr):
        raise ValueError("Expect tensor slice not {0}".format(type(tensor_slice)))
    tensor = tensor_slice.tensor
    old_axis = tensor.axis
    new_axis = []
    new_index = []
    
    if axis_idx < 0 or axis_idx >= len(old_axis):
        raise ValueError("axis_idx should be in range(0, {0})".format(len(old_axis)))
    for i in range(len(old_axis)):
        if i == axis_idx:
            # new axis
            outer = old_axis[i] // factor
            inner = VarExpr(length=factor)
            new_axis.append(outer)
            new_axis.append(inner)
            new_index.append(outer * factor + inner)
        else:
            new_axis.append(old_axis[i])
    tensor.axis = tuple(new_axis)
    tensor.producer = tensor.producer_function(new_index)
    tensor_slice.index = tuple(new_index)

def reorder(tensor_slice, new_axis_idx):
    tensor = tensor_slice.tensor
    old_axis = tensor.axis
    new_axis = []
    for idx in new_axis_idx:
        new_axis.append(old_axis[idx])
    tensor.axis = tuple(new_axis)

def compute_at(producer, consumer, axis_idx):
    pass

def infer_bound(tensor_slice):
    pass

def lower(tensor_slice):
    infer_bound(tensor_slice)

if __name__ == "__main__":
    # definition
    m = var("m")
    A = placeholder((m, ), name = "A")
    B = compute((m, ), lambda i: 2 + A[i], name = "B")
    C = compute((m, ), lambda i: 3 + B[i], name = "C")

    # schedule
    split(C, 0, 32)
    reorder(C, (1, 0))
    compute_at(B, C, 1)

    # lower
    lower(C)
