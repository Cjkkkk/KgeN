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

class BinaryExpr(Expr):
    def __init__(self, left, right, type):
        super().__init__()
        self.left = left
        self.right = right
        self.type = type

class VarExpr(Expr):
    def __init__(self, name):
        super().__init__()
        self.name = name

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
    return tensor

def split(tensor, axis_idx, factor):
    old_axis = tensor.axis
    new_axis = []
    for i in range(len(old_axis)):
        if i == axis_idx:
            outer = old_axis[i] // factor
            inner = factor
            new_axis.append(outer)
            new_axis.append(inner)
            tensor.producer = tensor.producer_function(outer * factor + inner)
        else:
            new_axis.append(old_axis[i])
    tensor.axis = tuple(new_axis)

def reorder(tensor, new_axis_idx):
    old_axis = tensor.axis
    new_axis = []
    for idx in new_axis_idx:
        new_axis.append(old_axis[idx])
    tensor.axis = tuple(new_axis)


def lower(expr):
    pass

if __name__ == "__main__":
    m = var("m")
    A = placeholder((m, ), name = "A")
    B = compute((m, ), lambda i: A[i] * 2, name = "B")
    split(B, 0, 32)
    reorder(B, (1, 0))
