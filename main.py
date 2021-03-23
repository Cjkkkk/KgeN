ADD = 0
MUL = 1
DIV = 2
mapping = ["+", "*", "/"]

class Expr:
    def __init__(self):
        pass

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

class BinaryExpr(Expr):
    def __init__(self, left, right, type):
        super().__init__()
        self.left = left
        self.right = right
        self.type = type

    def code_gen(self):
        return "{0} {1} {2}".format(self.left.code_gen(), mapping[self.type], self.right.code_gen())

class VarExpr(Expr):
    def __init__(self, name):
        super().__init__()
        self.name = name

class ConstExpr(Expr):
    def __init__(self, val):
        super().__init__()
        self.val = val

class TensorSliceExpr(Expr):
    def __init__(self, tensor, index):
        super().__init__()
        self.tensor = tensor
        self.index = index

    def __getitem__(self, index):
        return self.tensor[index]

    def __setitem__(self, index, item):
        raise NotImplementedError
    
    def code_gen(self, ident=""):
        res = ""
        for idx, axis in enumerate(self.tensor.axis):
            res += ident + "for(int {0} = 0; {0} < {1}; {0} ++;) {{\n".format(self.tensor.name + "_iter_" + str(idx), axis.code_gen())
            ident += "    "
        return res
    
class TensorExpr(Expr):
    def __init__(self, shape, name):
        super().__init__()
        self.shape = shape
        self.name = name
        self.producer = None
        self.producer_function = None
        self.axis = self.shape

    def __getitem__(self, index):
        return TensorSliceExpr(self, index)

    def __setitem__(self, index, item):
        raise NotImplementedError
    
    def code_gen(self, ident=""):
        pass

def var(name):
    return VarExpr(name)

def placeholder(shape, name):
    return TensorExpr(shape, name)

def compute(shape, function, name):
    tensor = TensorExpr(shape, name)
    tensor.producer = function(shape)
    tensor.producer_function = function
    tensor_slice = TensorSliceExpr(tensor, shape)
    return tensor_slice

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
            if isinstance(old_axis[i], ConstExpr):
                outer = ConstExpr(old_axis[i].val // factor)
            else:
                outer = old_axis[i] // factor
            # TODO: wrap factor into axis
            inner = ConstExpr(factor)
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
    consumer.tensor.axis[axis_idx].compute = producer

def infer_bound(tensor_slice):
    # TODO: only infer when there are const axis
    pass

def lower(tensor_slice):
    infer_bound(tensor_slice)
    return tensor_slice.code_gen()

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
    print(lower(C))
