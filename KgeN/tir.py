import math

# Expr IR
def wrap_number_as_const_expr(v):
    if isinstance(v, int) or isinstance(v, float):
        return ConstExpr(v)
    else:
        return v

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
    CEIL_DIV = 12
    NEG = 13
    mapping = ["+", "*", "/", "//", "-", "%", ">", ">=", "<", "<=", "min", "max", "ceildiv", "-"]
    
    def __init__(self, *subexprs):
        self.subexprs = subexprs

    def __add__(self, other):
        # TODO: add expr simpifier here
        other = wrap_number_as_const_expr(other)
        if isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val + other.val)
        elif isinstance(self, ConstExpr) and self.val == 0:
            return other
        elif isinstance(other, ConstExpr) and other.val == 0:
            return self
        elif isinstance(other, ConstExpr) and other.val < 0:
            return BinaryExpr(self, ConstExpr(-other.val), Expr.SUB)
        elif isinstance(other, UnaryExpr) and other.type < Expr.NEG:
            return BinaryExpr(self, other.expr, Expr.SUB)
        else:
            return BinaryExpr(self, other, Expr.ADD)

    def __sub__(self, other):
        other = wrap_number_as_const_expr(other)
        if self.same_as(other):
            return ConstExpr(0)
        elif isinstance(self, ConstExpr) and isinstance(other, ConstExpr):
            return ConstExpr(self.val - other.val)
        elif isinstance(other, ConstExpr) and other.val == 0:
            return self
        elif isinstance(other, ConstExpr) and other.val < 0:
            return BinaryExpr(self, ConstExpr(-other.val), Expr.ADD)
        elif isinstance(other, UnaryExpr) and other.type < Expr.NEG:
            return BinaryExpr(self, other.expr, Expr.ADD)
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

    def __neg__(self):
        if isinstance(self, ConstExpr):
            return ConstExpr(- self.val)
        return UnaryExpr(self, Expr.NEG)

    @staticmethod
    def min(a, b):
        a = wrap_number_as_const_expr(a)
        b = wrap_number_as_const_expr(b)
        if a.same_as(b):
            return a
        if isinstance(a, ConstExpr) and isinstance(b, ConstExpr):
            return ConstExpr(min(a.val, b.val))
        elif isinstance(a, ConstExpr) and a.val == math.inf:
            return b
        elif isinstance(a, ConstExpr) and a.val == -math.inf:
            return a
        elif isinstance(b, ConstExpr) and b.val == math.inf:
            return a
        elif isinstance(b, ConstExpr) and b.val == -math.inf:
            return b
        else:
            return BinaryExpr(a, b, Expr.MIN)
    
    @staticmethod
    def max(a, b):
        a = wrap_number_as_const_expr(a)
        b = wrap_number_as_const_expr(b)
        if a.same_as(b):
            return a
        if isinstance(a, ConstExpr) and isinstance(b, ConstExpr):
            return ConstExpr(max(a.val, b.val))
        elif isinstance(a, ConstExpr) and a.val == math.inf:
            return a
        elif isinstance(a, ConstExpr) and a.val == -math.inf:
            return b
        elif isinstance(b, ConstExpr) and b.val == math.inf:
            return b
        elif isinstance(b, ConstExpr) and b.val == -math.inf:
            return a
        else:
            return BinaryExpr(a, b, Expr.MAX)

    @staticmethod
    def ceildiv(a, b):
        a = wrap_number_as_const_expr(a)
        b = wrap_number_as_const_expr(b)
        if isinstance(a, ConstExpr) and isinstance(b, ConstExpr):
            return ConstExpr(math.ceil(a.val / b.val))
        else:
            return BinaryExpr(a, b, Expr.CEIL_DIV)

    def same_as(self, other):
        raise NotImplementedError

    def CUDA_codegen(self):
        raise NotImplementedError

Expr.function_mapping = [Expr.__add__, Expr.__mul__, Expr.__truediv__, Expr.__floordiv__, Expr.__sub__, Expr.__mod__, Expr.__gt__,
        Expr.__ge__, Expr.__lt__, Expr.__le__, Expr.min, Expr.max, Expr.ceildiv, Expr.__neg__]

class UnaryExpr(Expr):
    def __init__(self, expr, type_):
        super().__init__(expr)
        self.expr = expr
        self.type = type_

    def __str__(self):
        return "({0}({1}))".format(Expr.mapping[self.type], self.expr)

    def same_as(self, other):
        return isinstance(other, UnaryExpr) and self.type == other.type and self.expr.same_as(other.expr)

    def CUDA_codegen(self):
        return "({0}({1}))".format(Expr.mapping[self.type], self.expr.CUDA_codegen())


class BinaryExpr(Expr):
    def __init__(self, left, right, type_):
        super().__init__(left, right)
        self.left = left
        self.right = right
        self.type = type_

    def __str__(self):
        if self.type > 9: # min, max
            return "({1}({0}, {2}))".format(self.left, Expr.mapping[self.type], self.right)
        return "({0} {1} {2})".format(self.left, Expr.mapping[self.type], self.right)

    def same_as(self, other):
        return self is other or (isinstance(other, BinaryExpr) and self.type == other.type and self.left.same_as(other.left) and self.right.same_as(other.right))

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

    def same_as(self, other):
        return self is other or (isinstance(other, VarExpr) and self.name == other.name)
    
    def CUDA_codegen(self):
        return self.name

class ConstExpr(Expr):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def __str__(self):
        return str(self.val)

    def same_as(self, other):
        return self is other or (isinstance(other, ConstExpr) and self.val == other.val)

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

        if self.type == RangeType.CLOSED_CLOSED and self.start.same_as(self.end):
            # TODO: fix this, should be done at runtime
            self.is_single_point = True

    @staticmethod
    def single_point(expr):
        interval = Range(expr, expr, RangeType.CLOSED_CLOSED)
        interval.is_single_point = True
        return interval

    def normalize(self):
        shift = ConstExpr(0)
        if not self.start.same_as(ConstExpr(0)) and not self.is_single_point:
            shift = self.start
            self.end = self.end - self.start
            self.start = ConstExpr(0)
        return shift

class IterVar(Expr):
    NORMAL = 0
    SPLIT = 1
    FUSE = 2

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

    def same_as(self, other):
        return self is other or (isinstance(other, IterVar) and self.name == other.name)

    def CUDA_codegen(self):
        if self.range.is_single_point:
            return self.range.start.CUDA_codegen()
        elif self.bind_type == IterVar.BIND:
            return self.bind_name
        elif self.type == IterVar.SPLIT:
            return "(({0} * {1}) + {2})".format(self.outer.CUDA_codegen(), self.inner.range.end.CUDA_codegen(), self.inner.CUDA_codegen())
        elif self.type == IterVar.FUSE:
            if self is self.fused.outer:
                return "({0} // {1})".format(self.fused.CUDA_codegen(), self.fused.inner.range.end.CUDA_codegen())
            else:
                return "({0} % {1})".format(self.fused.CUDA_codegen(), self.fused.inner.range.end.CUDA_codegen())
        else:
            return self.name

class ReduceExpr(Expr):
    def __init__(self, combinator, init, expr, axis):
        super().__init__()
        self.combinator = wrap_number_as_const_expr(combinator)
        self.init = wrap_number_as_const_expr(init)
        self.expr = wrap_number_as_const_expr(expr)
        self.reduce_axis = axis if isinstance(axis, tuple) else (axis, )
    
    def __str__(self):
        raise NotImplementedError
    
    def same_as(self):
        raise NotImplementedError

    def CUDA_codegen(self):
        raise NotImplementedError
 
class IfThenElseExpr(Expr):
    def __init__(self, condition, then_expr, else_expr):
        super().__init__(condition, then_expr, else_expr)
        self.condition = wrap_number_as_const_expr(condition)
        self.then_expr = wrap_number_as_const_expr(then_expr)
        self.else_expr = wrap_number_as_const_expr(else_expr)

    def __str__(self):
        return "({0} ? {1} : {2})".format(str(self.condition), str(self.then_expr), str(self.else_expr))
    
    def same_as(self, other):
        return self is other or (isinstance(other, IfThenElseExpr) and self.condition.same_as(other.condition) and self.then_expr.same_as(other.then_expr) and self.else_expr.same_as(other.else_expr))

    def CUDA_codegen(self):
        return "({0} ? {1} : {2})".format(self.condition.CUDA_codegen(), self.then_expr.CUDA_codegen(), self.else_expr.CUDA_codegen())

class TensorSliceExpr(Expr):
    def __init__(self, tensor, index):
        # TODO: fix init
        super().__init__()
        self.tensor = tensor
        self.index = index

    def __getitem__(self, index):
        raise NotImplementedError

    def __setitem__(self, index, item):
        raise NotImplementedError
    
    def __str__(self):
        return self.tensor.name + "[" + ", ".join([str(index) for index in self.index]) + "]" 

    def same_as(self, other):
        if self is other:
            return True
        res = isinstance(other, TensorSliceExpr) and self.tensor.same_as(other.tensor)
        if res:
            idx_res = [self.index[i].same_as(other.index[i]) for i in range(len(self.index))]
            for i in idx_res:
                res = res and i
        return res

    def CUDA_codegen(self):
        return self.tensor.name + "[" + ", ".join([index.CUDA_codegen() for index in self.index]) + "]" 


def collect_inputs(expr):
    inputs = set()
    # TODO: fix same consumer add here
    consumers = {}
    q = [expr]
    while len(q) > 0:
        expr = q.pop()
        if isinstance(expr, TensorSliceExpr):
            if expr.tensor in consumers:
                consumers[expr.tensor].append(expr)
            else:
                consumers[expr.tensor] = [expr]
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
    return list(inputs), consumers


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
        self.attach_at = None
        # is_safe == True means that no boundary test is needed
        self.is_safe = True
        self.inputs = []
        self.outputs = []
        self.consumers = {}

        # leaf axis
        self.axis = [IterVar(self.name + "_" + compute_func.__code__.co_varnames[i] if compute_func is not None else 'i' + str(i), 0, v) for i, v in enumerate(self.shape)]
        self.root_axis = tuple(self.axis)
        self.reduce_axis = ()
        # tensor's attach_path, only used when compute_at
        # for example: A.compute_at(B, B.axis[1]), then A.attach_path = (B.axis[1], B.axis[0])
        self.attach_path = ()

        if tensor_type == TensorExpr.COMPUTE:
            self.expr = wrap_number_as_const_expr(compute_func(*self.axis))
            self.inputs, self.consumers = collect_inputs(self.expr)
            
            for inp in self.inputs:
                inp.outputs.append(self)

            if isinstance(self.expr, ReduceExpr):
                self.reduce_axis = self.expr.reduce_axis
                self.axis = list(self.root_axis + self.expr.reduce_axis)
                self.expr = self.expr.combinator(TensorSliceExpr(self, self.root_axis), self.expr.expr)
    
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index, )
        if len(index) != len(self.axis):
            raise ValueError("should provide exactly {0} axis, got {1}.".format(len(self.axis), len(index)))
        index = tuple([wrap_number_as_const_expr(idx) for idx in index])
        tensor_slice = TensorSliceExpr(self, index)
        return tensor_slice

    def __setitem__(self, index, item):
        raise NotImplementedError
    
    def __str__(self):
        return self.name

    def same_as(other):
        return self is other or (isinstance(other, TensorExpr) and self.name == other.name)

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
        
        # TODO: should add boundary check if can not prove no out of index happens
        body = "    " * scope + TensorSliceExpr(self, self.root_axis).CUDA_codegen() + " = " + self.expr.CUDA_codegen() + ";\n"
        return opening + body + closing
