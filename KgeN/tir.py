# Expr IR
def wrap_number_as_const_expr(v):
    if isinstance(v, int) or isinstance(v, float):
        return ConstExpr(v)
    else:
        return v

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

    def same_as(self, other):
        raise NotImplementedError

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

    def same_as(self, other):
        return isinstance(other, BinaryExpr) and self.type == other.type and self.left.same_as(other.left) and self.right.same_as(other.right)

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
        return self.name == other.name
    
    def CUDA_codegen(self):
        return self.name

class ConstExpr(Expr):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def __str__(self):
        return str(self.val)

    def same_as(self, other):
        return isinstance(other, ConstExpr) and self.val == other.val

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
        return isinstance(other, IterVar) and self.name == other.name

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
        return isinstance(other, IfThenElseExpr) and self.condition.same_as(other.condition) and self.then_expr.same_as(other.then_expr) and self.else_expr.same_as(other.else_expr)

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

    def same_as(self):
        res = isinstance(other, TensorSliceExpr) and self.tensor.same_as(other.tensor)
        if res:
            idx_res = [self.index[i].same_as(other.index[i]) for i in range(len(self.index))]
            for i in idx_res:
                res = res and i
        return res

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
        self.attach_at = None
        self.inputs = []
        self.consumers = []
        self.axis = ()
        self.root_axis = ()
        self.fixed_axis = ()
        self.reduce_axis = ()

        if tensor_type == TensorExpr.COMPUTE:
            self.axis = tuple([IterVar(self.name + "_" + compute_func.__code__.co_varnames[i], 0, v) for i, v in enumerate(self.shape)])
            self.root_axis = self.axis
            self.expr = wrap_number_as_const_expr(compute_func(*self.axis))
            self.inputs = collect_inputs(self.expr)

            if isinstance(self.expr, ReduceExpr):
                self.reduce_axis = self.expr.reduce_axis
                self.axis = tuple(self.axis + self.expr.reduce_axis)
                self.expr = self.expr.combinator(TensorSliceExpr(self, self.root_axis), self.expr.expr)
    
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index, )
        index = tuple([wrap_number_as_const_expr(idx) for idx in index])
        tensor_slice = TensorSliceExpr(self, index)
        self.consumers.append(tensor_slice)
        return tensor_slice

    def __setitem__(self, index, item):
        raise NotImplementedError
    
    def __str__(self):
        return self.name

    def same_as(other):
        return isinstance(other, TensorExpr) and self.name == other.name

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
