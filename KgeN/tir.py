import math
from .visitor import Visitor

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
        elif isinstance(other, UnaryExpr) and other.type == Expr.NEG:
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
        elif isinstance(other, UnaryExpr) and other.type == Expr.NEG:
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
    
    def accept(self, visitor):
        raise NotImplementedError

Expr.function_mapping = [Expr.__add__, Expr.__mul__, Expr.__truediv__, Expr.__floordiv__, Expr.__sub__, Expr.__mod__, Expr.__gt__,
        Expr.__ge__, Expr.__lt__, Expr.__le__, Expr.min, Expr.max, Expr.ceildiv, Expr.__neg__]

Expr.is_commutative = [True, True, False, False, False, False, False,
        False, False, False, True, True, False, False]

class UnaryExpr(Expr):
    def __init__(self, expr, type_):
        super().__init__(expr)
        self.expr = expr
        self.type = type_

    def __str__(self):
        return "({0}({1}))".format(Expr.mapping[self.type], self.expr)

    def same_as(self, other):
        return isinstance(other, UnaryExpr) and self.type == other.type and self.expr.same_as(other.expr)

    def accept(self, visitor):
        return visitor.visit_unary_expr(self)

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

    def accept(self, visitor):
        return visitor.visit_binary_expr(self)


class VarExpr(Expr):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def __str__(self):
        return self.name

    def same_as(self, other):
        return self is other or (isinstance(other, VarExpr) and self.name == other.name)

    def accept(self, visitor):
        return visitor.visit_var_expr(self)


class ConstExpr(Expr):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def __str__(self):
        return str(self.val)

    def same_as(self, other):
        return self is other or (isinstance(other, ConstExpr) and self.val == other.val)

    def accept(self, visitor):
        return visitor.visit_const_expr(self)


class Range:
    CLOSED_OPEN = 0
    CLOSED_CLOSED = 1
    def __init__(self, start, end, type_= CLOSED_OPEN):
        self.start = wrap_number_as_const_expr(start)
        self.end = wrap_number_as_const_expr(end)
        self.is_single_point = False
        self.type = type_

        if self.type == Range.CLOSED_CLOSED and self.start.same_as(self.end):
            # TODO: fix this, should be done at runtime
            self.is_single_point = True

    @staticmethod
    def single_point(expr):
        interval = Range(expr, expr, Range.CLOSED_CLOSED)
        interval.is_single_point = True
        return interval

    def as_closed_open(self):
        if self.type == Range.CLOSED_CLOSED:
            self.type = Range.CLOSED_OPEN
            self.end += 1
    
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
    def __init__(self, name, start=0, end=0, type_= Range.CLOSED_OPEN):
        super().__init__()
        self.name = name
        self.range = Range(start, end, type_)
        self.attached_computation = []
        self.type = IterVar.NORMAL
        self.bind_type = IterVar.NONE
        self.bind_name = ""

    def __str__(self):
        return "{0}: [{1}, {2} {3}".format(self.name, self.range.start, self.range.end, "]" if self.range.type == Range.CLOSED_CLOSED else ")")

    def same_as(self, other):
        return self is other or (isinstance(other, IterVar) and self.name == other.name)

    def accept(self, visitor):
        return visitor.visit_iter_expr(self)

    
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
 
    def accept(self, visitor):
        return visitor.visit_reduce_expr(self)


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

    def accept(self, visitor):
        return visitor.visit_if_then_else_expr(self)
    

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

    def accept(self, visitor):
        return visitor.visit_tensor_slice_expr(self)

class CollectInputVisitor(Visitor):
    def __init__(self):
        self.inputs = set()
        self.providers = {}

    def collect(self, expr):
        expr.accept(self)
        return list(self.inputs), self.providers
            
    def visit_binary_expr(self, expr):
        for subexpr in expr.subexprs:
            subexpr.accept(self)

    def visit_if_then_else_expr(self, expr):
        expr.condition.accept(self)
        expr.then_expr.accept(self)
        expr.else_expr.accept(self)
    
    def visit_reduce_expr(self, expr):
        expr.expr.accept(self)

    def visit_tensor_slice_expr(self, expr):
        if expr.tensor in self.providers:
            self.providers[expr.tensor].append(expr)
        else:
            self.providers[expr.tensor] = [expr]
        self.inputs.add(expr.tensor)
        for index in expr.index:
            index.accept(self)
    
    def visit_unary_expr(self, expr):
        pass

    def visit_var_expr(self, expr):
        pass

    def visit_const_expr(self, expr):
        pass

    def visit_iter_expr(self, expr):
        pass

    def visit_tensor_expr(self, expr):
        pass

class TensorExpr(Expr):
    PLACEHOLDER = 0
    COMPUTE = 1
    def __init__(self, shape, name, tensor_type, compute_func=None, dtype="float"):
        super().__init__()
        self.shape = tuple([ConstExpr(s) if isinstance(s, int) else s for s in shape])
        self.name = name
        self.type = tensor_type
        self.compute_func = compute_func
        self.dtype = dtype

        self.attached = False
        self.attach_at = None
        # is_safe == True means that no boundary test is needed
        self.is_safe = True

        # tensor's inputs and outputs
        self.inputs = []
        self.outputs = []
        self.providers = {}

        # leaf axis
        self.axis = [IterVar(self.name + "_" + compute_func.__code__.co_varnames[i] if compute_func is not None else 'i' + str(i), 0, v) for i, v in enumerate(self.shape)]
        self.root_axis = tuple(self.axis)
        self.reduce_axis = ()
        # tensor's attach_path, only used when compute_at
        # for example: A.compute_at(B, B.axis[1]), then A.attach_path = (B.axis[1], B.axis[0])
        self.attach_path = ()

        if tensor_type == TensorExpr.COMPUTE:
            self.expr = wrap_number_as_const_expr(compute_func(*self.axis))
            self.collect_input()

            if isinstance(self.expr, ReduceExpr):
                self.reduce_axis = self.expr.reduce_axis
                self.axis = list(self.root_axis + self.expr.reduce_axis)

    def collect_input(self):
        visitor = CollectInputVisitor()
        self.inputs, self.providers = visitor.collect(self.expr)

        for inp in self.inputs:
            inp.outputs.append(self)
            
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index, )
        if len(index) != len(self.root_axis):
            raise ValueError("should provide exactly {0} axis, got {1}.".format(len(self.root_axis), len(index)))
        index = tuple([wrap_number_as_const_expr(idx) for idx in index])
        tensor_slice = TensorSliceExpr(self, index)
        return tensor_slice
    
    def __str__(self):
        return self.name

    def is_output(self):
        return len(self.outputs) == 0

    def same_as(self, other):
        return self is other or (isinstance(other, TensorExpr) and self.name == other.name)

    def accept(self, visitor):
        return visitor.visit_tensor_expr(self)


class Stmt:
    def __init__(self):
        pass
    
    def accept(self, visitor):
        raise NotImplementedError

class FuncStmt(Stmt):
    def __init__(self):
        super().__init__()
        self.body = []
        self.tensors = []

    def accept(self, visitor):
        return visitor.visit_func_stmt(self)

class ForStmt(Stmt):
    def __init__(self, iter_var):
        super().__init__()
        self.iter_var = iter_var
        self.body = []
    
    def accept(self, visitor):
        return visitor.visit_for_stmt(self)

class AssignStmt(Stmt):
    def __init__(self, dest, source):
        super().__init__()
        self.dest = dest
        self.source = source
    
    def accept(self, visitor):
        return visitor.visit_assign_stmt(self)
