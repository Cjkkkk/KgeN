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
    AND = 13
    OR = 14
    NOT = 15
    NEG = 16
    
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
        elif isinstance(self, ConstExpr) and self.val == 1:
            return other
        elif isinstance(other, ConstExpr) and other.val == 0:
            return ConstExpr(0)
        elif isinstance(other, ConstExpr) and other.val == 1:
            return self
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
        elif isinstance(other, ConstExpr) and other.val == 1:
            return self
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

    @staticmethod
    def and_(a, b):
        a = wrap_number_as_const_expr(a)
        b = wrap_number_as_const_expr(b)
        if isinstance(a, ConstExpr) and isinstance(b, ConstExpr):
            return ConstExpr(a.val and b.val)
        else:
            return BinaryExpr(a, b, Expr.AND)

    @staticmethod
    def or_(a, b):
        a = wrap_number_as_const_expr(a)
        b = wrap_number_as_const_expr(b)
        if isinstance(a, ConstExpr) and isinstance(b, ConstExpr):
            return ConstExpr(a.val or b.val)
        else:
            return BinaryExpr(a, b, Expr.OR)
    
    @staticmethod
    def not_(a):
        a = wrap_number_as_const_expr(a)
        if isinstance(a, ConstExpr):
            return ConstExpr(not a.val)
        else:
            return UnaryExpr(a, Expr.NOT)

    def __str__(self):
        from .ir_printer import IR_Printer
        ir_printer = IR_Printer()
        return self.accept(ir_printer)

    def same_as(self, other):
        raise NotImplementedError
    
    def accept(self, visitor):
        raise NotImplementedError

Expr.function_mapping = [Expr.__add__, Expr.__mul__, Expr.__truediv__, Expr.__floordiv__, Expr.__sub__, Expr.__mod__, Expr.__gt__,
        Expr.__ge__, Expr.__lt__, Expr.__le__, Expr.min, Expr.max, Expr.ceildiv, Expr.and_, Expr.or_, Expr.not_, Expr.__neg__]


class UnaryExpr(Expr):
    def __init__(self, expr, type):
        super().__init__()
        self.expr = expr
        self.type = type

    def same_as(self, other):
        return isinstance(other, UnaryExpr) and self.type == other.type and self.expr.same_as(other.expr)

    def accept(self, visitor):
        return visitor.visit_unary_expr(self)

class BinaryExpr(Expr):
    def __init__(self, left, right, type):
        super().__init__()
        self.left = left
        self.right = right
        self.type = type

    def same_as(self, other):
        return self is other or (isinstance(other, BinaryExpr) and self.type == other.type and self.left.same_as(other.left) and self.right.same_as(other.right))

    def is_compare(self):
        return self.type == Expr.GE or self.type == Expr.GT or self.type == Expr.LE or self.type == Expr.LT
    
    def accept(self, visitor):
        return visitor.visit_binary_expr(self)


class VarExpr(Expr):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def same_as(self, other):
        return self is other or (isinstance(other, VarExpr) and self.name == other.name)

    def accept(self, visitor):
        return visitor.visit_var_expr(self)


class ConstExpr(Expr):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def same_as(self, other):
        return self is other or (isinstance(other, ConstExpr) and self.val == other.val)

    def accept(self, visitor):
        return visitor.visit_const_expr(self)

class Range:
    def __init__(self, start, end):
        self.start = wrap_number_as_const_expr(start)
        self.end = wrap_number_as_const_expr(end)
    
    @staticmethod
    def single_point(expr):
        range = Range(expr, expr + 1)
        return range

    @property
    def is_single_point(self):
        return (self.start + 1).same_as(self.end)

    def convert_to_interval(self):
        from .interval import Interval
        return Interval(self.start, self.end - 1)
    
    def __str__(self):
        return "[{0}, {1})".format(self.start, self.end)

class IterVar(Expr):
    # relation
    NORMAL = 0
    SPLIT = 1
    FUSE = 2

    # type
    DEFAULT = 3
    BIND = 4
    VECTORIZED=5
    UNROLL=6
    REDUCE=7
    def __init__(self, name, start=0, end=0):
        super().__init__()
        self.name = name
        self.range = Range(start, end)
        self.attached_computation = []
        self.relation = IterVar.NORMAL
        self.type = IterVar.DEFAULT
        self.bind_to = None

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
        axis_tuple = axis if isinstance(axis, tuple) else (axis, )
        
        for axis in axis_tuple:
            assert axis.type == IterVar.REDUCE, "axis {0} must be reduce axis.".format(axis.name)
        self.reduce_axis = axis_tuple
    
    def same_as(self):
        raise NotImplementedError
 
    def accept(self, visitor):
        return visitor.visit_reduce_expr(self)


class IfThenElseExpr(Expr):
    def __init__(self, condition, then_expr, else_expr):
        super().__init__()
        self.condition = wrap_number_as_const_expr(condition)
        self.then_expr = wrap_number_as_const_expr(then_expr)
        self.else_expr = wrap_number_as_const_expr(else_expr)
    
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

    def same_as(self, other):
        if self is other:
            return True
        res = isinstance(other, TensorSliceExpr) and self.tensor.same_as(other.tensor) and len(self.index) == len(other.index)
        for i in range(len(self.index)):
            if res:
                res = res and self.index[i].same_as(other.index[i])
        return res

    def accept(self, visitor):
        return visitor.visit_tensor_slice_expr(self)

class TensorExpr(Expr):
    PLACEHOLDER = 0
    COMPUTE = 1
    def __init__(self, shape, name, tensor_type, compute_func=None, dtype="float", scope="global"):
        super().__init__()
        self.shape = tuple([ConstExpr(s) if isinstance(s, int) else s for s in shape])
        self.name = name
        self.type = tensor_type
        self.compute_func = compute_func
        self.dtype = dtype
        self.scope = scope

        # is_safe == True means that no boundary test is needed
        self.is_safe = True

        # tensor's inputs and outputs
        self.inputs = []
        self.outputs = []
        self.providers = {}

        # leaf axis
        self.axis = tuple([IterVar(self.name + "_" + compute_func.__code__.co_varnames[i] if compute_func is not None else 'i' + str(i), 0, v) for i, v in enumerate(self.shape)])
        self.reduce_axis = ()
        if tensor_type == TensorExpr.COMPUTE:
            self.expr = wrap_number_as_const_expr(compute_func(*self.axis))

            if isinstance(self.expr, ReduceExpr):
                self.reduce_axis = self.expr.reduce_axis
            
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index, )
        assert len(index) == len(self.axis), "should provide exactly {0} axis, got {1}.".format(len(self.axis), len(index))
        index = tuple([wrap_number_as_const_expr(idx) for idx in index])
        tensor_slice = TensorSliceExpr(self, index)
        return tensor_slice

    def is_input(self):
        return self.type == TensorExpr.PLACEHOLDER
    
    def is_output(self):
        return len(self.outputs) == 0

    def same_as(self, other):
        return self is other or (isinstance(other, TensorExpr) and self.name == other.name)

    def accept(self, visitor):
        return visitor.visit_tensor_expr(self)


class Stmt:
    def __init__(self):
        pass
    
    def __str__(self):
        from .ir_printer import IR_Printer
        ir_printer = IR_Printer()
        self.accept(ir_printer)
        return ir_printer.res

    def accept(self, visitor):
        raise NotImplementedError

class FuncStmt(Stmt):
    def __init__(self):
        super().__init__()
        self.body = []
        self.schedule = None
        self.input_tensors = []
        self.output_tensors = []

    def accept(self, visitor):
        return visitor.visit_func_stmt(self)

class IfStmt(Stmt):
    def __init__(self, condition, then_stmt, else_stmt=None):
        super().__init__()
        self.condition = condition
        self.then_stmt = then_stmt
        self.else_stmt = else_stmt

    def accept(self, visitor):
        return visitor.visit_if_stmt(self)
    
class ForStmt(Stmt):
    def __init__(self, iter_var):
        super().__init__()
        self.iter_var = iter_var
        self.body = []
        self.need_sync_before = False
        self.need_sync_after = False

    def accept(self, visitor):
        return visitor.visit_for_stmt(self)

class AssignStmt(Stmt):
    def __init__(self, dest, source):
        super().__init__()
        self.dest = dest
        self.source = source
    
    def accept(self, visitor):
        return visitor.visit_assign_stmt(self)
