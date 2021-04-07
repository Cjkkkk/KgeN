import sympy
from .core import *

def convert_to_sympy_expr(expr, reverse_map):
    if isinstance(expr, IterVar):
        sym = sympy.Symbol(expr.name)
        reverse_map[sym] = expr
        return sym
    elif isinstance(expr, ConstExpr):
        return expr.val
    elif isinstance(expr, BinaryExpr):
        if expr.type == BinaryExpr.ADD:
            return convert_to_sympy_expr(expr.left, reverse_map) + convert_to_sympy_expr(expr.right, reverse_map)
        elif expr.type == BinaryExpr.SUB:
            return convert_to_sympy_expr(expr.left, reverse_map) - convert_to_sympy_expr(expr.right, reverse_map)
        elif expr.type == BinaryExpr.MUL:
            return convert_to_sympy_expr(expr.left, reverse_map) * convert_to_sympy_expr(expr.right, reverse_map)
        elif expr.type == BinaryExpr.FLOOR_DIV:
            return convert_to_sympy_expr(expr.left, reverse_map) // convert_to_sympy_expr(expr.right, reverse_map)
        else:
            pass
    else:
        pass

def convert_to_KgeN_expr(expr, reverse_map):
    if isinstance(expr, sympy.core.symbol.Symbol):
        return reverse_map[expr]
    elif isinstance(expr, sympy.core.numbers.Integer):
        return ConstExpr(expr.p)
    elif isinstance(expr, sympy.core.mul.Mul):
        left = convert_to_KgeN_expr(expr.args[0], reverse_map)
        right = convert_to_KgeN_expr(expr.args[1], reverse_map)
        return left * right
    elif isinstance(expr, sympy.core.add.Add):
        left = convert_to_KgeN_expr(expr.args[0], reverse_map)
        right = convert_to_KgeN_expr(expr.args[1], reverse_map)
        return left + right
    else:
        pass

def simplify(expr):
    reverse_map = {}
    sympy_expr = convert_to_sympy_expr(expr, reverse_map)
    sympy_expr = sympy.simplify(sympy_expr)
    expr = convert_to_KgeN_expr(sympy_expr, reverse_map)
    return expr


if __name__ == "__main__":
    expr = ConstExpr(1) + IterVar("x", 0, 0) - ConstExpr(1) 
    print(simplify(expr))

