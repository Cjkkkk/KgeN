import KgeN
from KgeN import te

def test_bound():
    A = te.placeholder((7, 7), name="A")
    B = te.compute((7, 7), lambda i, j: A[i, j], name="B")
    C = te.compute((3, 3), lambda i, j: B[i+1, j*2], name="C")

    s = te.create_schedule(C.op)
    KgeN.lower(s, [A, C])
    assert s[B].op.axis[0].range.end.val == 3
    assert s[B].op.axis[1].range.end.val == 5

def test_split():
    A = te.placeholder((8, 8), name="A")
    B = te.compute((8, 8), lambda i, j: A[i, j], name="B")

    s = te.create_schedule(B.op)
    x, _ = s[B].op.axis
    x_o, x_i = s[B].split(x, 2)
    KgeN.lower(s, [A, B])
    assert x_o.range.end.val == 4
    assert x_i.range.end.val == 2

def test_split1():
    A = te.placeholder((9, 9), name="A")
    B = te.compute((9, 9), lambda i, j: A[i, j], name="B")

    s = te.create_schedule(B.op)
    x, _ = s[B].op.axis
    x_o, x_i = s[B].split(x, 2)
    KgeN.lower(s, [A, B])
    assert x_o.range.end.val == 5
    assert x_i.range.end.val == 2

def test_fuse():
    A = te.placeholder((8, 8), name="A")
    B = te.compute((8, 8), lambda i, j: A[i, j], name="B")

    s = te.create_schedule(B.op)
    x, y = s[B].op.axis
    fused = s[B].fuse(x, y)
    KgeN.lower(s, [A, B])
    assert fused.range.end.val == 64

def test_if_then_else():
    A = te.placeholder((8, 8), name="A")
    B = te.compute((8, 8), lambda i, j: A[i, j], name="B")
    C = te.compute((3, 3), lambda i, j: te.if_then_else(te.all(i < 2, j < 1), B[i, j] + 1, 0), name="C")

    s = te.create_schedule(C.op)
    KgeN.lower(s, [A, C])
    x, y = s[B].op.axis
    assert x.range.end.val == 2
    assert y.range.end.val == 1

def test_if_then_else1():
    A = te.placeholder((8, 8), name="A")
    B = te.compute((8, 8), lambda i, j: A[i, j], name="B")
    C = te.compute((5, 5), lambda i, j: te.if_then_else(
        te.all(i < 4, j < 4), 
        te.if_then_else(
            te.all(i > 2, j > 1), 
            B[i, j] + 1, 
            0), 
        0), name="C")

    s = te.create_schedule(C.op)
    KgeN.lower(s, [A, C])
    x, y = s[B].op.axis
    assert x.range.end.val == 1
    assert y.range.end.val == 2