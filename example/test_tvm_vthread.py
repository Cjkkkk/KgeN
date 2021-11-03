import tvm
from tvm import te

M = 128
A = te.placeholder((M, ), name= "A")
B = te.compute((M, ), lambda i: A[i], name="B")
C = te.compute((M, ), lambda i: B[i], name="C")


# 1. not vthread
s = te.create_schedule(C.op)
x, = s[C].op.axis
xo, xi = s[C].split(x, factor=4)
s[C].reorder(xi, xo)
s[B].compute_at(s[C], xi)
tir = str(tvm.lower(s, [A, C], simple_mode=True))
print(tir)

# 2. vthread
s = te.create_schedule(C.op)
x, = s[C].op.axis
xo, xi = s[C].split(x, factor=4)
xio, xii = s[C].split(xi, factor=2)
s[C].bind(xo, te.thread_axis("vthread", name="vx"))
s[B].compute_at(s[C], xio)
tir = str(tvm.lower(s, [A, C], simple_mode=True))
print(tir)