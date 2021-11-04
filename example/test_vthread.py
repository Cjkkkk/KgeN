import KgeN
from KgeN import te

# 1. not vthread
# M = 128
# A = te.placeholder((M, ), name= "A")
# B = te.compute((M, ), lambda i: A[i], name="B")
# C = te.compute((M, ), lambda i: B[i], name="C")
# s = te.create_schedule(C)
# x, = C.axis
# xo, xi = s[C].split(x, factor=4)
# s[C].reorder(xi, xo)
# s[B].compute_at(s[C], xi)
# tir = str(KgeN.lower(s, [A, C]))
# print(tir)

# 2. vthread
M = 1024
A = te.placeholder((M, ), name= "A")
B = te.compute((M, ), lambda i: A[i], name="B")
C = te.compute((M, ), lambda i: B[i], name="C")
s = te.create_schedule(C)
x, = C.axis
xo, xi = s[C].split(x, factor=64)
xio, xii = s[C].split(xi, factor=4)
s[C].bind(xo, te.thread_axis("vthread", name="vx"))
s[C].bind(xio, te.thread_axis("vthread", name="vy"))
s[B].compute_at(s[C], xio)
tir = str(KgeN.lower(s, [A, C]))
print(tir)