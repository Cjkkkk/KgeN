import KgeN

# 1. not vthread
M = 128
A = KgeN.placeholder((M, ), name= "A")
B = KgeN.compute((M, ), lambda i: A[i], name="B")
C = KgeN.compute((M, ), lambda i: B[i], name="C")
s = KgeN.create_schedule(C)
x, = C.axis
xo, xi = s[C].split(x, factor=4)
s[C].reorder(xi, xo)
s[B].compute_at(s[C], xi)
tir = str(KgeN.lower(s, [A, B]))
print(tir)

# 2. vthread
M = 128
A = KgeN.placeholder((M, ), name= "A")
B = KgeN.compute((M, ), lambda i: A[i], name="B")
C = KgeN.compute((M, ), lambda i: B[i], name="C")
s = KgeN.create_schedule(C)
x, = C.axis
xo, xi = s[C].split(x, factor=4)
s[C].bind(xo, KgeN.thread_axis("vthread"))
s[B].compute_at(s[C], xi)
tir = str(KgeN.lower(s, [A, B]))
print(tir)