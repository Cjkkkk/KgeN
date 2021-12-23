import KgeN
from KgeN import te

M = 128

A = te.placeholder((M, ), name= "A")
B = te.compute((M, ), lambda i: A[i], name="B")
C = te.compute((M, ), lambda i: B[i], name="C")

s = te.create_schedule(C.op)
x,  = s[C].op.axis
xo, xi = s[C].split(x, 4)
s[C].reorder(xi, xo)
s[B].compute_at(s[C], xi)
func = KgeN.lower(s, [A, C])
print(KgeN.build(func))