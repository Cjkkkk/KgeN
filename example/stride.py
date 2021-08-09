import KgeN

M = 128

A = KgeN.placeholder((M, ), name= "A")
B = KgeN.compute((M, ), lambda i: A[i], name="B")
C = KgeN.compute((M, ), lambda i: B[i], name="C")

s = KgeN.create_schedule(C)
x,  = C.axis
xo, xi = s[C].split(x, 4)
s[C].reorder(xi, xo)
s[B].compute_at(s[C], xi)
func = KgeN.lower(s, [A, C])
print(KgeN.build(func))