import KgeN

M = 128

A = KgeN.placeholder((M, ), name= "A")
B = KgeN.compute((M, ), lambda i: A[i], name="B")
C = KgeN.compute((M, ), lambda i: B[i], name="C")

x,  = C.axis
xo, xi = KgeN.split(C, x, 4)
KgeN.reorder(C, xi, xo)
KgeN.compute_at(B, C, xi)
func = KgeN.lower([A, C])
print(KgeN.build(func))