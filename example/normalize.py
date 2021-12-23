import KgeN
from KgeN import te

M = 128

A = te.placeholder((M, M), name= "A")
B = te.compute((M, M - 2), lambda i, j: ( A[i, j - 1] + A[i, j] + A[i, j + 1] ) / 3, name="B")
C = te.compute((M - 2, M - 2), lambda i, j: ( B[i - 1, j] + B[i, j] + B[i + 1, j] ) / 3, name="C")

s = te.create_schedule(C.op)
func = KgeN.lower(s, [A, C])
print(KgeN.build(func))