import KgeN
from KgeN import te

M = 128

A = te.placeholder((M, M), name= "A")
B = te.compute((M - 2, M - 2), lambda i, j: ( A[i - 1, j] + A[i, j] + A[i + 1, j] ) / 3, name="B")

s = te.create_schedule(B.op)
func = KgeN.lower(s, [A, B])
print(KgeN.build(func))