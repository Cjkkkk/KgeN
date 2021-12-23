import KgeN
from KgeN import te

M = 128

A = te.placeholder((M, M), name= "A")
B = te.compute((M, M), lambda i, j: te.if_then_else(i * j > 64, A[i, j], 0), name="B")

s = te.create_schedule(B.op)
func = KgeN.lower(s, [A, B])
print(KgeN.build(func))