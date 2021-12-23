import KgeN
from KgeN import te

M = 128
A = te.compute((M, M), lambda i, j: 1, name="A")

s = te.create_schedule(A.op)
func = KgeN.lower(s, [A])
print(KgeN.build(func))