import KgeN
from KgeN import te

M = 128
N = 128
K = 128

A = te.placeholder((M, K), name= "A")
B = te.placeholder((K, N), name= "B")
k = te.reduce_axis(K, name="k")
C = te.compute((M, N), 
    lambda i, j: te.reduce_sum(A[i, k] * B[k, j], axis=k), 
    name="C")

s = te.create_schedule(C.op)
M, N = s[C].op.axis
K, = C.reduce_axis
s[C].reorder(K, N)
func = KgeN.lower(s, [A, B, C])
print(KgeN.build(func))