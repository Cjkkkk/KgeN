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
AA = te.cache_read(A, "shared", [C])
BB = te.cache_read(B, "shared", [C])

s = te.create_schedule(C)
M, N = C.axis
K, = C.reduce_axis
Mo, Mi = s[C].split(M, 16)
No, Ni = s[C].split(N, 16)
Ko, Ki = s[C].split(K, 16)
s[C].reorder(Mo, No, Ko, Mi, Ni, Ki)
s[AA].compute_at(s[C], Ko)
s[BB].compute_at(s[C], Ko)
func = KgeN.lower(s, [A, B, C])
print(KgeN.build(func))