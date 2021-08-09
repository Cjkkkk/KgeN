import KgeN

M = 128
N = 128
K = 128

A = KgeN.placeholder((M, K), name= "A")
B = KgeN.placeholder((K, N), name= "B")
k = KgeN.reduce_axis(K, name="k")
C = KgeN.compute((M, N), 
    lambda i, j: KgeN.reduce_sum(A[i, k] * B[k, j], axis=k), 
    name="C")
AA = KgeN.cache_read(A, "shared", [C])
BB = KgeN.cache_read(B, "shared", [C])

s = KgeN.create_schedule(C)
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