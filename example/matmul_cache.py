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

M, N, K = C.axis
Mo, Mi = KgeN.split(C, M, 16)
No, Ni = KgeN.split(C, N, 16)
Ko, Ki = KgeN.split(C, K, 16)
KgeN.reorder(C, Mo, No, Ko, Mi, Ni, Ki)
AA = KgeN.cache_read(A, "shared", [C])
BB = KgeN.cache_read(B, "shared", [C])
KgeN.compute_at(AA, C, Ko)
KgeN.compute_at(BB, C, Ko)
func = KgeN.lower([A, B, C])
print(KgeN.build(func))