import KgeN

M = 64
N = 64
K = 64

A = KgeN.placeholder((M, K), name= "A")
B = KgeN.placeholder((K, N), name= "B")
k = KgeN.reduce_axis(K, name="k")
C = KgeN.compute((M, N), 
    lambda i, j: KgeN.reduce_sum(A[i, k] * B[k, j], axis=k), 
    name="C")

M, N, K = C.axis
Mo, Mi = KgeN.split(C, M, 4)
No, Ni = KgeN.split(C, N, 4)
Bx, Tx = KgeN.split(C, Mo, 4)
By, Ty = KgeN.split(C, No, 4)
Ko, Ki = KgeN.split(C, K, 32)
KgeN.reorder(C, (Bx, By, Tx, Ty, Ko, Ki, Mi, Ni))
AA = KgeN.cache_read(A, "shared", [C])
BB = KgeN.cache_read(B, "shared", [C])
KgeN.compute_at(AA, C, By)
KgeN.compute_at(BB, C, By)
print(KgeN.lower(C))