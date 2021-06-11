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
KgeN.reorder(C, K, N)
print(KgeN.lower(C))