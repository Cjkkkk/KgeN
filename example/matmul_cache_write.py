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

M, N, _ = C.axis
AA = KgeN.cache_read(A, "shared", [C])
BB = KgeN.cache_read(B, "shared", [C])
AAA = KgeN.cache_read(AA, "local", [C])
BBB = KgeN.cache_read(BB, "local", [C])
CCC = KgeN.cache_write(C, "local")

Mo, Mi = KgeN.split(C, M, 4)
No, Ni = KgeN.split(C, N, 4)
Bx, Tx = KgeN.split(C, Mo, 4)
By, Ty = KgeN.split(C, No, 4)
KgeN.reorder(C, Bx, By, Tx, Ty, Mi, Ni)

AM, AK = AA.axis
BK, BN = BB.axis
ATx, _ = KgeN.split(AA, AM, 4)
ATy, _ = KgeN.split(AA, AK, 8)
BTx, _ = KgeN.split(BB, BK, 8)
BTy, _ = KgeN.split(BB, BN, 4)

M, N, K = CCC.axis
Ko, Ki = KgeN.split(CCC, K, 32)
KgeN.reorder(CCC, Ko, Ki, M, N)
KgeN.unroll(CCC, M)
KgeN.unroll(CCC, N)
KgeN.compute_at(AA, CCC, Ko)
KgeN.compute_at(BB, CCC, Ko)
KgeN.compute_at(AAA, CCC, Ki)
KgeN.compute_at(BBB, CCC, Ki)
KgeN.compute_at(CCC, C, Ty)
KgeN.bind(Bx, "blockIdx.x")
KgeN.bind(By, "blockIdx.y")
KgeN.bind(Tx, "threadIdx.x")
KgeN.bind(Ty, "threadIdx.y")
KgeN.bind(ATx, "threadIdx.x")
KgeN.bind(ATy, "threadIdx.y")
KgeN.bind(BTx, "threadIdx.x")
KgeN.bind(BTy, "threadIdx.y")

func = KgeN.lower([A, B, C])
print(str(func))
print(KgeN.build(func))