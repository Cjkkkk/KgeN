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
KgeN.reorder(C, Bx, By, Tx, Ty, Ko, Ki, Mi, Ni)
AA = KgeN.cache_read(A, "shared", [C])
BB = KgeN.cache_read(B, "shared", [C])
AAA = KgeN.cache_read(AA, "local", [C])
BBB = KgeN.cache_read(BB, "local", [C])

block_x = KgeN.thread_axis("blockIdx.x")
block_y = KgeN.thread_axis("blockIdx.y")
thread_x = KgeN.thread_axis("threadIdx.x")
thread_y = KgeN.thread_axis("threadIdx.y")

AM, AK = AA.axis
BK, BN = BB.axis

ATx, _ = KgeN.split(AA, AM, 4)
ATy, _ = KgeN.split(AA, AK, 8)
BTx, _ = KgeN.split(BB, BK, 8)
BTy, _ = KgeN.split(BB, BN, 4)

KgeN.compute_at(AA, C, Ko)
KgeN.compute_at(BB, C, Ko)
KgeN.compute_at(AAA, C, Ki)
KgeN.compute_at(BBB, C, Ki)
KgeN.bind(Bx, block_x)
KgeN.bind(By, block_y)
KgeN.bind(Tx, thread_x)
KgeN.bind(Ty, thread_y)
KgeN.bind(ATx, thread_x)
KgeN.bind(ATy, thread_y)
KgeN.bind(BTx, thread_x)
KgeN.bind(BTy, thread_y)
func = KgeN.lower([A, B, C])
print(KgeN.build(func))