import KgeN
from KgeN import te

M = 64
N = 64
K = 64

A = te.placeholder((M, K), name= "A")
B = te.placeholder((K, N), name= "B")
k = te.reduce_axis(K, name="k")
C = te.compute((M, N), 
    lambda i, j: te.reduce_sum(A[i, k] * B[k, j], axis=k), 
    name="C")

s = te.create_schedule(C.op)

AA = s.cache_read(A, "shared", [C])
BB = s.cache_read(B, "shared", [C])
AAA = s.cache_read(AA, "local", [C])
BBB = s.cache_read(BB, "local", [C])

M, N = s[C].op.axis
K, = C.reduce_axis
Mo, Mi = s[C].split(M, 4)
No, Ni = s[C].split(N, 4)
Bx, Tx = s[C].split(Mo, 4)
By, Ty = s[C].split(No, 4)
Ko, Ki = s[C].split(K, 32)
s[C].reorder(Bx, By, Tx, Ty, Ko, Ki, Mi, Ni)

block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")

AM, AK = s[AA].op.axis
BK, BN = s[BB].op.axis

ATx, _ = s[AA].split(AM, 4)
ATy, _ = s[AA].split(AK, 8)
BTx, _ = s[BB].split(BK, 8)
BTy, _ = s[BB].split(BN, 4)

s[AA].compute_at(s[C], Ko)
s[BB].compute_at(s[C], Ko)
s[AAA].compute_at(s[C], Ki)
s[BBB].compute_at(s[C], Ki)
s[C].bind(Bx, block_x)
s[C].bind(By, block_y)
s[C].bind(Tx, thread_x)
s[C].bind(Ty, thread_y)
s[C].bind(ATx, thread_x)
s[C].bind(ATy, thread_y)
s[C].bind(BTx, thread_x)
s[C].bind(BTy, thread_y)
func = KgeN.lower([A, B, C])
print(KgeN.build(func))