import KgeN
from KgeN import te

m = 256
n = 128
A = te.placeholder((n, n), name = "A")
B = te.compute((n, n), lambda i, j: 2 + A[i, j], name = "B")
C = te.compute((m, m), lambda i, j: B[i, j] * 2, name = "C")
# schedule
s = te.create_schedule(C)
outer, inner = s[C].split(C.axis[0], 32)
B_outer, B_inner = s[B].split(B.axis[0], 32)
s[B].compute_at(s[C], C.axis[1])

# lower
func = KgeN.lower(s, [A, C])
print(KgeN.build(func))
print(A.is_safe, B.is_safe)