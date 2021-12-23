import KgeN
from KgeN import te

m = 256
n = 128
A = te.placeholder((n, n), name = "A")
B = te.compute((n, n), lambda i, j: 2 + A[i, j], name = "B")
C = te.compute((m, m), lambda i, j: B[i, j] * 2, name = "C")
# schedule
s = te.create_schedule(C.op)
outer, inner = s[C].split(s[C].op.axis[0], 32)
B_outer, B_inner = s[B].split(s[B].op.axis[0], 32)
s[B].compute_at(s[C], s[C].op.axis[1])

# lower
func = KgeN.lower(s, [A, C])
print(KgeN.build(func))