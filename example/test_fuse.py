import KgeN
from KgeN import te

m = 8
n = 64
A = te.placeholder((m, n), name = "A")
B = te.compute((m, n), lambda i, j: 2 + A[i, j], name = "B")
C = te.compute((m, n), lambda i, j: 2 + B[i, j], name = "C")
# schedule
s = te.create_schedule(C.op)
fused = s[B].fuse(*s[B].op.axis)
s[C].reorder(s[C].op.axis[1], s[C].op.axis[0])
s[B].compute_at(s[C], s[C].op.axis[0])

# lower
func = KgeN.lower(s, [A, C])
print(str(func))