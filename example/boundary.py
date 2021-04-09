import sys
sys.path.append("..")

import KgeN

m = 256
n = 128
A = KgeN.placeholder((n, n), name = "A")
B = KgeN.compute((n, n), lambda i, j: 2 + A[i, j], name = "B")
C = KgeN.compute((m, m), lambda i, j: B[i, j] * 2, name = "C")
# schedule
outer, inner = KgeN.split(C, C.axis[0], 32)
B_outer, B_inner = KgeN.split(B, B.axis[0], 32)
# KgeN.reorder(C, (inner, outer))
# fused = KgeN.fuse(C, (outer, inner))
KgeN.compute_at(B, C, C.axis[1])

# lower
print(KgeN.lower(C))
print(A.is_safe, B.is_safe)