import sys
sys.path.append("..")

import KgeN

m = 256
A = KgeN.placeholder((m, m), name = "A")
B = KgeN.compute((m, m), lambda i, j: 2 + A[i, j], name = "B")
k = KgeN.reduce_axis(128, name="k")
C = KgeN.compute((m, ), lambda i: KgeN.reduce_sum(A[i, k] * B[i, k], axis=k), name = "C")
# schedule
outer, inner = KgeN.split(C, C.axis[0], 32)
# KgeN.reorder(C, (inner, outer))
# fused = KgeN.fuse(C, (outer, inner))
KgeN.compute_at(B, C, outer)

# lower
print(KgeN.lower(C))