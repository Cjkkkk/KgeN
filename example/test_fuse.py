import KgeN

m = 8
n = 64
A = KgeN.placeholder((m, n), name = "A")
B = KgeN.compute((m, n), lambda i, j: 2 + A[i, j], name = "B")
C = KgeN.compute((m, n), lambda i, j: 2 + B[i, j], name = "C")
# schedule
fused = KgeN.fuse(B, *B.axis)
KgeN.reorder(C, C.axis[1], C.axis[0])
KgeN.compute_at(B, C, C.axis[0])

# lower
func = KgeN.lower(C)
print(str(func))