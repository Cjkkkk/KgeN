import KgeN

m = 8
n = 64
A = KgeN.placeholder((m, n), name = "A")
B = KgeN.compute((m, n), lambda i, j: 2 + A[i, j], name = "B")
C = KgeN.compute((m, n), lambda i, j: 2 + B[i, j], name = "C")
# schedule
s = KgeN.create_schedule(C)
fused = s[B].fuse(*B.axis)
s[C].reorder(C.axis[1], C.axis[0])
s[B].compute_at(s[C], C.axis[0])

# lower
func = KgeN.lower(s, [A, C])
print(str(func))