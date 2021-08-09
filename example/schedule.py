import KgeN

m = 256
A = KgeN.placeholder((m, m), name = "A")
B = KgeN.compute((m, m), lambda i, j: 2 + A[i, j], name = "B")
C = KgeN.compute((m, m), lambda i, j: B[i + j, j] * 2, name = "C")
# schedule
s = KgeN.create_schedule(C)
outer, inner = s[C].split(C.axis[0], 32)
B_outer, B_inner = s[B].split(B.axis[0], 32)
# s[C].reorder(inner, outer)
# fused = s[C].fuse(outer, inner)
s[B].compute_at(s[C], C.axis[1])

# lower
func = KgeN.lower(s, [A, C])
print(KgeN.build(func))