import KgeN

C = KgeN.compute((5, 16), lambda ci, cj : ci * cj, name='C')
D = KgeN.compute((5, 16), lambda di, dj : C[di, dj]*2, name='D')
E = KgeN.compute((5, 16), lambda ei, ej : D[ei, ej]*4 + D[ei + 1, ej]*4, name='E')

s = KgeN.create_schedule(E)
# s[C].compute_at(s[D], D.axis[1])
s[D].compute_at(s[E], E.axis[1])
func = KgeN.lower(s, [E])
print(KgeN.build(func))