import KgeN
from KgeN import te

C = te.compute((5, 16), lambda ci, cj : 5, name='C')
D = te.compute((5, 16), lambda di, dj : C[di, dj]*2, name='D')
E = te.compute((5, 16), lambda ei, ej : D[ei, ej]*4, name='E')

s = te.create_schedule(E.op)
s[C].compute_at(s[D], s[D].op.axis[1])
s[D].compute_at(s[E], s[E].op.axis[1])

func = KgeN.lower(s, [E])
print(KgeN.build(func))