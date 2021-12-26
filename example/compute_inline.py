import KgeN
from KgeN import te

C = te.compute((5, 16), lambda ci, cj : ci * cj, name='C')
D = te.compute((5, 16), lambda di, dj : C[di, dj]*2, name='D')

s = te.create_schedule(D.op)
# s[C].compute_at(s[D], s[D].op.axis[1])
print(s[C], s[D], id(s[D].op.axis[1]))
s[C].compute_inline()
func = KgeN.lower(s, [D])
print(KgeN.build(func))