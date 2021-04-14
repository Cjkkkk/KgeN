import sys
sys.path.append("..")

import KgeN

C = KgeN.compute((5, 16), lambda ci, cj : ci * cj, name='C')
D = KgeN.compute((5, 16), lambda di, dj : C[di, dj]*2, name='D')
E = KgeN.compute((5, 16), lambda ei, ej : D[ei, ej]*4, name='E')
# KgeN.compute_at(C, D, D.axis[1])
KgeN.compute_at(D, E, E.axis[1])
print(KgeN.lower(E))