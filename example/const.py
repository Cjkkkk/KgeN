import KgeN

M = 128
A = KgeN.compute((M, M), lambda i, j: 1, name="A")

s = KgeN.create_schedule(A)
func = KgeN.lower(s, [A])
print(KgeN.build(func))