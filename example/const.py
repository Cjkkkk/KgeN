import KgeN

M = 128
A = KgeN.compute((M, M), lambda i, j: 1, name="A")

func = KgeN.lower([A])
print(KgeN.build(func))