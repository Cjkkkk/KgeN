import KgeN

M = 128

A = KgeN.placeholder((M, M), name= "A")
B = KgeN.compute((M, M), lambda i, j: KgeN.if_then_else(i * j > 64, A[i, j], 0), name="B")

func = KgeN.lower(B)
print(KgeN.build(func))