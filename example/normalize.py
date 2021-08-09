import KgeN

M = 128

A = KgeN.placeholder((M, M), name= "A")
B = KgeN.compute((M, M - 2), lambda i, j: ( A[i, j - 1] + A[i, j] + A[i, j + 1] ) / 3, name="B")
C = KgeN.compute((M - 2, M - 2), lambda i, j: ( B[i - 1, j] + B[i, j] + B[i + 1, j] ) / 3, name="C")

s = KgeN.create_schedule(C)
func = KgeN.lower(s, [A, C])
print(KgeN.build(func))