import KgeN

M = 128

A = KgeN.placeholder((M, M), name= "A")
B = KgeN.compute((M - 2, M - 2), lambda i, j: ( A[i - 1, j] + A[i, j] + A[i + 1, j] ) / 3, name="B")

print(KgeN.lower(B))