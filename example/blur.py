import sys
sys.path.append("..")

import main as KgeN

M = 128

A = KgeN.placeholder((M, M), name= "A")
B = KgeN.compute((M - 2, M - 2), lambda i, j: ( A[i - 1] + A[i] + A[i + 1] ) / 3, name="B")

print(KgeN.lower(B))