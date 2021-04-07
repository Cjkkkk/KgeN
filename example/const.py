import sys
sys.path.append("..")

import KgeN

M = 128
A = KgeN.compute((M, M), lambda i, j: 1, name="A")

print(KgeN.lower(A))