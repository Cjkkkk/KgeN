import sys
sys.path.append("..")

import main as KgeN

M = 128
A = KgeN.compute((M, M), lambda i, j: 1, name="A")

print(KgeN.lower(A))