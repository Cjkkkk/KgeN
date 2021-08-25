import KgeN
from KgeN import te

m = 8
n = 64
A = te.placeholder((m, n), name = "A")
B = te.compute((m, n), lambda i, j: 2 + A[i, j], name = "B")
# schedule
s = te.create_schedule(B)
ax = te.thread_axis(8, "threadIdx.x")
s[B].bind(B.axis[0], ax)

# lower
func = KgeN.lower(s, [A, B])
print(str(func))