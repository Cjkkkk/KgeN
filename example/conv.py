import KgeN

# https://tvm.apache.org/docs/tutorials/optimize/opt_conv_cuda.html

batch = 256
in_channel = 256
out_channel = 512
in_size = 14
kernel = 3
pad = 0
stride = 1


A = KgeN.placeholder((in_size, in_size, in_channel, batch), name="A")
W = KgeN.placeholder((kernel, kernel, in_channel, out_channel), name="W")
out_size = (in_size - kernel + 2 * pad) // stride + 1
# Pad input
Apad = KgeN.compute(
    (in_size + 2 * pad, in_size + 2 * pad, in_channel, batch),
    lambda yy, xx, cc, nn: KgeN.if_then_else(
        KgeN.all(yy >= pad, yy - pad < in_size, xx >= pad, xx - pad < in_size),
        A[yy - pad, xx - pad, cc, nn],
        0,
    ),
    name="Apad",
)
# Create reduction variables
rc = KgeN.reduce_axis(in_channel, name="rc")
ry = KgeN.reduce_axis(kernel, name="ry")
rx = KgeN.reduce_axis(kernel, name="rx")
# Compute the convolution
B = KgeN.compute(
    (out_size, out_size, out_channel, batch),
    lambda yy, xx, ff, nn: KgeN.reduce_sum(
        Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff], axis=(ry, rx, rc)
    ),
    name="B",
)


AA = KgeN.cache_read(Apad, "shared", [B])
WW = KgeN.cache_read(W, "shared", [B])
AL = KgeN.cache_read(AA, "local", [B])
WL = KgeN.cache_read(WW, "local", [B])
BL = KgeN.cache_write(B, "local")

tile = 8
num_thread = 8
block_factor = tile * num_thread
step = 8
vthread = 2

hi, wi, fi, ni = B.axis
bz = KgeN.fuse(B, hi, wi)
by, fi = KgeN.split(B, fi, factor=block_factor)
bx, ni = KgeN.split(B, ni, factor=block_factor)

# Bind the iteration variables to GPU thread indices
KgeN.bind(bz, "blockIdx.z")
KgeN.bind(by, "blockIdx.y")
KgeN.bind(bx, "blockIdx.x")

ty, fi = KgeN.split(B, fi, nparts=num_thread)
tx, ni = KgeN.split(B, ni, nparts=num_thread)

func = KgeN.lower(B)
print(str(func))
# print(KgeN.build(func))