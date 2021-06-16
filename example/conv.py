import KgeN

# https://tvm.apache.org/docs/tutorials/optimize/opt_conv_cuda.html

batch = 256
in_channel = 256
out_channel = 512
in_size = 14
kernel = 3
pad = 1
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

KgeN.compute_inline(Apad)
AA = KgeN.cache_read(Apad, "shared", [B])
WW = KgeN.cache_read(W, "shared", [B])
AL = KgeN.cache_read(AA, "local", [B])
WL = KgeN.cache_read(WW, "local", [B])
BL = KgeN.cache_write(B, "local")

tile = 8
num_thread = 8
block_factor = 64
step = 8

hi, wi, fi, ni = B.axis
bz = KgeN.fuse(B, hi, wi)
by, fi = KgeN.split(B, fi, factor=block_factor)
bx, ni = KgeN.split(B, ni, factor=block_factor)
ty, fi = KgeN.split(B, fi, nparts=num_thread)
tx, ni = KgeN.split(B, ni, nparts=num_thread)
KgeN.reorder(B, bz, by, bx, ty, tx, fi, ni)


# Bind the iteration variables to GPU thread indices
KgeN.bind(bz, "blockIdx.z")
KgeN.bind(by, "blockIdx.y")
KgeN.bind(bx, "blockIdx.x")
KgeN.bind(ty, "threadIdx.y")
KgeN.bind(tx, "threadIdx.x")


# Schedule BL local write
KgeN.compute_at(BL, B, tx)
yi, xi, fi, ni, ry, rx, rc = BL.axis
rco, rci = KgeN.split(BL, rc, factor=step)
KgeN.reorder(BL, rco, ry, rx, rci, fi, ni)

# Attach computation to iteration variables
KgeN.compute_at(AA, BL, rx)
KgeN.compute_at(WW, BL, rx)
KgeN.compute_at(AL, BL, rci)
KgeN.compute_at(WL, BL, rci)

# Schedule for A's shared memory load
yi, xi, ci, ni = AA.axis
ty, ci = KgeN.split(AA, ci, nparts=num_thread)
tx, ni = KgeN.split(AA, ni, nparts=num_thread)
KgeN.reorder(AA, ty, tx, yi, xi, ci, ni)
KgeN.bind(ty, "threadIdx.y")
KgeN.bind(tx, "threadIdx.x")

# Schedule for W's shared memory load
yi, xi, ci, fi = WW.axis
ty, ci = KgeN.split(WW, ci, nparts=num_thread)
tx, fi = KgeN.split(WW, fi, nparts=num_thread)
KgeN.reorder(WW, ty, tx, yi, xi, ci, fi)
KgeN.bind(ty, "threadIdx.y")
KgeN.bind(tx, "threadIdx.x")

func = KgeN.lower([A, W, B])
print(str(func))
# print(KgeN.build(func))