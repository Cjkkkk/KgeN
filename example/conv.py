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

AA = KgeN.cache_read(Apad, "shared", [B])
WW = KgeN.cache_read(W, "shared", [B])
AL = KgeN.cache_read(AA, "local", [B])
WL = KgeN.cache_read(WW, "local", [B])
BL = KgeN.cache_write(B, "local")

# schedule
s = KgeN.create_schedule(B)
s[Apad].compute_inline()
tile = 8
num_thread = 8
block_factor = 64
step = 8

# Get the GPU thread indices
block_x = KgeN.thread_axis("blockIdx.x")
block_y = KgeN.thread_axis("blockIdx.y")
block_z = KgeN.thread_axis("blockIdx.z")
thread_x = KgeN.thread_axis(num_thread, "threadIdx.x")
thread_y = KgeN.thread_axis(num_thread, "threadIdx.y")

hi, wi, fi, ni = B.axis
bz = s[B].fuse(hi, wi)
by, fi = s[B].split(fi, factor=block_factor)
bx, ni = s[B].split(ni, factor=block_factor)
ty, fi = s[B].split(fi, nparts=num_thread)
tx, ni = s[B].split(ni, nparts=num_thread)
s[B].reorder(bz, by, bx, ty, tx, fi, ni)


# Bind the iteration variables to GPU thread indices
s[B].bind(bz, block_z)
s[B].bind(by, block_y)
s[B].bind(bx, block_x)
s[B].bind(ty, thread_y)
s[B].bind(tx, thread_x)


# Schedule BL local write
s[BL].compute_at(s[B], tx)
yi, xi, fi, ni = BL.axis
ry, rx, rc = BL.reduce_axis
rco, rci = s[BL].split(rc, factor=step)
s[BL].reorder(rco, ry, rx, rci, fi, ni)

# Attach computation to iteration variables
s[AA].compute_at(s[BL], rx)
s[WW].compute_at(s[BL], rx)
s[AL].compute_at(s[BL], rci)
s[WL].compute_at(s[BL], rci)

# Schedule for A's shared memory load
yi, xi, ci, ni = AA.axis
ty, ci = s[AA].split(ci, nparts=num_thread)
tx, ni = s[AA].split(ni, nparts=num_thread)
s[AA].reorder(ty, tx, yi, xi, ci, ni)
s[AA].bind(ty, thread_y)
s[AA].bind(tx, thread_x)

# Schedule for W's shared memory load
yi, xi, ci, fi = WW.axis
ty, ci = s[WW].split(ci, nparts=num_thread)
tx, fi = s[WW].split(fi, nparts=num_thread)
s[WW].reorder(ty, tx, yi, xi, ci, fi)
s[WW].bind(ty, thread_y)
s[WW].bind(tx, thread_x)

func = KgeN.lower(s, [A, W, B])
print(str(func))
print(KgeN.build(func))