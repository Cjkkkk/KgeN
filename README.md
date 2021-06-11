# KgeN
A TVM-like CUDA code generator.

# Component
* Expression IR
* Compute primitives
* Schedule primitives
* Infer bound pass
* Cuda codegen pass

# TODO
- [x] consolidate ranges
- [x] if_then_else expression
- [x] reduce expression
- [x] bind to thread
- [x] fix pass up and pass down
- [x] bound normalization and cosumer index change
- [x] fix eval_expr_bound with opening and closing corner case
- [ ] add expr comparison for min max expr
- [x] add codegen for reduce and if_then_else expr
- [ ] add boundary test to avoid out of index
- [x] add symbolic expression simplify
- [x] apply expr simplifier
- [x] fix attach.py
- [x] fix bound normalization
- [x] fix recursive attach path
- [x] change codegen to visitor pattern
- [x] transform into stmts
- [x] tensor index flatten 
- [x] cache read dataflow rewrite
- [ ] cache read virtual thread and reduce axis
- [x] cache write
- [x] move collect input as build graph pass
- [x] expand split axis to enable expr simplify(i - > i_outer * 32 + i_inner)
- [x] normalize single point or not?
- [ ] add expr simplify single point iter var as const expr
- [x] add sync_threads()
- [x] add unroll

# example
```
python3 -m example.matmul_cache_write
```

```c
// tensor: C[64, 64]
// tensor: C_local[4, 4]
// tensor: A_shared_local[4, 1]
// tensor: A_shared[16, 32]
// tensor: A[64, 64]
// tensor: B_shared_local[1, 4]
// tensor: B_shared[32, 16]
// tensor: B[64, 64]
// gridDim: [4, 4, 1]
// blockDim: [4, 4, 1]
__global__ void kernel(float* A, float* B, float* C) {
    float C_local[16];
    float A_shared_local[4];
    __shared__ float A_shared[512];
    float B_shared_local[4];
    __shared__ float B_shared[512];
    #pragma unroll
    for (int C_local_i = 0; C_local_i < 4 ; C_local_i += 1) {
        #pragma unroll
        for (int C_local_j = 0; C_local_j < 4 ; C_local_j += 1) {
            C_local[((C_local_i * 4) + C_local_j)] = 0;
        }
    }
    for (int k_outer = 0; k_outer < 2 ; k_outer += 1) {
        for (int A_shared_i0_inner = 0; A_shared_i0_inner < 4 ; A_shared_i0_inner += 1) {
            for (int A_shared_i1_inner = 0; A_shared_i1_inner < 8 ; A_shared_i1_inner += 1) {
                A_shared[((((threadIdx.x * 4) + A_shared_i0_inner) * 32) + ((threadIdx.y * 8) + A_shared_i1_inner))] = A[(((((threadIdx.x * 4) + A_shared_i0_inner) + (blockIdx.x * 16)) * 64) + (((threadIdx.y * 8) + A_shared_i1_inner) + (k_outer * 32)))];
            }
        }
        for (int B_shared_i0_inner = 0; B_shared_i0_inner < 8 ; B_shared_i0_inner += 1) {
            for (int B_shared_i1_inner = 0; B_shared_i1_inner < 4 ; B_shared_i1_inner += 1) {
                B_shared[((((threadIdx.x * 8) + B_shared_i0_inner) * 16) + ((threadIdx.y * 4) + B_shared_i1_inner))] = B[(((((threadIdx.x * 8) + B_shared_i0_inner) + (k_outer * 32)) * 64) + (((threadIdx.y * 4) + B_shared_i1_inner) + (blockIdx.y * 16)))];
            }
        }
        __syncthreads();
        for (int k_inner = 0; k_inner < 32 ; k_inner += 1) {
            for (int A_shared_local_i0 = 0; A_shared_local_i0 < 4 ; A_shared_local_i0 += 1) {
                A_shared_local[(A_shared_local_i0 + 0)] = A_shared[((((A_shared_local_i0 + (((blockIdx.x * 4) + threadIdx.x) * 4)) - (blockIdx.x * 16)) * 32) + ((0 + ((k_outer * 32) + k_inner)) - (k_outer * 32)))];
            }
            for (int B_shared_local_i1 = 0; B_shared_local_i1 < 4 ; B_shared_local_i1 += 1) {
                B_shared_local[((0 * 4) + B_shared_local_i1)] = B_shared[((((0 + ((k_outer * 32) + k_inner)) - (k_outer * 32)) * 16) + ((B_shared_local_i1 + (((blockIdx.y * 4) + threadIdx.y) * 4)) - (blockIdx.y * 16)))];
            }
            #pragma unroll
            for (int C_local_i = 0; C_local_i < 4 ; C_local_i += 1) {
                #pragma unroll
                for (int C_local_j = 0; C_local_j < 4 ; C_local_j += 1) {
                    C_local[((C_local_i * 4) + C_local_j)] = (C_local[((C_local_i * 4) + C_local_j)] + (A_shared_local[C_local_i] * B_shared_local[C_local_j]));
                }
            }
        }
        __syncthreads();
    }
    for (int C_i_inner = 0; C_i_inner < 4 ; C_i_inner += 1) {
        for (int C_j_inner = 0; C_j_inner < 4 ; C_j_inner += 1) {
            C[((((((blockIdx.x * 4) + threadIdx.x) * 4) + C_i_inner) * 64) + ((((blockIdx.y * 4) + threadIdx.y) * 4) + C_j_inner))] = C_local[(((((((blockIdx.x * 4) + threadIdx.x) * 4) + C_i_inner) - (((blockIdx.x * 4) + threadIdx.x) * 4)) * 4) + (((((blockIdx.y * 4) + threadIdx.y) * 4) + C_j_inner) - (((blockIdx.y * 4) + threadIdx.y) * 4)))];
        }
    }
}
```