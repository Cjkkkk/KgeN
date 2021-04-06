# KgeN
A TVM-like CUDA code generator.

# Component
* Expression IR
* Compute primitives
* Schedule primitives
* Infer bound pass
* Cuda codegen pass

# TODO
* consolidate ranges (done)
* if_then_else expression (done)
* reduce expression (done)
* bind to thread (done)
* fix eval_expr_bound with opening and closing corner case
* add expr comparison for min max expr
* add codegen for reduce and if_then_else expr (done)