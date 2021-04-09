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
- [ ] fix eval_expr_bound with opening and closing corner case
- [ ] add expr comparison for min max expr
- [ ] add codegen for reduce and if_then_else expr
- [ ] add boundary test to avoid out of index
- [ ] add symbolic expression simplify