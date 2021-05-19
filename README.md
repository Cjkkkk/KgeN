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