"""
LEGO Backend

MLIR-based compilation and dialect infrastructure for LEGO layout transforms.

Submodules:
  - symbolic:   SymPy → MLIR dialect → arith → SymPy evaluation
  - compiler:   IR builder, JIT compiler, GPU codegen helpers
  - torch_ops:  PyTorch autograd integration
  - dialects:   LEGO MLIR dialect Python bindings
"""
