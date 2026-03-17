"""
LEGO Backend

MLIR-based compilation and dialect infrastructure for LEGO layout transforms.

Submodules:
  - compiler:       JIT-compile layout descriptors to LLVM via MLIR
  - mlir_roundtrip:  SymPy <-> MLIR roundtrip for symbolic simplification
  - torch_ops:       PyTorch autograd integration
  - dialects:        LEGO MLIR dialect Python bindings
"""
