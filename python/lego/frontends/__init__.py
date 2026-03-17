"""
LEGO Frontends

Three user-facing paths into the LEGO layout system:

  symbolic    — SymPy-based layout algebra (CPU, symbolic index math)
  triton_jit  — AST-transforming @jit decorator for Triton GPU kernels
  python_mlir — JIT-compiled layout transforms via MLIR (NumPy / PyTorch)
"""
