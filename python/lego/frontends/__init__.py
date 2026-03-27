"""
LEGO Frontends

User-facing paths into the LEGO layout system:

  triton_jit  — AST-transforming @jit decorator for Triton GPU kernels
  cutile_jit  — AST-transforming @jit decorator for NVIDIA cuTile kernels
  numba_jit   — AST-transforming @jit decorator for Numba CUDA kernels
  jax_jit     — AST-transforming @jit decorator for JAX functions
  python_mlir — JIT-compiled layout transforms via MLIR (NumPy / PyTorch)

All JIT frontends share the DSL-agnostic rewriter (lego.rewriter) and
implement the DSLAdapter interface (_adapter.py).
"""
