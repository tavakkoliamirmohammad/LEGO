# 09_gemm_zmorton — Z-Morton GEMM

**CASTLE candidate:** 01 — polybench-gemm-zmorton  
**Layout class:** Z-Morton  
**Prior verdicts:** AMD WIN (2.07–3.30×), Intel WIN (1.15–2.27×)

## Kernel

GEMM-style gather where the A matrix is stored in Z-Morton (Z-order) layout.
The cpu_dsl version approximates this with a 1-D Morton index swizzle using
bitwise interleaving operators (`&`, `>>`, `|`, `<<`) supported in the DSL.

## Expected behavior

The NonAffine gather path in lego-vectorize emits `vector.gather` for the
Morton-decoded read of `A[morton(i)]`. This path is exercised by this candidate.

vec_iso ≥ 1.0× expected (gather + FMA should be comparable to scalar loop).
