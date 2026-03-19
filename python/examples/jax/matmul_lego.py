"""JAX tiled matmul — LEGO layouts for flat-offset gather + block accumulation.

Mirrors the Triton LEGO matmul pattern: defines TileBy layouts for A, B, C,
then uses LEGO slice indexing to gather (BM, BK) / (BK, BN) tiles from
flattened arrays.  Verified against PyTorch (torch.matmul).

Requires: jax, jaxlib, torch.
"""

import functools
import numpy as np
import torch
import jax
import jax.numpy as jnp
from lego.frontends.jax_jit import jit as lego_jit
from lego.core import OrderBy, Row


@lego_jit
@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def matmul(A_flat, B_flat, M, N, K, BM, BN, BK):
    """Tiled matmul: C(M,N) = A(M,K) @ B(K,N), using LEGO offset layouts."""

    # LEGO layouts — identical to the Triton matmul example
    L_A = OrderBy(Row(M, K)).TileBy([M / BM, K / BK], [BM, BK])
    L_B = OrderBy(Row(K, N)).TileBy([K / BK, N / BN], [BK, BN])
    L_C = OrderBy(Row(M, N)).TileBy([M / BM, N / BN], [BM, BN])

    C_flat = jnp.zeros(M * N, dtype=A_flat.dtype)

    for pm in range(M // BM):
        for pn in range(N // BN):
            acc = jnp.zeros((BM, BN), dtype=jnp.float32)
            for k in range(K // BK):
                # Gather tiles via LEGO offsets (same pattern as Triton)
                a_tile = A_flat[L_A[pm, k, :, :]]   # (BM, BK)
                b_tile = B_flat[L_B[k, pn, :, :]]   # (BK, BN)
                acc = acc + a_tile @ b_tile

            # Scatter result tile via LEGO offsets
            c_offs = L_C[pm, pn, :, :]               # (BM, BN)
            C_flat = C_flat.at[c_offs].set(acc)

    return C_flat


def jax_matmul(A, B, BM=32, BN=32, BK=32):
    M, K = A.shape
    _, N = B.shape
    C_flat = matmul(A.reshape(-1), B.reshape(-1), M, N, K, BM, BN, BK)
    return C_flat.reshape(M, N)


if __name__ == "__main__":
    torch.manual_seed(0)
    for size in [64, 128, 256]:
        M, N, K = size, size, size
        a_torch = torch.randn(M, K)
        b_torch = torch.randn(K, N)

        a_jax = jnp.array(a_torch.numpy())
        b_jax = jnp.array(b_torch.numpy())

        # Compare LEGO tiled matmul against jnp.dot (same device/precision)
        c_lego = jax_matmul(a_jax, b_jax)
        c_ref = jnp.dot(a_jax, b_jax)

        ok = np.allclose(np.asarray(c_lego), np.asarray(c_ref), atol=1e-4)
        print(f"{M}x{K} @ {K}x{N}  match={ok}")
        assert ok, f"Mismatch at size={size}"
    print("PASS: JAX LEGO matmul matches jnp.dot")
