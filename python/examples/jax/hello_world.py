"""JAX vecadd — LEGO hello world. Requires jax + jaxlib."""

import functools
import numpy as np
import jax
import jax.numpy as jnp
from lego.frontends.jax_jit import jit as lego_jit
from lego.core import OrderBy, Row


@lego_jit
@functools.partial(jax.jit, static_argnums=(2,))
def vecadd(x, y, N):
    L = OrderBy(Row(N)).TileBy([N])
    offs = L[:]
    return x[offs] + y[offs]


if __name__ == "__main__":
    np.random.seed(0)
    for N in [1024, 8192, 2**16]:
        x_np = np.random.randn(N).astype(np.float32)
        y_np = np.random.randn(N).astype(np.float32)
        expected = x_np + y_np

        x_jax = jnp.array(x_np)
        y_jax = jnp.array(y_np)
        z_jax = vecadd(x_jax, y_jax, N)
        z_np = np.asarray(z_jax)

        ok = np.allclose(z_np, expected, atol=1e-5)
        print(f"N={N:>6d}  match={ok}")
        assert ok, f"Mismatch at N={N}"
    print("PASS: JAX vecadd matches NumPy")
