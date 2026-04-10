"""
LEGO Layout Autotuning

Grid search over tile sizes to find the optimal configuration for a given
shape and device. Properly synchronizes CUDA for accurate timing.

Usage:
    from lego.torch.autotune import autotune
    layout = autotune(shape=(512, 512), device="cuda")
"""

import time
import torch
import numpy as np

_CACHE: dict = {}


def _default_tile_candidates(shape):
    """Generate tile size candidates that evenly divide the shape."""
    sizes = [2, 4, 8, 16, 32, 64, 128]
    candidates = []
    for t in sizes:
        if all(s % t == 0 and t <= s for s in shape):
            candidates.append(tuple(t for _ in shape))
    return candidates if candidates else [tuple(min(4, s) for s in shape)]


def _benchmark_tile(shape, tile_shape, n_iters, device):
    """Benchmark a single tile configuration. Returns mean time in seconds."""
    from lego.frontends.python_mlir import TiledPermute

    layout = TiledPermute(shape, tile_shape=tile_shape)

    if device == "cpu":
        data = np.random.randn(*shape).astype(np.float32)
        for _ in range(3):
            layout.transform(data)
        start = time.perf_counter()
        for _ in range(n_iters):
            layout.transform(data)
        elapsed = time.perf_counter() - start
    else:
        data = torch.randn(*shape, device=device)
        for _ in range(3):
            layout.transform(data)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            layout.transform(data)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return elapsed / n_iters


def autotune(shape, tile_candidates=None, n_iters=20, device="cpu", force=False):
    """Find the best tile size for a shape by benchmarking.

    Parameters
    ----------
    shape : tuple of int
    tile_candidates : list of tuple, optional
    n_iters : int
    device : str
    force : bool

    Returns
    -------
    LegoLayout
    """
    from lego.frontends.python_mlir import TiledPermute

    cache_key = f"{shape}:{device}"
    if not force and cache_key in _CACHE:
        return _CACHE[cache_key]

    if tile_candidates is None:
        tile_candidates = _default_tile_candidates(shape)

    best_time = float("inf")
    best_tile = tile_candidates[0]

    for tile in tile_candidates:
        try:
            t = _benchmark_tile(shape, tile, n_iters, device)
            if t < best_time:
                best_time = t
                best_tile = tile
        except (ValueError, RuntimeError):
            continue

    result = TiledPermute(shape, tile_shape=best_tile)
    _CACHE[cache_key] = result
    return result


def clear_cache():
    """Clear the autotune cache."""
    _CACHE.clear()
