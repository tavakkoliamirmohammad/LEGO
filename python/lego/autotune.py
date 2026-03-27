"""
LEGO Layout Autotuning

Grid search over tile sizes to find the optimal configuration for a given
shape and device. Results are cached to a JSON file for reuse.

Usage:
    from lego.autotune import autotune
    layout = autotune(shape=(512, 512),
                      tile_candidates=[(32,32), (64,64), (128,128)])
    result = layout.transform(data)
"""

import json
import os
import time
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "lego")
_CACHE_FILE = os.path.join(_CACHE_DIR, "autotune_cache.json")

# In-memory cache
_AUTOTUNE_CACHE = {}


def _load_disk_cache():
    """Load autotune results from disk."""
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_disk_cache(cache):
    """Save autotune results to disk."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def _cache_key(shape, device="cpu"):
    """Create a cache key from shape and device."""
    return f"{shape}:{device}"


def _default_tile_candidates(shape):
    """Generate default tile size candidates for a shape."""
    candidates = []
    rank = len(shape)
    tile_sizes = [2, 4, 8, 16, 32, 64, 128]

    if rank == 2:
        for t in tile_sizes:
            if all(s % t == 0 for s in shape) and t <= min(shape):
                candidates.append((t, t))
    else:
        # For higher rank, try uniform tile sizes
        for t in tile_sizes:
            tile = tuple(t for _ in shape)
            if all(s % t == 0 for s, t in zip(shape, tile)):
                candidates.append(tile)
    return candidates


def _benchmark_tile(shape, tile_shape, n_iters=20, device="cpu"):
    """Benchmark a single tile configuration. Returns mean time in ms."""
    from lego.frontends.python_mlir import Tiled
    try:
        layout = Tiled(shape, tile_shape)
    except ValueError:
        return float('inf')

    if _HAS_TORCH and device != "cpu":
        data = torch.randn(*shape, device=device)
    else:
        data = np.random.randn(*shape).astype(np.float32)

    # Warmup
    for _ in range(3):
        layout.transform(data)

    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        layout.transform(data)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times)


def autotune(shape, tile_candidates=None, n_iters=20, device="cpu", force=False):
    """Find the best tile size for a given shape by benchmarking.

    Parameters
    ----------
    shape : tuple of int
        Tensor shape to tile.
    tile_candidates : list of tuples, optional
        Tile sizes to try. If None, generates candidates automatically.
    n_iters : int
        Number of iterations per candidate.
    device : str
        Device to benchmark on ("cpu" or "cuda").
    force : bool
        If True, re-run even if cached result exists.

    Returns
    -------
    TiledView
        The layout with the best tile size.
    """
    from lego.frontends.python_mlir import Tiled

    key = _cache_key(shape, device)

    # Check in-memory cache
    if not force and key in _AUTOTUNE_CACHE:
        return Tiled(shape, _AUTOTUNE_CACHE[key])

    # Check disk cache
    if not force:
        disk_cache = _load_disk_cache()
        if key in disk_cache:
            tile = tuple(disk_cache[key])
            _AUTOTUNE_CACHE[key] = tile
            return Tiled(shape, tile)

    if tile_candidates is None:
        tile_candidates = _default_tile_candidates(shape)

    if not tile_candidates:
        raise ValueError(
            f"No valid tile candidates for shape {shape}. "
            f"Provide tile_candidates explicitly."
        )

    best_tile = None
    best_time = float('inf')

    for tile in tile_candidates:
        t = _benchmark_tile(shape, tile, n_iters=n_iters, device=device)
        if t < best_time:
            best_time = t
            best_tile = tile

    # Cache the result
    _AUTOTUNE_CACHE[key] = best_tile
    disk_cache = _load_disk_cache()
    disk_cache[key] = list(best_tile)
    _save_disk_cache(disk_cache)

    return Tiled(shape, best_tile)


def clear_cache():
    """Clear all autotune caches (in-memory and disk)."""
    _AUTOTUNE_CACHE.clear()
    if os.path.exists(_CACHE_FILE):
        os.remove(_CACHE_FILE)
