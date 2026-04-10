"""
LEGO Path C: Index Injection for Pointwise + Reduction Ops

When a pointwise kernel operates on a LEGO-annotated buffer, the standard
contiguous index ``i`` is replaced with a LEGO-generated index function::

    # Standard inductor:
    for i in range(N):
        val = load(buf, i)
        store(out, i, relu(val))

    # With LEGO index injection:
    for i in range(N):
        idx = lego_index(i)
        val = load(buf, idx)
        store(out, idx, relu(val))

This module provides the index-function generation infrastructure.
The actual injection into inductor's IR is performed by the compile backend.
"""

import numpy as np


def make_index_function(layout):
    """Generate a permutation-table index function from a LEGO layout.

    Returns ``(fwd_table, inv_table)`` as numpy int64 arrays.
    ``fwd_table[logical_flat_idx] == physical_flat_idx``.
    """
    from lego.backend.compiler import LayoutCompiler

    base = layout._base if hasattr(layout, "_base") else layout
    compiler = LayoutCompiler(base._layout, base._shape, "i64")
    fwd, inv = compiler.get_permutation_table()
    return np.ascontiguousarray(fwd), np.ascontiguousarray(inv)


def make_index_tensor(layout, device="cpu"):
    """Like :func:`make_index_function` but returns torch tensors on *device*."""
    import torch

    fwd, inv = make_index_function(layout)
    return (
        torch.from_numpy(fwd).to(device),
        torch.from_numpy(inv).to(device),
    )


def materialize_layouts(example_inputs, layout_map, placeholders):
    """Convert virtually-annotated inputs to physical order for inductor.

    For inputs with a non-identity virtual layout, applies the permutation
    table so inductor generates code accessing data in LEGO order.

    Returns a new list of (unwrapped, possibly rearranged) inputs.
    """
    import torch
    from .tensor import LegoTensor

    result = []
    for inp in example_inputs:
        if isinstance(inp, LegoTensor) and not inp._is_physical:
            layout = inp.lego_layout
            fwd, _ = make_index_function(layout)
            fwd_t = torch.from_numpy(fwd).to(inp.device)
            rearranged = inp._data.reshape(-1)[fwd_t].reshape(inp.shape)
            result.append(rearranged)
        elif isinstance(inp, LegoTensor):
            result.append(inp._data)
        else:
            result.append(inp)
    return result
