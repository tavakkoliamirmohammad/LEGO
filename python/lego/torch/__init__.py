"""
LEGO PyTorch Integration (Layers 1–3)

Public API::

    import lego

    # Virtual — metadata only, no data movement
    x = lego.annotate(tensor, layout=lego.ZCurve(M, N))

    # Physical — rearrange data, then annotate
    y = lego.rearrange(tensor, layout=lego.Swizzle(64, 64, bits=3))

    # Compiled mode
    compiled = torch.compile(model, backend="lego")
"""

import torch
import numpy as np
from .tensor import LegoTensor, _warned_ops  # noqa: F401
from . import ops as _ops  # noqa: F401  — registers lego::mm, lego::bmm
from .ops import torch_op  # noqa: F401


# ============================================================================
# annotate — virtual layout (metadata only)
# ============================================================================

def annotate(tensor, layout):
    """Attach layout metadata to a tensor without moving data.

    If the tensor already carries a layout, the two layouts are composed::

        annotate(annotate(x, L1), L2) == annotate(x, L2 ∘ L1)
    """
    if isinstance(tensor, LegoTensor):
        try:
            composed = tensor.lego_layout.compose(layout)
        except (ValueError, AttributeError):
            composed = layout
        return LegoTensor(tensor._data, composed, tensor._is_physical)
    return LegoTensor(tensor, layout, is_physical=False)


# ============================================================================
# rearrange — physical layout (data movement + annotate)
# ============================================================================

def rearrange(tensor, layout):
    """Physically rearrange tensor data according to *layout*, then annotate.

    Uses ``lego::permute`` custom op so rearrangements survive torch.compile.
    """
    from lego.backend.compiler import LayoutCompiler

    base = layout._base if hasattr(layout, "_base") else layout
    compiler = LayoutCompiler(base._layout, base._shape, "i64")
    fwd, _ = compiler.get_permutation_table()
    fwd_idx = torch.from_numpy(np.ascontiguousarray(fwd)).to(tensor.device)

    rearranged = torch.ops.lego.permute(tensor.reshape(-1), fwd_idx).reshape(tensor.shape)
    return LegoTensor(rearranged, layout, is_physical=True)
