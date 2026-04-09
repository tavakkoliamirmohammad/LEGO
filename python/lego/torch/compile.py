"""
LEGO Layer 3: torch.compile / Inductor Extension

Registers ``backend="lego"`` for ``torch.compile``.  The backend:

1. Propagates layout metadata through the FX graph.
2. Runs the cross-op layout planner (inserts rearrangements).
3. Registers Triton lowerings for ``lego::*`` ops  (Path B).
4. Injects LEGO index arithmetic for pointwise ops   (Path C).
5. Delegates to inductor for final compilation.
"""

import torch
from torch._dynamo import register_backend


# ============================================================================
# Path B: Register lego::* ops as inductor-compilable fallback kernels.
# ============================================================================

def _register_lego_lowerings():
    try:
        from torch._inductor.lowering import make_fallback
        import lego.torch.ops  # noqa: F401
        make_fallback(torch.ops.lego.mm)
        make_fallback(torch.ops.lego.bmm)
    except (ImportError, AttributeError):
        pass  # inductor internals changed — degrade gracefully


_register_lego_lowerings()


# ============================================================================
# Backend
# ============================================================================

@register_backend
def lego(gm, example_inputs):
    """LEGO torch.compile backend."""
    from torch._inductor.compile_fx import compile_fx
    from .tensor import LegoTensor
    from .planner import plan_layouts
    from .fusion import materialize_layouts

    # 1. Extract layout metadata from LegoTensor inputs
    layout_map = {}
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for i, inp in enumerate(example_inputs):
        if isinstance(inp, LegoTensor) and i < len(placeholders):
            layout_map[placeholders[i].name] = inp.lego_layout

    # 2. Run the cross-op layout planner
    if layout_map:
        plan_layouts(gm, layout_map)

    # 3. Materialize virtual layouts → physical order for inductor (Path C)
    unwrapped = materialize_layouts(example_inputs, layout_map, placeholders)

    # 4. Compile with inductor
    compiled_fn = compile_fx(gm, unwrapped)

    # 5. Wrapper: unwrap and materialize LegoTensor inputs at call time
    def wrapper(*args):
        plain = []
        for a in args:
            if isinstance(a, LegoTensor):
                plain.append(a._data)
            else:
                plain.append(a)
        return compiled_fn(*plain)

    return wrapper
