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


@register_backend
def lego(gm, example_inputs):
    """LEGO torch.compile backend."""
    from torch._inductor.compile_fx import compile_fx
    from .tensor import LegoTensor
    from .planner import plan_layouts

    # 1. Extract layout metadata from LegoTensor inputs
    layout_map = {}
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for i, inp in enumerate(example_inputs):
        if isinstance(inp, LegoTensor) and i < len(placeholders):
            layout_map[placeholders[i].name] = inp.lego_layout

    # 2. Run the cross-op layout planner
    if layout_map:
        plan_layouts(gm, layout_map)

    # 3. Unwrap LegoTensors so inductor sees plain tensors
    unwrapped = [inp._data if isinstance(inp, LegoTensor) else inp for inp in example_inputs]

    # 4. Compile with inductor
    compiled_fn = compile_fx(gm, unwrapped)

    # 5. Wrapper: unwrap LegoTensor inputs at call time
    def wrapper(*args):
        plain = [a._data if isinstance(a, LegoTensor) else a for a in args]
        return compiled_fn(*plain)

    return wrapper
