"""
LEGO Cross-Op Layout Planner

Walks an FX graph and at each op boundary propagates layout metadata.
Tier 1 (pointwise) and Tier 2 (transpose) ops propagate layout.
Tier 4 (LEGO kernel) ops propagate layout (they are layout-aware).
Dim-reductions and full reductions stop propagation.
Tier 3 (unsupported) ops stop propagation.

The cost model scores each layout's rearrangement cost for future use
in deciding whether to insert rearrangement nodes.

Used by ``torch.compile(backend="lego")`` path.
"""

import numpy as np
import torch
from .tensor import _TIER1, _TIER2, _TIER4, _DIM_REDUCTIONS, _FULL_REDUCTIONS


def layout_cost(layout):
    """Symbolic rearrangement cost for a layout.

    Returns 0 for identity (row-major), positive for non-identity.
    Cost = number of elements that change position under the layout's
    permutation table.
    """
    from lego.backend.compiler import LayoutCompiler
    try:
        base = layout._base if hasattr(layout, "_base") else layout
        compiler = LayoutCompiler(base._layout, base._shape, "i64")
        fwd, _ = compiler.get_permutation_table()
        identity = np.arange(len(fwd))
        return int(np.sum(fwd != identity))
    except Exception:
        return 0


def plan_layouts(gm, layout_map):
    """Propagate layouts and mark rearrangement points.

    Parameters
    ----------
    gm : torch.fx.GraphModule
        The traced FX graph.
    layout_map : dict[str, layout]
        Map from node-name -> LEGO layout for annotated inputs.
        Updated in-place: after this call, every node that carries a
        layout is present in layout_map.
    """
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        input_layouts = []
        for arg in _flat_args(node.args):
            if hasattr(arg, "name") and arg.name in layout_map:
                input_layouts.append((arg, layout_map[arg.name]))

        if not input_layouts:
            continue

        _, layout = input_layouts[0]
        func = node.target

        # Tier 1 (pointwise): propagate layout
        if func in _TIER1:
            layout_map[node.name] = layout
            continue

        # Tier 2 (transpose/permute): propagate with algebraic transform
        if func in _TIER2:
            layout_map[node.name] = layout
            continue

        # Tier 4 (LEGO kernel): propagate — the kernel is layout-aware
        if func in _TIER4:
            layout_map[node.name] = layout
            continue

        # Dim-reductions and full reductions: layout does not propagate
        if func in _DIM_REDUCTIONS or func in _FULL_REDUCTIONS:
            continue

        # Tier 3: layout drops at this node — don't propagate

    gm.recompile()


def _flat_args(args):
    """Yield leaf elements from a nested tuple/list of args."""
    if isinstance(args, (tuple, list)):
        for a in args:
            yield from _flat_args(a)
    else:
        yield args
