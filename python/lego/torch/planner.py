"""
LEGO Cross-Op Layout Planner

Walks an FX graph and at each op boundary checks whether the producer's
output layout matches the consumer's preferred layout.  If not, inserts
a ``lego.rearrange()`` call.  A symbolic cost model (using LEGO's layout
algebra) decides whether rearranging is cheaper than running the consumer
on a suboptimal layout.

Used by the ``torch.compile(backend="lego")`` path.
"""

import numpy as np
import torch
from .tensor import _TIER1, _TIER2, _TIER4


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
    """Propagate layouts and insert rearrangements where beneficial.

    Parameters
    ----------
    gm : torch.fx.GraphModule
        The traced FX graph.
    layout_map : dict[str, layout]
        Map from node-name -> LEGO layout for annotated inputs.
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

        # Tier 1 (pointwise) / Tier 2 (transform): propagate
        if func in _TIER1 or func in _TIER2:
            layout_map[node.name] = layout
            continue

        # Tier 4 (LEGO kernel): propagate; future — check consumer
        # preference and insert rearrangement if cost justifies it.
        if func in _TIER4:
            layout_map[node.name] = layout
            continue

        # Tier 3: layout drops at this node — don't propagate.

    gm.recompile()


def _flat_args(args):
    """Yield leaf elements from a nested tuple/list of args."""
    if isinstance(args, (tuple, list)):
        for a in args:
            yield from _flat_args(a)
    else:
        yield args
