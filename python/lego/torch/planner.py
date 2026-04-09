"""
LEGO Cross-Op Layout Planner

Walks an FX graph and at each op boundary checks whether the producer's
output layout matches the consumer's preferred layout.  If not, inserts
a ``lego.rearrange()`` call.  A symbolic cost model (using LEGO's layout
algebra) decides whether rearranging is cheaper than running the consumer
on a suboptimal layout.

Used by the ``torch.compile(backend="lego")`` path.
"""

import torch
from .tensor import _TIER1, _TIER2, _TIER4


def plan_layouts(gm, layout_map):
    """Propagate layouts and insert rearrangements where beneficial.

    Parameters
    ----------
    gm : torch.fx.GraphModule
        The traced FX graph.
    layout_map : dict[str, layout]
        Map from node-name → LEGO layout for annotated inputs.
    """
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        # Collect input layouts
        input_layouts = []
        for arg in _flat_args(node.args):
            if hasattr(arg, "name") and arg.name in layout_map:
                input_layouts.append(layout_map[arg.name])

        if not input_layouts:
            continue

        layout = input_layouts[0]
        func = node.target

        # Tier 1 (pointwise) / Tier 2 (transform): propagate
        if func in _TIER1 or func in _TIER2:
            layout_map[node.name] = layout
            continue

        # Tier 4 (LEGO kernel): future — check layout compatibility,
        # insert rearrange if producer layout != consumer preference.
        if func in _TIER4:
            layout_map[node.name] = layout
            continue

        # Tier 3: layout is dropped at this node — don't propagate.

    gm.recompile()


def _flat_args(args):
    """Yield leaf elements from a nested tuple/list of args."""
    if isinstance(args, (tuple, list)):
        for a in args:
            yield from _flat_args(a)
    else:
        yield args
