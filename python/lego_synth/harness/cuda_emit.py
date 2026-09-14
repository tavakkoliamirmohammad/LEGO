"""Emit a concrete layout's element index as a CUDA C expression in `i` and `j`."""
from ..layouts import RowMajor, XorSwizzle, Rotation


def index_expr(layout) -> str:
    if isinstance(layout, RowMajor):
        return f"((i)*{layout.cols + layout.pad} + (j))"
    if isinstance(layout, XorSwizzle):
        return f"((i)*{layout.cols} + ((j) ^ ((((i) >> {layout.shift}) & {layout.mask}) * {layout.unit})))"
    if isinstance(layout, Rotation):
        if layout.unit == 1:
            return f"((i)*{layout.cols} + (((j) + {layout.r}*(i)) % {layout.cols}))"
        n = layout.cols // layout.unit
        return (f"((i)*{layout.cols} + ((((j)/{layout.unit} + {layout.r}*(i)) % {n})*{layout.unit}"
                f" + (j)%{layout.unit}))")
    raise TypeError(f"no CUDA emitter for {type(layout).__name__}")
