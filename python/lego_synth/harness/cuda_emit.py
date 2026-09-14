"""Emit a concrete layout's element index as a CUDA C expression in `i` and `j`."""
from ..layouts import RowMajor, XorSwizzle, Rotation


def index_expr(layout) -> str:
    if isinstance(layout, RowMajor):
        return f"((i)*{layout.cols + layout.pad} + (j))"
    if isinstance(layout, XorSwizzle):
        return f"((i)*{layout.cols} + ((j) ^ ((((i) >> {layout.shift}) & {layout.mask}) * {layout.unit})))"
    if isinstance(layout, Rotation):
        return f"((i)*{layout.cols} + (((j) + {layout.r}*(i)) % {layout.cols}))"
    raise TypeError(f"no CUDA emitter for {type(layout).__name__}")
