"""Named synthesis problems: which warp patterns touch a rows x cols tile."""
from .machine import MachineModel
from .patterns import row_vector, col_vector, ldmatrix_x4, mma_a_direct
from .search import Problem


def problem_for(kind: str, rows: int, w: int, elem_bytes: int, machine: MachineModel) -> Problem:
    """transpose: warp-per-row store, warp-per-column load, one element per lane (idle lanes).
    transpose_dense: the same with an x-major thread block, so a warp may straddle rows/columns.
    staging: cp.async-style dense 16 B stores and ldmatrix.x4 loads on a 2-byte tile.
    staging_direct: the same stores with direct 32-bit mma.sync A-fragment loads instead of ldmatrix."""
    if kind in ("transpose", "transpose_dense"):
        dense = kind == "transpose_dense"
        pats = [row_vector(rows, w, machine.warp, dense=dense), col_vector(rows, w, machine.warp, dense=dense)]
        return Problem(rows=rows, cols=w, elem_bytes=elem_bytes, patterns=pats, machine=machine, kind=kind)
    if kind in ("staging", "staging_direct"):
        unit = 16 // elem_bytes
        loads = ldmatrix_x4(rows, w) if kind == "staging" else mma_a_direct(rows, w)
        pats = [row_vector(rows, w, machine.warp, vec=unit, dense=True), loads]
        return Problem(rows=rows, cols=w, elem_bytes=elem_bytes, patterns=pats, machine=machine,
                       unit=unit, kind=kind)
    raise ValueError(kind)
