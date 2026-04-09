"""
Prototype: Automatic Communication Pattern Derivation from LEGO Layouts

Given:
  - A set of tensors, each with a LEGO distribution (data layout + proc layout)
  - An operation with index dependencies (matmul, elementwise, stencil, reduction)

Derive:
  - The communication collective type (AllGather, AllReduce, HaloExchange, None)
  - The symbolic communication volume
  - Which dimensions are local vs communicated

All index math via LEGO's simplify_via_mlir. Zero manual floor/mod.
"""
import sympy as sp
from sympy import symbols, floor
from lego.core import Row, Col, OrderBy
from lego.backend.symbolic import simplify_via_mlir
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# ============================================================================
# Core abstractions
# ============================================================================

class CollectiveType(Enum):
    NONE = "none"
    ALLGATHER = "allgather"
    ALLREDUCE = "allreduce"
    REDUCE_SCATTER = "reduce_scatter"
    HALO_EXCHANGE = "halo_exchange"
    ALLTOALL = "alltoall"


@dataclass
class TensorDist:
    """A distributed tensor: shape + block sizes + LEGO data layout.

    Distribution is a 2D block distribution over a Pr x Pc processor grid.
    Proc ownership is computed via LEGO Row.inv (no manual indexing).

    Args:
        name: tensor name (for display)
        indices: symbolic index variables, e.g. (i, k) for A in matmul
        shape: symbolic dimensions, e.g. (M, K)
        blocks: block size per dimension, e.g. (floor(M/Pr), floor(K/Pc))
        tiles: number of tiles per dimension, e.g. (Pr, Pc)
        data_layout: LEGO layout block (Row or Col) for physical storage
    """
    name: str
    indices: Tuple[sp.Symbol, ...]
    shape: Tuple[sp.Expr, ...]
    blocks: Tuple[sp.Expr, ...]
    tiles: Tuple[sp.Expr, ...]
    data_layout: object  # Row(...) or Col(...)

    def owner(self):
        """Compute symbolic proc_id via LEGO Row.inv + Row.apply.

        Row(tiles_d, block_d).inv(idx_d) → [tile_coord, local_offset]
        tile_coord IS the proc coordinate for dimension d.
        """
        proc_coords = []
        for idx, tile_count, block_size in zip(self.indices, self.tiles, self.blocks):
            decomp = simplify_via_mlir(
                Row(tile_count, block_size), 'inv', idx,
                constraints={idx: (0, tile_count * block_size)})
            proc_coords.append(decomp[0])

        # Linearize proc coords into a scalar proc_id
        proc_id = simplify_via_mlir(
            Row(*self.tiles), 'apply', proc_coords)
        return proc_id

    def flat(self):
        """Compute flat data index via LEGO data layout apply."""
        return simplify_via_mlir(
            self.data_layout, 'apply', list(self.indices),
            constraints={idx: (0, dim) for idx, dim in zip(self.indices, self.shape)})

    def tile_flat(self):
        """Compute TileBy flat index (physical layout within proc blocks)."""
        dist = OrderBy(self.data_layout).TileBy(list(self.tiles), list(self.blocks))
        # TileBy has rank 2*ndim: (tile_0, tile_1, ..., local_0, local_1, ...)
        tile_syms = [sp.Symbol(f'_t{d}', nonneg=True, integer=True)
                     for d in range(len(self.tiles))]
        local_syms = [sp.Symbol(f'_l{d}', nonneg=True, integer=True)
                      for d in range(len(self.blocks))]
        all_syms = tile_syms + local_syms
        bounds = {}
        for s, bound in zip(tile_syms, self.tiles):
            bounds[s] = (0, bound)
        for s, bound in zip(local_syms, self.blocks):
            bounds[s] = (0, bound)
        return simplify_via_mlir(dist, 'apply', all_syms, constraints=bounds)


@dataclass
class Dependency:
    """Describes how an output index maps to an input index.

    For matmul C[i,j] = sum_k A[i,k] * B[k,j]:
      A's dependency: output (i,j) needs A at indices (i, k) for all k in [0,K)
      index_map: {i: i, k: k}  (i comes from output, k is the contracted var)
      contracted: [k]  (k is summed over — ranges freely)

    For elementwise C[i,j] = A[i,j] + B[i,j]:
      index_map: {i: i, j: j}
      contracted: []  (no free-ranging indices)

    For stencil C[i] = A[i-1] + A[i] + A[i+1]:
      index_map: {i: i}  (center), plus offsets
      stencil_offsets: {i: [-1, 0, 1]}
    """
    input_tensor: TensorDist
    index_map: Dict[sp.Symbol, sp.Expr]
    contracted: List[sp.Symbol] = field(default_factory=list)
    stencil_offsets: Dict[sp.Symbol, List[int]] = field(default_factory=dict)


@dataclass
class CommPattern:
    """Result of communication analysis for one input tensor."""
    input_name: str
    collective: CollectiveType
    mismatch_expr: sp.Expr
    free_indices: set
    local_indices: set
    communicated_indices: set
    volume_expr: Optional[sp.Expr] = None
    detail: str = ""


# ============================================================================
# The derivation engine
# ============================================================================

def derive_communication(output: TensorDist,
                         deps: List[Dependency]) -> List[CommPattern]:
    """
    Derive communication patterns for all input tensors of an operation.

    Algorithm (all in LEGO flat space):
      1. Compute owner_output(output_indices) via LEGO Row.inv + Row.apply
      2. For each input dep:
         a. Compute owner_input(input_indices) via LEGO Row.inv + Row.apply
         b. Substitute the dependency's index_map to express owner_input
            in terms of the OUTPUT indices + any contracted indices
         c. mismatch = owner_output - owner_input (substituted)
         d. Analyze free variables in mismatch → classify collective
    """
    owner_out = output.owner()
    results = []

    for dep in deps:
        inp = dep.input_tensor
        owner_in_raw = inp.owner()

        # Substitute: express input ownership in terms of output + contracted indices
        # e.g., for matmul A dep: owner_A is in terms of (i, k)
        #   index_map says i→i, k→k (already the right symbols)
        #   but for more complex ops, the map might remap indices
        sub_map = {}
        for inp_idx, out_expr in dep.index_map.items():
            sub_map[inp_idx] = out_expr
        owner_in = owner_in_raw.subs(sub_map)

        # Form mismatch
        mismatch = sp.expand(owner_out - owner_in)

        # Collect all "loop" indices: output indices + contracted indices
        all_loop_indices = set(output.indices) | set(dep.contracted)
        free = mismatch.free_symbols & all_loop_indices
        local = all_loop_indices - free
        communicated = free & set(dep.contracted)

        # Classify
        if dep.stencil_offsets:
            collective = _classify_stencil(output, dep, owner_out, owner_in)
        elif mismatch == 0 or sp.simplify(mismatch) == 0:
            collective = CollectiveType.NONE
        elif communicated:
            # Contracted dim appears in mismatch → need collective over it
            # If the output is distributed along the contracted dim → ReduceScatter
            # If the output is NOT distributed along contracted dim → AllGather or AllReduce
            collective = _classify_reduction(output, dep, mismatch, communicated)
        elif free:
            collective = CollectiveType.ALLTOALL
        else:
            collective = CollectiveType.NONE

        # Symbolic volume
        volume = _compute_volume(output, dep, collective)

        # Detail string
        detail = _format_detail(collective, local, communicated, free, dep)

        results.append(CommPattern(
            input_name=inp.name,
            collective=collective,
            mismatch_expr=mismatch,
            free_indices=free,
            local_indices=local,
            communicated_indices=communicated,
            volume_expr=volume,
            detail=detail,
        ))

    return results


def _classify_reduction(output, dep, mismatch, communicated):
    """Classify whether a reduction-axis mismatch is AllGather or AllReduce.

    AllGather: input is distributed along contracted dim, output is not.
      Each proc needs to GATHER the full contracted axis from other procs,
      then compute locally. Communication happens BEFORE the reduction.

    AllReduce: each proc computes a PARTIAL result over its local slice of
      the contracted dim, then reduces across procs. Communication happens
      AFTER the reduction.

    In practice, the choice depends on whether the input distribution
    along the contracted dim matches the output distribution. For SUMMA-style:
    the input (A) is distributed along k, the output (C) is not → AllGather.
    """
    # Check if the contracted dim has the same distribution in input and output
    # If so, each proc computes a partial sum → AllReduce
    # If not, need to gather first → AllGather
    #
    # Heuristic: if the mismatch involves ONLY contracted indices
    # (no output indices), it's a pure gather along the contracted axis.
    output_indices_in_mismatch = mismatch.free_symbols & set(output.indices)
    contracted_only = (communicated == mismatch.free_symbols & (set(output.indices) | set(dep.contracted)))

    if not output_indices_in_mismatch - communicated:
        # Mismatch is purely from contracted dims → AllGather
        return CollectiveType.ALLGATHER
    else:
        return CollectiveType.ALLREDUCE


def _classify_stencil(output, dep, owner_out, owner_in):
    """Classify stencil pattern by checking if offsets cross proc boundaries."""
    # For stencil: check if owner(i+offset) can differ from owner(i)
    for idx, offsets in dep.stencil_offsets.items():
        for off in offsets:
            if off == 0:
                continue
            shifted_owner = owner_in.subs(idx, idx + off)
            diff = sp.expand(owner_out - shifted_owner)
            if diff != 0 and sp.simplify(diff) != 0:
                return CollectiveType.HALO_EXCHANGE
    return CollectiveType.NONE


def _compute_volume(output, dep, collective):
    """Compute symbolic communication volume per processor."""
    inp = dep.input_tensor
    if collective == CollectiveType.NONE:
        return sp.Integer(0)

    if collective == CollectiveType.ALLGATHER:
        # Volume = elements needed - elements owned (per proc)
        # Needed: full extent of contracted dims × local extents of other dims
        # Owned: local extents of all dims
        owned = sp.Integer(1)
        needed = sp.Integer(1)
        for idx, block, tile, shape in zip(inp.indices, inp.blocks, inp.tiles, inp.shape):
            if idx in dep.contracted:
                owned *= block           # local slice of contracted dim
                needed *= shape          # full extent of contracted dim
            else:
                owned *= block
                needed *= block
        return sp.simplify(needed - owned)

    if collective == CollectiveType.HALO_EXCHANGE:
        # Volume = boundary elements per face × number of faces
        total = sp.Integer(0)
        for idx, offsets in dep.stencil_offsets.items():
            radius = max(abs(o) for o in offsets if o != 0)
            # Halo elements = radius × product of other block dims
            face_size = sp.Integer(radius)
            for other_idx, block in zip(inp.indices, inp.blocks):
                if other_idx != idx:
                    face_size *= block
            total += 2 * face_size  # both sides
        return sp.simplify(total)

    if collective == CollectiveType.ALLREDUCE:
        # Volume = local output block size (each proc contributes a partial result)
        vol = sp.Integer(1)
        for block in output.blocks:
            vol *= block
        return sp.simplify(vol)

    return None


def _format_detail(collective, local, communicated, free, dep):
    """Human-readable explanation."""
    parts = []
    if local:
        parts.append(f"local dims: {{{', '.join(str(v) for v in sorted(local, key=str))}}}")
    if communicated:
        parts.append(f"communicated dims: {{{', '.join(str(v) for v in sorted(communicated, key=str))}}}")
    if collective == CollectiveType.ALLGATHER:
        parts.append("gather full extent of contracted dim from all procs")
    elif collective == CollectiveType.HALO_EXCHANGE:
        for idx, offsets in dep.stencil_offsets.items():
            r = max(abs(o) for o in offsets if o != 0)
            parts.append(f"exchange {r}-element halo along {idx}")
    elif collective == CollectiveType.ALLREDUCE:
        parts.append("each proc computes partial result, then reduce across procs")
    return "; ".join(parts)


# ============================================================================
# Pretty printer
# ============================================================================

def print_results(op_name, output, deps, results):
    """Print the full derivation."""
    print("\n" + "=" * 70)
    print(f"  {op_name}")
    print("=" * 70)

    print(f"\n  Output: {output.name}[{','.join(str(s) for s in output.indices)}]"
          f"  shape={output.shape}  tiles={output.tiles}  blocks={output.blocks}")
    print(f"  Output owner: {output.owner()}")

    for dep, res in zip(deps, results):
        inp = dep.input_tensor
        print(f"\n  Input: {inp.name}[{','.join(str(s) for s in inp.indices)}]"
              f"  shape={inp.shape}  tiles={inp.tiles}  blocks={inp.blocks}"
              f"  layout={type(inp.data_layout).__name__}")
        print(f"  Input owner:  {inp.owner()}")
        print(f"  Index map:    {dep.index_map}")
        if dep.contracted:
            print(f"  Contracted:   {dep.contracted}")
        if dep.stencil_offsets:
            print(f"  Stencil:      {dep.stencil_offsets}")

        print(f"\n  Mismatch:     {res.mismatch_expr}")
        print(f"  Free indices: {res.free_indices}")
        print(f"  Local:        {res.local_indices}")
        print(f"  → Collective: {res.collective.value.upper()}")
        if res.volume_expr is not None:
            print(f"  → Volume/proc: {res.volume_expr}")
        print(f"  → {res.detail}")


# ============================================================================
# Test cases
# ============================================================================

def test_matmul():
    """C[i,j] = sum_k A[i,k] * B[k,j] on Pr x Pc grid."""
    M, N, K = symbols('M N K', positive=True, integer=True)
    Pr, Pc = symbols('Pr Pc', positive=True, integer=True)
    i, j, k = symbols('i j k', nonneg=True, integer=True)
    bm, bk, bn = floor(M/Pr), floor(K/Pc), floor(N/Pc)
    bk_B = floor(K/Pr)

    A = TensorDist("A", (i, k), (M, K), (bm, bk), (Pr, Pc), Row(M, K))
    B = TensorDist("B", (k, j), (K, N), (bk_B, bn), (Pr, Pc), Row(K, N))
    C = TensorDist("C", (i, j), (M, N), (bm, bn), (Pr, Pc), Row(M, N))

    dep_A = Dependency(A, {i: i, k: k}, contracted=[k])
    dep_B = Dependency(B, {k: k, j: j}, contracted=[k])

    results = derive_communication(C, [dep_A, dep_B])
    print_results("MATMUL: C[i,j] = sum_k A[i,k]*B[k,j]", C, [dep_A, dep_B], results)
    return results


def test_matmul_col():
    """Same matmul but A stored Col-major."""
    M, N, K = symbols('M N K', positive=True, integer=True)
    Pr, Pc = symbols('Pr Pc', positive=True, integer=True)
    i, j, k = symbols('i j k', nonneg=True, integer=True)
    bm, bk, bn = floor(M/Pr), floor(K/Pc), floor(N/Pc)
    bk_B = floor(K/Pr)

    A = TensorDist("A", (i, k), (M, K), (bm, bk), (Pr, Pc), Col(M, K))
    B = TensorDist("B", (k, j), (K, N), (bk_B, bn), (Pr, Pc), Row(K, N))
    C = TensorDist("C", (i, j), (M, N), (bm, bn), (Pr, Pc), Row(M, N))

    dep_A = Dependency(A, {i: i, k: k}, contracted=[k])
    dep_B = Dependency(B, {k: k, j: j}, contracted=[k])

    results = derive_communication(C, [dep_A, dep_B])
    print_results("MATMUL (A=Col): C[i,j] = sum_k A[i,k]*B[k,j]", C, [dep_A, dep_B], results)
    return results


def test_elementwise():
    """C[i,j] = A[i,j] + B[i,j] — same distribution, no communication."""
    M, N = symbols('M N', positive=True, integer=True)
    Pr, Pc = symbols('Pr Pc', positive=True, integer=True)
    i, j = symbols('i j', nonneg=True, integer=True)
    bm, bn = floor(M/Pr), floor(N/Pc)

    A = TensorDist("A", (i, j), (M, N), (bm, bn), (Pr, Pc), Row(M, N))
    B = TensorDist("B", (i, j), (M, N), (bm, bn), (Pr, Pc), Row(M, N))
    C = TensorDist("C", (i, j), (M, N), (bm, bn), (Pr, Pc), Row(M, N))

    dep_A = Dependency(A, {i: i, j: j})
    dep_B = Dependency(B, {i: i, j: j})

    results = derive_communication(C, [dep_A, dep_B])
    print_results("ELEMENTWISE: C[i,j] = A[i,j] + B[i,j]", C, [dep_A, dep_B], results)
    return results


def test_reduction():
    """C[i] = sum_j A[i,j] — reduce along distributed j axis."""
    M, N = symbols('M N', positive=True, integer=True)
    P = symbols('P', positive=True, integer=True)
    i, j = symbols('i j', nonneg=True, integer=True)
    bm, bn = floor(M/P), floor(N/P)

    # A is 2D distributed over P x P grid
    A = TensorDist("A", (i, j), (M, N), (bm, bn), (P, P), Row(M, N))
    # C is 1D: only distributed along i over P procs
    C = TensorDist("C", (i,), (M,), (bm,), (P,), Row(M))

    # C[i] depends on A[i, j] for all j — j is contracted
    dep_A = Dependency(A, {i: i, j: j}, contracted=[j])

    results = derive_communication(C, [dep_A])
    print_results("REDUCTION: C[i] = sum_j A[i,j]", C, [dep_A], results)
    return results


def test_stencil_1d():
    """C[i] = A[i-1] + A[i] + A[i+1] — 1D 3-point stencil."""
    N = symbols('N', positive=True, integer=True)
    P = symbols('P', positive=True, integer=True)
    i = symbols('i', nonneg=True, integer=True)
    bn = floor(N/P)

    A = TensorDist("A", (i,), (N,), (bn,), (P,), Row(N))
    C = TensorDist("C", (i,), (N,), (bn,), (P,), Row(N))

    dep_A = Dependency(A, {i: i}, stencil_offsets={i: [-1, 0, 1]})

    results = derive_communication(C, [dep_A])
    print_results("STENCIL 1D: C[i] = A[i-1] + A[i] + A[i+1]", C, [dep_A], results)
    return results


def test_stencil_2d():
    """C[i,j] = A[i-1,j] + A[i+1,j] + A[i,j-1] + A[i,j+1] + A[i,j]"""
    M, N = symbols('M N', positive=True, integer=True)
    Pr, Pc = symbols('Pr Pc', positive=True, integer=True)
    i, j = symbols('i j', nonneg=True, integer=True)
    bm, bn = floor(M/Pr), floor(N/Pc)

    A = TensorDist("A", (i, j), (M, N), (bm, bn), (Pr, Pc), Row(M, N))
    C = TensorDist("C", (i, j), (M, N), (bm, bn), (Pr, Pc), Row(M, N))

    dep_A = Dependency(A, {i: i, j: j},
                       stencil_offsets={i: [-1, 0, 1], j: [-1, 0, 1]})

    results = derive_communication(C, [dep_A])
    print_results("STENCIL 2D: C[i,j] = 5-point stencil of A", C, [dep_A], results)
    return results


def test_transpose():
    """C[i,j] = A[j,i] — A distributed (i,j), C distributed (i,j), but A accessed transposed."""
    M, N = symbols('M N', positive=True, integer=True)
    Pr, Pc = symbols('Pr Pc', positive=True, integer=True)
    i, j = symbols('i j', nonneg=True, integer=True)
    bm, bn = floor(M/Pr), floor(N/Pc)

    # A has shape (N, M) distributed on Pr x Pc — but we access it as A[j, i]
    A = TensorDist("A", (j, i), (N, M), (floor(N/Pr), floor(M/Pc)), (Pr, Pc), Row(N, M))
    C = TensorDist("C", (i, j), (M, N), (bm, bn), (Pr, Pc), Row(M, N))

    dep_A = Dependency(A, {j: j, i: i})

    results = derive_communication(C, [dep_A])
    print_results("TRANSPOSE: C[i,j] = A[j,i]", C, [dep_A], results)
    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  LEGO Communication Pattern Derivation Prototype")
    print("  All index math via LEGO layouts + simplify_via_mlir")
    print("=" * 70)

    r1 = test_matmul()
    r1c = test_matmul_col()

    # Verify Row vs Col gives same communication
    assert r1[0].collective == r1c[0].collective, "Row vs Col should give same collective for A"
    assert r1[1].collective == r1c[1].collective, "Row vs Col should give same collective for B"
    print("\n  [check] Row vs Col matmul: same collectives ✓")

    r2 = test_elementwise()
    assert all(r.collective == CollectiveType.NONE for r in r2)
    print("\n  [check] Elementwise: no communication ✓")

    r3 = test_reduction()
    assert r3[0].collective in (CollectiveType.ALLGATHER, CollectiveType.ALLREDUCE)
    print(f"\n  [check] Reduction: {r3[0].collective.value} ✓")

    r4 = test_stencil_1d()
    assert r4[0].collective == CollectiveType.HALO_EXCHANGE
    print(f"\n  [check] 1D Stencil: halo exchange ✓")

    r5 = test_stencil_2d()
    assert r5[0].collective == CollectiveType.HALO_EXCHANGE
    print(f"\n  [check] 2D Stencil: halo exchange ✓")

    r6 = test_transpose()
    print(f"\n  [check] Transpose: {r6[0].collective.value}")

    print("\n" + "=" * 70)
    print("  All tests passed.")
    print("=" * 70)
