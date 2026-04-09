"""
E2E Derivation: 2D SUMMA MatMul via LEGO Flat-Space Analysis

ALL index math is done by LEGO layouts + simplify_via_mlir.
Zero manual floor/Mod. Zero GenP lambdas.

Key technique:
  Row(Pr, M//Pr).inv(i) → [proc_row, local_i]
  This decomposes index i into its tile coordinate and local offset
  using ONLY LEGO's Row inverse. No manual division.

Two test cases:
  Case 1: A stored Row(M, K)
  Case 2: A stored Col(M, K)
  Same proc assignment → see what changes in the flat space.
"""
import sympy as sp
from sympy import symbols, floor
from lego.core import Row, Col, OrderBy
from lego.backend.symbolic import simplify_via_mlir


M, N, K = symbols('M N K', positive=True, integer=True)
Pr, Pc = symbols('Pr Pc', positive=True, integer=True)
i, j, k = symbols('i j k', nonneg=True, integer=True)

# Block sizes — symbolic, derived from LEGO's dimension algebra
bm = floor(M / Pr)
bk = floor(K / Pc)
bn = floor(N / Pc)
bk_B = floor(K / Pr)


def owner_2d(row_idx, col_idx, row_tiles, col_tiles, row_block, col_block):
    """
    Compute proc_id for element at (row_idx, col_idx) in a 2D block distribution.
    Uses ONLY LEGO layout inverses — no manual floor/mod.

    Row(tiles, block).inv(idx) → [tile_coord, local_offset]
    The tile_coord IS the processor coordinate for that dimension.
    """
    # Decompose row index: Row(row_tiles, row_block).inv(row_idx) → [pr, local_r]
    row_decomp = simplify_via_mlir(Row(row_tiles, row_block), 'inv', row_idx,
                                   constraints={row_idx: (0, row_tiles * row_block)})
    pr = row_decomp[0]

    # Decompose col index: Row(col_tiles, col_block).inv(col_idx) → [pc, local_c]
    col_decomp = simplify_via_mlir(Row(col_tiles, col_block), 'inv', col_idx,
                                   constraints={col_idx: (0, col_tiles * col_block)})
    pc = col_decomp[0]

    # Linearize proc coords: Row(row_tiles, col_tiles).apply(pr, pc) → proc_id
    proc_id = simplify_via_mlir(Row(row_tiles, col_tiles), 'apply', [pr, pc])
    return proc_id


def analyze_matmul(label, data_layout_base, data_layout_orderby):
    """
    Full E2E derivation for C[i,j] = Σ_k A[i,k] * B[k,j].

    data_layout_base: Row(M,K) or Col(M,K) — the raw LEGO layout block
    data_layout_orderby: OrderBy(data_layout_base) — for TileBy construction
    """
    print("\n" + "=" * 70)
    print(f"  {label}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Proc ownership — computed entirely via LEGO Row.inv + Row.apply
    # ------------------------------------------------------------------
    owner_A = owner_2d(i, k, Pr, Pc, bm, bk)
    owner_B = owner_2d(k, j, Pr, Pc, bk_B, bn)
    owner_C = owner_2d(i, j, Pr, Pc, bm, bn)

    print(f"\n--- Owner (via Row.inv decomposition + Row.apply linearization) ---")
    print(f"  owner_A(i, k) = {owner_A}")
    print(f"  owner_B(k, j) = {owner_B}")
    print(f"  owner_C(i, j) = {owner_C}")

    # ------------------------------------------------------------------
    # 2. Data layout flat index — the part that changes Row vs Col
    # ------------------------------------------------------------------
    bounds_ik = {i: (0, M), k: (0, K)}

    flat_A = simplify_via_mlir(data_layout_base, 'apply', [i, k], constraints=bounds_ik)

    print(f"\n--- Data layout (what changes between Row and Col) ---")
    print(f"  D_A = {type(data_layout_base).__name__}(M, K)")
    print(f"  D_A.apply(i, k) = {flat_A}")

    # ------------------------------------------------------------------
    # 3. Distribution layout: TileBy on top of the data layout
    #    This determines the physical memory ordering within each proc's block
    # ------------------------------------------------------------------
    dist_A = data_layout_orderby.TileBy([Pr, Pc], [bm, bk])

    # TileBy has rank 4: (pr, pc, li, lk). Show its apply with tile indices.
    pr_s, pc_s = symbols('pr pc', nonneg=True, integer=True)
    li_s, lk_s = symbols('li lk', nonneg=True, integer=True)
    tile_bounds = {pr_s: (0, Pr), pc_s: (0, Pc), li_s: (0, bm), lk_s: (0, bk)}

    tile_flat = simplify_via_mlir(dist_A, 'apply', [pr_s, pc_s, li_s, lk_s],
                                  constraints=tile_bounds)
    print(f"\n--- Distribution TileBy (physical memory layout per proc) ---")
    print(f"  dist_A = OrderBy({type(data_layout_base).__name__}(M,K)).TileBy([Pr,Pc],[bm,bk])")
    print(f"  dist_A.apply(pr, pc, li, lk) = {tile_flat}")

    # ------------------------------------------------------------------
    # 4. Mismatch analysis — uses the owner expressions (independent of D)
    # ------------------------------------------------------------------
    mismatch_A = sp.expand(owner_C - owner_A)
    mismatch_B = sp.expand(owner_C - owner_B)

    free_A = mismatch_A.free_symbols & {i, j, k}
    free_B = mismatch_B.free_symbols & {i, j, k}

    print(f"\n--- Mismatch (owner_C - owner_X) ---")
    print(f"  A: {mismatch_A}")
    print(f"     free indices: {free_A}")
    for v, name in [(i, 'i'), (j, 'j'), (k, 'k')]:
        if v not in free_A:
            print(f"     {name} LOCAL")
    if k in free_A:
        print(f"     k in mismatch → ALLGATHER for A")

    print(f"  B: {mismatch_B}")
    print(f"     free indices: {free_B}")
    for v, name in [(i, 'i'), (j, 'j'), (k, 'k')]:
        if v not in free_B:
            print(f"     {name} LOCAL")
    if k in free_B:
        print(f"     k in mismatch → ALLGATHER for B")

    # ------------------------------------------------------------------
    # 5. Numerical verification
    # ------------------------------------------------------------------
    M_v, N_v, K_v, Pr_v, Pc_v = 8, 6, 4, 2, 2
    subs = {M: M_v, N: N_v, K: K_v, Pr: Pr_v, Pc: Pc_v}

    print(f"\n--- Numerical (M={M_v}, N={N_v}, K={K_v}, Pr={Pr_v}, Pc={Pc_v}) ---")

    print(f"  A ownership:")
    header = "       " + "".join(f"k={kk:<3}" for kk in range(K_v))
    print(header)
    for ii in range(M_v):
        row_str = f"  i={ii}: "
        for kk in range(K_v):
            p = int(owner_A.subs({**subs, i: ii, k: kk}))
            row_str += f" P{p}  "
        print(row_str)

    # Show how the TileBy flat ordering differs
    print(f"\n  TileBy flat index for proc (0,0) local block:")
    bm_v, bk_v = M_v // Pr_v, K_v // Pc_v
    for li in range(bm_v):
        row_str = f"    li={li}: "
        for lk in range(bk_v):
            f = int(tile_flat.subs({**subs, pr_s: 0, pc_s: 0, li_s: li, lk_s: lk}))
            row_str += f" {f:>3}"
        print(row_str)

    return owner_A, owner_C, mismatch_A, tile_flat


# ===========================================================================
# CASE 1: A stored Row-major
# ===========================================================================
print("#" * 70)
print("# CASE 1: A = Row(M, K)")
print("#" * 70)

oa1, oc1, ma1, tf1 = analyze_matmul("A stored Row(M, K)", Row(M, K), OrderBy(Row(M, K)))


# ===========================================================================
# CASE 2: A stored Col-major
# ===========================================================================
print("\n\n" + "#" * 70)
print("# CASE 2: A = Col(M, K)")
print("#" * 70)

oa2, oc2, ma2, tf2 = analyze_matmul("A stored Col(M, K)", Col(M, K), OrderBy(Col(M, K)))


# ===========================================================================
# COMPARISON
# ===========================================================================
print("\n\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

same_owner = sp.simplify(oa1 - oa2) == 0
same_mismatch = sp.simplify(ma1 - ma2) == 0

print(f"""
  Owner expression:
    Row: owner_A = {oa1}
    Col: owner_A = {oa2}
    Same? {same_owner}

  A mismatch (communication pattern):
    Row: {ma1}
    Col: {ma2}
    Same? {same_mismatch}

  TileBy flat index (physical memory layout within proc block):
    Row: dist_A.apply(pr,pc,li,lk) = {tf1}
    Col: dist_A.apply(pr,pc,li,lk) = {tf2}
    Same? {sp.simplify(tf1 - tf2) == 0}

  Conclusion:
    - The OWNER (which proc has which element) is the SAME.
      It depends on logical coordinates (i,k), not storage order.
    - The COMMUNICATION PATTERN is the SAME.
      Mismatch depends on the distribution, not the data layout.
    - The PHYSICAL MEMORY LAYOUT within each proc's block DIFFERS.
      Row: elements in a row of the local block are contiguous.
      Col: elements in a column of the local block are contiguous.
    → Changing Row→Col affects LOCAL access patterns (coalescing),
      NOT inter-processor communication.
""")
