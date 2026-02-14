// RUN: lego-opt %s | FileCheck %s --check-prefix=PARSE
// RUN: lego-opt --lego-desugar --lego-to-arith %s | FileCheck %s --check-prefix=LOWER

// ============================================================================
// GenP — General Permutation with user-defined apply + inv logic
// Tests based on antidiag layout from Python lego.py L490-L531
//
// GenP now carries two regions:
//   apply { ^bb(indices...): ... lego.yield %flat }
//   inv   { ^bb(%flat):     ... lego.yield %i, %j, ... }
// ============================================================================

// --- Simple GenP: apply only (no inv region) ---
// PARSE-LABEL: func.func @genp_add
// PARSE:       lego.gen_p [4, 4] apply
// PARSE:       lego.yield

// LOWER-LABEL: func.func @genp_add
// LOWER-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// LOWER-NOT:   lego.gen_p
// LOWER-NOT:   lego.apply
// LOWER:       %[[SUM:.*]] = arith.addi %[[I]], %[[J]] : index
// LOWER:       return %[[SUM]] : index
func.func @genp_add(%i: index, %j: index) -> index {
  %layout = lego.gen_p [4, 4] apply {
  ^bb0(%a: index, %b: index):
    %sum = arith.addi %a, %b : index
    lego.yield %sum : index
  } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// --- GenP with both apply AND inv: simple divmod ---
// PARSE-LABEL: func.func @genp_apply_inv_divmod
// PARSE:       lego.gen_p [4, 8] apply
// PARSE:       inv

// LOWER-LABEL: func.func @genp_apply_inv_divmod
// LOWER-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// LOWER-DAG:   %[[C8:.*]] = arith.constant 8 : index
// Apply: i*8 + j
// LOWER:       %[[MUL:.*]] = arith.muli %[[I]], %[[C8]] : index
// LOWER:       %[[FLAT:.*]] = arith.addi %[[MUL]], %[[J]] : index
// Inv: flat/8, flat%8
// LOWER:       %[[II:.*]] = arith.divui %[[FLAT]], %[[C8]] : index
// LOWER:       %[[JJ:.*]] = arith.remui %[[FLAT]], %[[C8]] : index
// LOWER:       return %[[II]], %[[JJ]] : index, index
func.func @genp_apply_inv_divmod(%i: index, %j: index) -> (index, index) {
  %layout = lego.gen_p [4, 8] apply {
  ^bb0(%a: index, %b: index):
    %c8 = arith.constant 8 : index
    %t = arith.muli %a, %c8 : index
    %f = arith.addi %t, %b : index
    lego.yield %f : index
  } inv {
  ^bb0(%flat: index):
    %c8 = arith.constant 8 : index
    %ii = arith.divui %flat, %c8 : index
    %jj = arith.remui %flat, %c8 : index
    lego.yield %ii, %jj : index, index
  } : !lego.layout
  // Apply forward then inverse — roundtrip
  %flat = lego.apply %layout(%i, %j) : !lego.layout
  %ri, %rj = lego.apply_inverse %layout(%flat) : !lego.layout -> index, index
  return %ri, %rj : index, index
}

// --- GenP 1D identity apply + inv ---
// PARSE-LABEL: func.func @genp_1d_roundtrip
// PARSE:       lego.gen_p [16] apply
// PARSE:       inv

// LOWER-LABEL: func.func @genp_1d_roundtrip
// LOWER-SAME:  (%[[X:.*]]: index)
// LOWER:       return %[[X]] : index
func.func @genp_1d_roundtrip(%x: index) -> index {
  %layout = lego.gen_p [16] apply {
  ^bb0(%a: index):
    lego.yield %a : index
  } inv {
  ^bb0(%f: index):
    lego.yield %f : index
  } : !lego.layout
  %flat = lego.apply %layout(%x) : !lego.layout
  %back = lego.apply_inverse %layout(%flat) : !lego.layout -> index
  return %back : index
}

// ============================================================================
// antidiag — Forward mapping (Python L490-L506)
//
// For n=4 (4×4 matrix), the anti-diagonal traversal mapping is:
//   (0,0)→0  (0,1)→1  (0,2)→3  (0,3)→6
//   (1,0)→2  (1,1)→4  (1,2)→7  (1,3)→10
//   (2,0)→5  (2,1)→8  (2,2)→11 (2,3)→13
//   (3,0)→9  (3,1)→12 (3,2)→14 (3,3)→15
//
// ad = i + j + 1
// if ad <= n:  flat = ad*(ad-1)/2 + i
// else:        flat = (n*n-n) + i - (2*n-ad)*(2*n-ad-1)/2
// ============================================================================

// PARSE-LABEL: func.func @antidiag_4x4_apply
// PARSE:       lego.gen_p [4, 4] apply
// PARSE:       scf.if
// PARSE:       lego.yield

// LOWER-LABEL: func.func @antidiag_4x4_apply
// LOWER-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// LOWER-NOT:   lego.gen_p
// LOWER-NOT:   lego.apply
// LOWER-NOT:   lego.yield
// LOWER-DAG:   %[[C1:.*]] = arith.constant 1 : index
// LOWER-DAG:   %[[C4:.*]] = arith.constant 4 : index
// LOWER:       %[[IJ:.*]] = arith.addi %[[I]], %[[J]] : index
// LOWER:       %[[AD:.*]] = arith.addi %[[IJ]], %[[C1]] : index
// LOWER:       %[[CMP:.*]] = arith.cmpi sle, %[[AD]], %[[C4]] : index
// LOWER:       %[[RES:.*]] = scf.if %[[CMP]]
// LOWER:       return %[[RES]] : index
func.func @antidiag_4x4_apply(%i: index, %j: index) -> index {
  %layout = lego.gen_p [4, 4] apply {
  ^bb0(%idx_i: index, %idx_j: index):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %n  = arith.constant 4 : index
    %ij = arith.addi %idx_i, %idx_j : index
    %ad = arith.addi %ij, %c1 : index
    %cmp = arith.cmpi sle, %ad, %n : index
    %result = scf.if %cmp -> (index) {
      %ad_m1 = arith.subi %ad, %c1 : index
      %prod = arith.muli %ad, %ad_m1 : index
      %half = arith.divui %prod, %c2 : index
      %flat = arith.addi %half, %idx_i : index
      scf.yield %flat : index
    } else {
      %nn = arith.muli %n, %n : index
      %nn_n = arith.subi %nn, %n : index
      %two_n = arith.muli %c2, %n : index
      %diff = arith.subi %two_n, %ad : index
      %diff_m1 = arith.subi %diff, %c1 : index
      %prod2 = arith.muli %diff, %diff_m1 : index
      %half2 = arith.divui %prod2, %c2 : index
      %t1 = arith.addi %nn_n, %idx_i : index
      %flat = arith.subi %t1, %half2 : index
      scf.yield %flat : index
    }
    lego.yield %result : index
  } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// ============================================================================
// antidiag with BOTH apply and inv regions (Python L490-L531)
//
// The inv region implements:
//   S1 = n*(n+1)/2
//   if x0 < S1:
//     k = floor((sqrt(8*x0+1)-1)/2) + 1
//     i = x0 - k*(k-1)/2;  j = (k-1) - i
//   else:
//     m2 = x0 - S1
//     d = floor((2*n-1 - sqrt((2*n-1)^2 - 8*m2))/2) + 1
//     prev = (d-1)*n - (d-1)*d/2
//     i = d + (m2 - prev);  j = (n+d-1) - i
// ============================================================================

// PARSE-LABEL: func.func @antidiag_4x4_full
// PARSE:       lego.gen_p [4, 4] apply
// PARSE:       inv
// PARSE:       math.sqrt
// PARSE:       math.floor

// LOWER-LABEL: func.func @antidiag_4x4_full
// LOWER-SAME:  (%[[X0:.*]]: index)
// LOWER-NOT:   lego.gen_p
// LOWER-NOT:   lego.apply_inverse
// LOWER-NOT:   lego.yield
// LOWER:       arith.cmpi slt
// LOWER:       scf.if
// LOWER:       math.sqrt
// LOWER:       math.floor
// LOWER:       return
func.func @antidiag_4x4_full(%x0: index) -> (index, index) {
  %layout = lego.gen_p [4, 4] apply {
  ^bb0(%idx_i: index, %idx_j: index):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %n  = arith.constant 4 : index
    %ij = arith.addi %idx_i, %idx_j : index
    %ad = arith.addi %ij, %c1 : index
    %cmp = arith.cmpi sle, %ad, %n : index
    %result = scf.if %cmp -> (index) {
      %ad_m1 = arith.subi %ad, %c1 : index
      %prod = arith.muli %ad, %ad_m1 : index
      %half = arith.divui %prod, %c2 : index
      %flat = arith.addi %half, %idx_i : index
      scf.yield %flat : index
    } else {
      %nn = arith.muli %n, %n : index
      %nn_n = arith.subi %nn, %n : index
      %two_n = arith.muli %c2, %n : index
      %diff = arith.subi %two_n, %ad : index
      %diff_m1 = arith.subi %diff, %c1 : index
      %prod2 = arith.muli %diff, %diff_m1 : index
      %half2 = arith.divui %prod2, %c2 : index
      %t1 = arith.addi %nn_n, %idx_i : index
      %flat = arith.subi %t1, %half2 : index
      scf.yield %flat : index
    }
    lego.yield %result : index
  } inv {
  ^bb0(%flat: index):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %n  = arith.constant 4 : index
    %one_f = arith.constant 1.0 : f64
    %two_f = arith.constant 2.0 : f64
    %eight_f = arith.constant 8.0 : f64

    // S1 = n*(n+1)/2 = 10
    %n1 = arith.addi %n, %c1 : index
    %nn1 = arith.muli %n, %n1 : index
    %S1 = arith.divui %nn1, %c2 : index

    %cmp = arith.cmpi slt, %flat, %S1 : index
    %i_result, %j_result = scf.if %cmp -> (index, index) {
      // Case 1: k = floor((sqrt(8*x0+1)-1)/2) + 1
      %x8 = arith.muli %flat, %c8 : index
      %x8p1 = arith.addi %x8, %c1 : index
      %x8p1_i64 = arith.index_castui %x8p1 : index to i64
      %x8p1_f = arith.uitofp %x8p1_i64 : i64 to f64
      %sqrt_val = math.sqrt %x8p1_f : f64
      %sub1 = arith.subf %sqrt_val, %one_f : f64
      %div2 = arith.divf %sub1, %two_f : f64
      %floored = math.floor %div2 : f64
      %k_f = arith.addf %floored, %one_f : f64
      %k_i64 = arith.fptoui %k_f : f64 to i64
      %k = arith.index_cast %k_i64 : i64 to index
      // i = x0 - k*(k-1)/2
      %k_m1 = arith.subi %k, %c1 : index
      %kk1 = arith.muli %k, %k_m1 : index
      %tri = arith.divui %kk1, %c2 : index
      %ii = arith.subi %flat, %tri : index
      // j = (k-1) - i
      %jj = arith.subi %k_m1, %ii : index
      scf.yield %ii, %jj : index, index
    } else {
      // Case 2: m2 = x0 - S1
      %m2 = arith.subi %flat, %S1 : index
      %two_n = arith.muli %c2, %n : index
      %two_n_m1 = arith.subi %two_n, %c1 : index
      %tnm1_i64 = arith.index_castui %two_n_m1 : index to i64
      %tnm1_f = arith.uitofp %tnm1_i64 : i64 to f64
      // d = floor((2*n-1 - sqrt((2*n-1)^2 - 8*m2))/2) + 1
      %sq = arith.muli %two_n_m1, %two_n_m1 : index
      %m2x8 = arith.muli %m2, %c8 : index
      %disc = arith.subi %sq, %m2x8 : index
      %disc_i64 = arith.index_castui %disc : index to i64
      %disc_f = arith.uitofp %disc_i64 : i64 to f64
      %sqrt_disc = math.sqrt %disc_f : f64
      %num = arith.subf %tnm1_f, %sqrt_disc : f64
      %num_h = arith.divf %num, %two_f : f64
      %d_floor = math.floor %num_h : f64
      %d_plus = arith.addf %d_floor, %one_f : f64
      %d_i64 = arith.fptoui %d_plus : f64 to i64
      %d = arith.index_cast %d_i64 : i64 to index
      // prev = (d-1)*n - (d-1)*d/2
      %d_m1 = arith.subi %d, %c1 : index
      %p1 = arith.muli %d_m1, %n : index
      %p2 = arith.muli %d_m1, %d : index
      %p3 = arith.divui %p2, %c2 : index
      %prev = arith.subi %p1, %p3 : index
      // i = d + (m2 - prev)
      %mdiff = arith.subi %m2, %prev : index
      %ii = arith.addi %d, %mdiff : index
      // j = (n + d - 1) - i
      %nd = arith.addi %n, %d : index
      %ndm1 = arith.subi %nd, %c1 : index
      %jj = arith.subi %ndm1, %ii : index
      scf.yield %ii, %jj : index, index
    }
    lego.yield %i_result, %j_result : index, index
  } : !lego.layout
  %i, %j = lego.apply_inverse %layout(%x0) : !lego.layout -> index, index
  return %i, %j : index, index
}

// ============================================================================
// GenP composed with other layout ops
// ============================================================================

// --- GenP inside OrderBy ---
// PARSE-LABEL: func.func @genp_in_orderby
// PARSE:       lego.gen_p [4] apply
// PARSE:       lego.order_by

// LOWER-LABEL: func.func @genp_in_orderby
// LOWER-NOT:   lego.gen_p
// LOWER-NOT:   lego.order_by
// LOWER-NOT:   lego.apply
// LOWER:       return
func.func @genp_in_orderby(%i: index, %j: index) -> index {
  %double = lego.gen_p [4] apply {
  ^bb0(%a: index):
    %c2 = arith.constant 2 : index
    %res = arith.muli %a, %c2 : index
    lego.yield %res : index
  } : !lego.layout
  %id = lego.reg_p perm [0] dims [8] : !lego.layout
  %ob = lego.order_by(%double, %id) : !lego.layout
  %f = lego.apply %ob(%i, %j) : !lego.layout
  return %f : index
}

// --- GenP inside GroupBy ---
// PARSE-LABEL: func.func @genp_in_groupby
// PARSE:       lego.gen_p [6] apply
// PARSE:       lego.group_by [6]

// LOWER-LABEL: func.func @genp_in_groupby
// LOWER-NOT:   lego.gen_p
// LOWER-NOT:   lego.group_by
// LOWER-NOT:   lego.apply
// LOWER:       return
func.func @genp_in_groupby(%i: index) -> index {
  %rev = lego.gen_p [6] apply {
  ^bb0(%a: index):
    %c5 = arith.constant 5 : index
    %res = arith.subi %c5, %a : index
    lego.yield %res : index
  } : !lego.layout
  %gb = lego.group_by [6](%rev) : !lego.layout
  %f = lego.apply %gb(%i) : !lego.layout
  return %f : index
}

// ============================================================================
// All ops apply + inverse roundtrip tests
// Verifies every op supports both lego.apply and lego.apply_inverse
// ============================================================================

// --- Row apply + inv ---
// LOWER-LABEL: func.func @row_roundtrip
// LOWER-NOT:   lego.row
// LOWER-NOT:   lego.apply
// LOWER:       return
func.func @row_roundtrip(%i: index, %j: index) -> (index, index) {
  %r = lego.row [4, 8] : !lego.layout
  %flat = lego.apply %r(%i, %j) : !lego.layout
  %ri, %rj = lego.apply_inverse %r(%flat) : !lego.layout -> index, index
  return %ri, %rj : index, index
}

// --- Col apply + inv ---
// LOWER-LABEL: func.func @col_roundtrip
// LOWER-NOT:   lego.col
// LOWER-NOT:   lego.apply
// LOWER:       return
func.func @col_roundtrip(%i: index, %j: index) -> (index, index) {
  %c = lego.col [4, 8] : !lego.layout
  %flat = lego.apply %c(%i, %j) : !lego.layout
  %ri, %rj = lego.apply_inverse %c(%flat) : !lego.layout -> index, index
  return %ri, %rj : index, index
}

// --- RegP apply + inv ---
// LOWER-LABEL: func.func @regp_roundtrip
// LOWER-NOT:   lego.reg_p
// LOWER-NOT:   lego.apply
// LOWER:       return
func.func @regp_roundtrip(%i: index, %j: index) -> (index, index) {
  %rp = lego.reg_p perm [1, 0] dims [4, 8] : !lego.layout
  %flat = lego.apply %rp(%i, %j) : !lego.layout
  %ri, %rj = lego.apply_inverse %rp(%flat) : !lego.layout -> index, index
  return %ri, %rj : index, index
}

// --- OrderBy apply + inv ---
// LOWER-LABEL: func.func @orderby_roundtrip
// LOWER-NOT:   lego.order_by
// LOWER-NOT:   lego.apply
// LOWER:       return
func.func @orderby_roundtrip(%i: index, %j: index) -> (index, index) {
  %p1 = lego.reg_p perm [0] dims [4] : !lego.layout
  %p2 = lego.reg_p perm [0] dims [8] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %flat = lego.apply %ob(%i, %j) : !lego.layout
  %ri, %rj = lego.apply_inverse %ob(%flat) : !lego.layout -> index, index
  return %ri, %rj : index, index
}

// --- GroupBy apply + inv ---
// LOWER-LABEL: func.func @groupby_roundtrip
// LOWER-NOT:   lego.group_by
// LOWER-NOT:   lego.apply
// LOWER:       return
func.func @groupby_roundtrip(%i: index, %j: index) -> (index, index) {
  %rp = lego.reg_p perm [1, 0] dims [4, 8] : !lego.layout
  %ob = lego.order_by(%rp) : !lego.layout
  %gb = lego.group_by [4, 8](%ob) : !lego.layout
  %flat = lego.apply %gb(%i, %j) : !lego.layout
  %ri, %rj = lego.apply_inverse %gb(%flat) : !lego.layout -> index, index
  return %ri, %rj : index, index
}

// --- TileBy apply + inv ---
// LOWER-LABEL: func.func @tileby_roundtrip
// LOWER-NOT:   lego.tile_by
// LOWER-NOT:   lego.apply
// LOWER:       return
func.func @tileby_roundtrip(%i_t: index, %j_t: index, %i_b: index, %j_b: index) -> (index, index, index, index) {
  // TileBy(Row(8, 32), [[2, 4], [4, 8]])
  // d=2. q=2. Flattened dims [2, 4, 4, 8].
  %inner = lego.row [8, 32] : !lego.layout
  %ob = lego.order_by(%inner) : !lego.layout
  %tb = lego.tile_by %ob tile_dims [[2, 4], [4, 8]] : !lego.layout
  
  %flat = lego.apply %tb(%i_t, %j_t, %i_b, %j_b) : !lego.layout
  %ri_t, %rj_t, %ri_b, %rj_b = lego.apply_inverse %tb(%flat) : !lego.layout -> index, index, index, index
  return %ri_t, %rj_t, %ri_b, %rj_b : index, index, index, index
}

// --- GenP apply + inv ---
// LOWER-LABEL: func.func @genp_roundtrip
// LOWER-NOT:   lego.gen_p
// LOWER-NOT:   lego.apply
// LOWER:       return
func.func @genp_roundtrip(%i: index, %j: index) -> (index, index) {
  %layout = lego.gen_p [3, 5] apply {
  ^bb0(%a: index, %b: index):
    %c5 = arith.constant 5 : index
    %t = arith.muli %a, %c5 : index
    %f = arith.addi %t, %b : index
    lego.yield %f : index
  } inv {
  ^bb0(%flat: index):
    %c5 = arith.constant 5 : index
    %ii = arith.divui %flat, %c5 : index
    %jj = arith.remui %flat, %c5 : index
    lego.yield %ii, %jj : index, index
  } : !lego.layout
  %flat = lego.apply %layout(%i, %j) : !lego.layout
  %ri, %rj = lego.apply_inverse %layout(%flat) : !lego.layout -> index, index
  return %ri, %rj : index, index
}

// ============================================================================
// Benchmark-inspired: matmul layout TileBy with Row inner (4D)
// TileBy(Row(32, 16), [[4, 4], [8, 4]])
// d=2. q=2. Flattened dims [4, 4, 8, 4].
// ============================================================================

// LOWER-LABEL: func.func @matmul_tileby_apply
// LOWER-NOT:   lego.tile_by
// LOWER-NOT:   lego.apply
// LOWER:       return
func.func @matmul_tileby_apply(%i_t: index, %j_t: index, %i_b: index, %j_b: index) -> index {
  // Matmul M=32, K=16. BM=8, BK=4.
  // Row(32, 16)
  // TileB([32/8, 16/4], [8, 4]) = [4, 4], [8, 4].
  %inner = lego.row [32, 16] : !lego.layout
  %ob = lego.order_by(%inner) : !lego.layout
  %tb = lego.tile_by %ob tile_dims [[4, 4], [8, 4]] : !lego.layout
  %flat = lego.apply %tb(%i_t, %j_t, %i_b, %j_b) : !lego.layout
  return %flat : index
}

// Inverse — L.inv(pid) pattern from matmul_sympy.py
// LOWER-LABEL: func.func @matmul_tileby_inv
// LOWER-NOT:   lego.tile_by
// LOWER-NOT:   lego.apply_inverse
// LOWER:       return
func.func @matmul_tileby_inv(%pid: index) -> (index, index, index, index) {
  %inner = lego.row [32, 16] : !lego.layout
  %ob = lego.order_by(%inner) : !lego.layout
  %tb = lego.tile_by %ob tile_dims [[4, 4], [8, 4]] : !lego.layout
  %i_t, %j_t, %i_b, %j_b = lego.apply_inverse %tb(%pid) : !lego.layout -> index, index, index, index
  return %i_t, %j_t, %i_b, %j_b : index, index, index, index
}

// ============================================================================
// Benchmark-inspired: OrderBy with TileBy (bricks-style)
// OrderBy(Row(6,6,6), Row(8,8,8)).TileBy([[6,6,6], [8,8,8]])
// d=3. q=2. Flattened [6, 6, 6, 8, 8, 8].
// ============================================================================

// LOWER-LABEL: func.func @bricks_tileby
// LOWER-NOT:   lego.order_by
// LOWER-NOT:   lego.tile_by
// LOWER:       return
func.func @bricks_tileby(%bx: index, %by: index, %bz: index, 
                         %tx: index, %ty: index, %tz: index) -> index {
  // N=48, B=8.
  // OrderBy(Row(6,6,6), Row(8,8,8)).
  // TileBy([6,6,6], [8,8,8]).
  // tile_dims = [6,6,6, 8,8,8]. d=3.
  %r1 = lego.row [6, 6, 6] : !lego.layout
  %r2 = lego.row [8, 8, 8] : !lego.layout
  %ob = lego.order_by(%r1, %r2) : !lego.layout
  %tb = lego.tile_by %ob tile_dims [[6, 6, 6], [8, 8, 8]] : !lego.layout
  %flat = lego.apply %tb(%bx, %by, %bz, %tx, %ty, %tz) : !lego.layout
  return %flat : index
}
