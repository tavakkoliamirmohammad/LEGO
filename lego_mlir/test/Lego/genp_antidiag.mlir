// RUN: lego-opt %s | FileCheck %s --check-prefix=PARSE
// RUN: lego-opt --lego-to-arith %s | FileCheck %s --check-prefix=LOWER

// ============================================================================
// GenP — General Permutation with user-defined logic
// Tests for the `antidiag` layout from Python lego.py L490-L531
//
// Python:
//   antidiag(n, (i, j)):
//     ad = i + j + 1
//     if ad <= n:  flat = ad*(ad-1)//2 + i
//     else:        flat = (n*n - n) + i - (2*n - ad)*(2*n - ad - 1)//2
//
//   antidiag_inv(n, x0):
//     S1 = n*(n+1)//2
//     if x0 < S1:
//       k = floor((sqrt(8*x0+1)-1)/2) + 1
//       i = x0 - k*(k-1)//2
//       j = (k-1) - i
//     else:
//       m2 = x0 - S1
//       d = floor((2*n-1-sqrt((2*n-1)^2-8*m2))/2) + 1
//       prev = (d-1)*n - (d-1)*d//2
//       i = d + (m2 - prev)
//       j = (n+d-1) - i
// ============================================================================

// --- Simple GenP: addition ---
// PARSE-LABEL: func.func @genp_add
// PARSE:       lego.gen_p [4, 4]
// PARSE:       lego.yield

// LOWER-LABEL: func.func @genp_add
// LOWER-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// LOWER-NOT:   lego.gen_p
// LOWER-NOT:   lego.apply
// LOWER:       %[[SUM:.*]] = arith.addi %[[I]], %[[J]] : index
// LOWER:       return %[[SUM]] : index
func.func @genp_add(%i: index, %j: index) -> index {
  %layout = lego.gen_p [4, 4] {
  ^bb0(%a: index, %b: index):
    %sum = arith.addi %a, %b : index
    lego.yield %sum : index
  } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// --- Simple GenP: multiplication ---
// PARSE-LABEL: func.func @genp_mul
// PARSE:       lego.gen_p [8, 8]
// PARSE:       arith.muli
// PARSE:       lego.yield

// LOWER-LABEL: func.func @genp_mul
// LOWER-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// LOWER:       %[[MUL:.*]] = arith.muli %[[I]], %[[J]] : index
// LOWER:       return %[[MUL]] : index
func.func @genp_mul(%i: index, %j: index) -> index {
  %layout = lego.gen_p [8, 8] {
  ^bb0(%a: index, %b: index):
    %prod = arith.muli %a, %b : index
    lego.yield %prod : index
  } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// --- GenP with constants: 3*i + 2*j + 5 ---
// PARSE-LABEL: func.func @genp_affine
// PARSE:       lego.gen_p [10, 10]

// LOWER-LABEL: func.func @genp_affine
// LOWER-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// LOWER-DAG:   %[[C2:.*]] = arith.constant 2 : index
// LOWER-DAG:   %[[C3:.*]] = arith.constant 3 : index
// LOWER-DAG:   %[[C5:.*]] = arith.constant 5 : index
// LOWER:       %[[T1:.*]] = arith.muli %[[I]], %[[C3]] : index
// LOWER:       %[[T2:.*]] = arith.muli %[[J]], %[[C2]] : index
// LOWER:       %[[T3:.*]] = arith.addi %[[T1]], %[[T2]] : index
// LOWER:       %[[RES:.*]] = arith.addi %[[T3]], %[[C5]] : index
// LOWER:       return %[[RES]] : index
func.func @genp_affine(%i: index, %j: index) -> index {
  %layout = lego.gen_p [10, 10] {
  ^bb0(%a: index, %b: index):
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %t1 = arith.muli %a, %c3 : index
    %t2 = arith.muli %b, %c2 : index
    %t3 = arith.addi %t1, %t2 : index
    %res = arith.addi %t3, %c5 : index
    lego.yield %res : index
  } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// --- GenP 1D: identity ---
// PARSE-LABEL: func.func @genp_1d_identity
// PARSE:       lego.gen_p [16]

// LOWER-LABEL: func.func @genp_1d_identity
// LOWER-SAME:  (%[[I:.*]]: index)
// LOWER:       return %[[I]] : index
func.func @genp_1d_identity(%i: index) -> index {
  %layout = lego.gen_p [16] {
  ^bb0(%a: index):
    lego.yield %a : index
  } : !lego.layout
  %f = lego.apply %layout(%i) : !lego.layout
  return %f : index
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
//
// After lowering, constants fold: n=4, n*n=16, n*n-n=12, 2*n=8
// ============================================================================

// PARSE-LABEL: func.func @antidiag_4x4
// PARSE:       lego.gen_p [4, 4]
// PARSE:       arith.addi
// PARSE:       arith.cmpi sle
// PARSE:       scf.if
// PARSE:       lego.yield

// LOWER-LABEL: func.func @antidiag_4x4
// LOWER-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// LOWER-NOT:   lego.gen_p
// LOWER-NOT:   lego.apply
// LOWER-NOT:   lego.yield
// LOWER-DAG:   %[[C1:.*]] = arith.constant 1 : index
// LOWER-DAG:   %[[C2:.*]] = arith.constant 2 : index
// LOWER-DAG:   %[[C4:.*]] = arith.constant 4 : index
// ad = i + j + 1
// LOWER:       %[[IJ:.*]] = arith.addi %[[I]], %[[J]] : index
// LOWER:       %[[AD:.*]] = arith.addi %[[IJ]], %[[C1]] : index
// LOWER:       %[[CMP:.*]] = arith.cmpi sle, %[[AD]], %[[C4]] : index
// LOWER:       %[[RES:.*]] = scf.if %[[CMP]]
// Case 1: ad*(ad-1)/2 + i
// LOWER:         arith.muli %[[AD]]
// LOWER:         arith.divui
// LOWER:         arith.addi
// LOWER:         scf.yield
// Case 2: 12 + i - (8-ad)*(8-ad-1)/2
// LOWER:       } else {
// LOWER:         arith.subi
// LOWER:         arith.muli
// LOWER:         arith.divui
// LOWER:         arith.subi
// LOWER:         scf.yield
// LOWER:       }
// LOWER:       return %[[RES]] : index
func.func @antidiag_4x4(%i: index, %j: index) -> index {
  %layout = lego.gen_p [4, 4] {
  ^bb0(%idx_i: index, %idx_j: index):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %n  = arith.constant 4 : index

    // ad = i + j + 1
    %ij = arith.addi %idx_i, %idx_j : index
    %ad = arith.addi %ij, %c1 : index

    // ad <= n ?
    %cmp = arith.cmpi sle, %ad, %n : index

    %result = scf.if %cmp -> (index) {
      // Case 1: ad*(ad-1)/2 + i
      %ad_m1 = arith.subi %ad, %c1 : index
      %prod = arith.muli %ad, %ad_m1 : index
      %half = arith.divui %prod, %c2 : index
      %flat = arith.addi %half, %idx_i : index
      scf.yield %flat : index
    } else {
      // Case 2: (n*n - n) + i - (2*n - ad)*(2*n - ad - 1)/2
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

// --- antidiag parametric n=8 (8×8 matrix) ---
// Same formula, different constants: n=8, n*n-n=56, 2*n=16
// PARSE-LABEL: func.func @antidiag_8x8
// PARSE:       lego.gen_p [8, 8]

// LOWER-LABEL: func.func @antidiag_8x8
// LOWER-DAG:   %[[C8:.*]] = arith.constant 8 : index
// LOWER:       arith.cmpi sle
// LOWER:       scf.if
// LOWER:       return
func.func @antidiag_8x8(%i: index, %j: index) -> index {
  %layout = lego.gen_p [8, 8] {
  ^bb0(%idx_i: index, %idx_j: index):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %n  = arith.constant 8 : index
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
// antidiag_inv — Inverse mapping (Python L509-L531)
//
// antidiag_inv(n, x0):
//   S1 = n*(n+1)/2
//   if x0 < S1:
//     k = floor((sqrt(8*x0+1)-1)/2) + 1
//     i = x0 - k*(k-1)/2
//     j = (k-1) - i
//   else:
//     m2 = x0 - S1
//     d = floor((2*n-1 - sqrt((2*n-1)^2 - 8*m2))/2) + 1
//     prev = (d-1)*n - (d-1)*d/2
//     i = d + (m2 - prev)
//     j = (n+d-1) - i
//
// This uses math.sqrt and math.floor, so we use index->f64 conversion
// and math.sqrt/math.floor ops.
// ============================================================================

// PARSE-LABEL: func.func @antidiag_inv_4x4
// PARSE:       lego.gen_p [16]
// PARSE:       arith.cmpi slt
// PARSE:       scf.if
// PARSE:       math.sqrt
// PARSE:       math.floor
// PARSE:       lego.yield

// LOWER-LABEL: func.func @antidiag_inv_4x4
// LOWER-NOT:   lego.gen_p
// LOWER-NOT:   lego.apply
// LOWER-NOT:   lego.yield
// LOWER:       arith.cmpi slt
// LOWER:       scf.if
// LOWER:       math.sqrt
// LOWER:       math.floor
// LOWER:       return
func.func @antidiag_inv_4x4(%x0: index) -> (index, index) {
  %layout = lego.gen_p [16] {
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

    // x0 < S1 ?
    %cmp = arith.cmpi slt, %flat, %S1 : index
    %i_result, %j_result = scf.if %cmp -> (index, index) {
      // Case 1: Within first triangle
      // k = floor((sqrt(8*x0+1)-1)/2) + 1
      %x8 = arith.muli %flat, %c8 : index
      %x8p1 = arith.addi %x8, %c1 : index
      %x8p1_f = arith.index_castui %x8p1 : index to i64
      %x8p1_fp = arith.uitofp %x8p1_f : i64 to f64
      %sqrt_val = math.sqrt %x8p1_fp : f64
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
      // Case 2: Beyond first triangle
      // m2 = x0 - S1
      %m2 = arith.subi %flat, %S1 : index

      // 2*n - 1
      %two_n = arith.muli %c2, %n : index
      %two_n_m1 = arith.subi %two_n, %c1 : index
      %two_n_m1_f_i = arith.index_castui %two_n_m1 : index to i64
      %two_n_m1_f = arith.uitofp %two_n_m1_f_i : i64 to f64

      // d = floor((2*n-1 - sqrt((2*n-1)^2 - 8*m2))/2) + 1
      %sq = arith.muli %two_n_m1, %two_n_m1 : index
      %m2x8 = arith.muli %m2, %c8 : index
      %disc = arith.subi %sq, %m2x8 : index
      %disc_i = arith.index_castui %disc : index to i64
      %disc_f = arith.uitofp %disc_i : i64 to f64
      %sqrt_disc = math.sqrt %disc_f : f64
      %num = arith.subf %two_n_m1_f, %sqrt_disc : f64
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
    // Pack i,j into flat via row-major: i*n + j
    %n_val = arith.constant 4 : index
    %packed = arith.muli %i_result, %n_val : index
    %out = arith.addi %packed, %j_result : index
    lego.yield %out : index
  } : !lego.layout

  // The GenP maps flat_index -> packed(i,j) = i*4 + j
  // So we apply the layout then unflatten
  %packed = lego.apply %layout(%x0) : !lego.layout
  %row = lego.row [4, 4] : !lego.layout
  %i, %j = lego.apply_inverse %row(%packed) : !lego.layout -> index, index
  return %i, %j : index, index
}

// ============================================================================
// GenP inside OrderBy — composing GenP with other layout ops
// ============================================================================

// PARSE-LABEL: func.func @genp_in_orderby
// PARSE:       lego.gen_p [4]
// PARSE:       lego.order_by

// LOWER-LABEL: func.func @genp_in_orderby
// LOWER-NOT:   lego.gen_p
// LOWER-NOT:   lego.order_by
// LOWER-NOT:   lego.apply
// LOWER:       return
func.func @genp_in_orderby(%i: index, %j: index) -> index {
  // GenP block that doubles its input
  %double = lego.gen_p [4] {
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

// ============================================================================
// GenP inside GroupBy — composing GenP with GroupBy
// ============================================================================

// PARSE-LABEL: func.func @genp_in_groupby
// PARSE:       lego.gen_p [6]
// PARSE:       lego.group_by [6]

// LOWER-LABEL: func.func @genp_in_groupby
// LOWER-NOT:   lego.gen_p
// LOWER-NOT:   lego.group_by
// LOWER-NOT:   lego.apply
// LOWER:       return
func.func @genp_in_groupby(%i: index) -> index {
  // GenP: f(x) = 5 - x  (reversal on 6 elements)
  %rev = lego.gen_p [6] {
  ^bb0(%a: index):
    %c5 = arith.constant 5 : index
    %res = arith.subi %c5, %a : index
    lego.yield %res : index
  } : !lego.layout
  %gb = lego.group_by [6](%rev) : !lego.layout
  %f = lego.apply %gb(%i) : !lego.layout
  return %f : index
}
