// RUN: lego-opt --lego-normalization --lego-to-arith %s | FileCheck %s

// ============================================================================
// Tests for lego-to-arith lowering of GenP layouts.
//
// GenP carries explicit apply and inv regions. The lowering should inline
// these regions at the call site, replacing lego.yield with the actual
// result values.
// ============================================================================

// --- GenP 1D identity: apply(x) = x ---
// CHECK-LABEL: func.func @genp_1d_identity_apply
// CHECK-SAME:  (%[[X:.*]]: index)
// CHECK:       return %[[X]] : index
func.func @genp_1d_identity_apply(%x: index) -> index {
  %c16 = arith.constant 16 : index
  %layout = lego.gen_p [%c16] apply (%a: index) {
    lego.yield %a : index
  } inv (%f: index) {
    lego.yield %f : index
  } : !lego.layout
  %f = lego.apply %layout(%x) : !lego.layout
  return %f : index
}

// --- GenP 1D identity: apply_inverse(flat) = flat ---
// CHECK-LABEL: func.func @genp_1d_identity_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK:       return %[[F]] : index
func.func @genp_1d_identity_inv(%f: index) -> index {
  %c16 = arith.constant 16 : index
  %layout = lego.gen_p [%c16] apply (%a: index) {
    lego.yield %a : index
  } inv (%flat: index) {
    lego.yield %flat : index
  } : !lego.layout
  %r = lego.apply_inverse %layout(%f) : !lego.layout -> index
  return %r : index
}

// --- GenP 2D row-major: apply(i, j) = i*8 + j ---
// CHECK-LABEL: func.func @genp_2d_row_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[I]], %[[C8]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[MUL]], %[[J]] : index
// CHECK:       return %[[RES]] : index
func.func @genp_2d_row_apply(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %layout = lego.gen_p [%c4, %c8] apply (%a: index, %b: index) {
    %c8_inner = arith.constant 8 : index
    %t = arith.muli %a, %c8_inner : index
    %f = arith.addi %t, %b : index
    lego.yield %f : index
  } inv (%flat: index) {
    %c8_inv = arith.constant 8 : index
    %ii = arith.divui %flat, %c8_inv : index
    %jj = arith.remui %flat, %c8_inv : index
    lego.yield %ii, %jj : index, index
  } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// --- GenP 2D row-major: apply_inverse(flat) = (flat/8, flat%8) ---
// CHECK-LABEL: func.func @genp_2d_row_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[I:.*]] = arith.divui %[[F]], %[[C8]] : index
// CHECK:       %[[J:.*]] = arith.remui %[[F]], %[[C8]] : index
// CHECK:       return %[[I]], %[[J]] : index, index
func.func @genp_2d_row_inv(%f: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %layout = lego.gen_p [%c4, %c8] apply (%a: index, %b: index) {
    %c8_inner = arith.constant 8 : index
    %t = arith.muli %a, %c8_inner : index
    %res = arith.addi %t, %b : index
    lego.yield %res : index
  } inv (%flat: index) {
    %c8_inv = arith.constant 8 : index
    %ii = arith.divui %flat, %c8_inv : index
    %jj = arith.remui %flat, %c8_inv : index
    lego.yield %ii, %jj : index, index
  } : !lego.layout
  %i, %j = lego.apply_inverse %layout(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// --- GenP with symbolic dimensions: apply uses captured dim ---
// CHECK-LABEL: func.func @genp_symbolic_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[D:.*]]: index)
// CHECK:       %[[MUL:.*]] = arith.muli %[[I]], %[[D]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[MUL]], %[[J]] : index
// CHECK:       return %[[RES]] : index
func.func @genp_symbolic_apply(%i: index, %j: index, %d: index) -> index {
  %layout = lego.gen_p [%d, %d]
    apply (%a: index, %b: index) {
      %mul = arith.muli %a, %d : index
      %res = arith.addi %mul, %b : index
      lego.yield %res : index
    }
    inv (%flat: index) {
      %ii = arith.divui %flat, %d : index
      %jj = arith.remui %flat, %d : index
      lego.yield %ii, %jj : index, index
    } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// --- GenP with add-based mapping (not row-major): apply(i,j) = i + j ---
// CHECK-LABEL: func.func @genp_add_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK:       %[[SUM:.*]] = arith.addi %[[I]], %[[J]] : index
// CHECK:       return %[[SUM]] : index
func.func @genp_add_apply(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %layout = lego.gen_p [%c4, %c4] apply (%a: index, %b: index) {
    %sum = arith.addi %a, %b : index
    lego.yield %sum : index
  } inv (%flat: index) {
    %z = arith.constant 0 : index
    lego.yield %z, %z : index, index
  } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// --- GenP 2D column-major: apply(i,j) = j*4 + i ---
// CHECK-LABEL: func.func @genp_col_major_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[J]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[MUL]], %[[I]] : index
// CHECK:       return %[[RES]] : index
func.func @genp_col_major_apply(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %layout = lego.gen_p [%c4, %c8] apply (%a: index, %b: index) {
    %c4_inner = arith.constant 4 : index
    %t = arith.muli %b, %c4_inner : index
    %f = arith.addi %t, %a : index
    lego.yield %f : index
  } inv (%flat: index) {
    %c4_inv = arith.constant 4 : index
    %jj = arith.divui %flat, %c4_inv : index
    %ii = arith.remui %flat, %c4_inv : index
    lego.yield %ii, %jj : index, index
  } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// --- GenP with scf.if in apply (antidiag-style): inlines the if ---
// CHECK-LABEL: func.func @genp_with_branch_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-NOT:   lego.gen_p
// CHECK-NOT:   lego.apply
// CHECK-NOT:   lego.yield
// CHECK:       arith.cmpi
// CHECK:       scf.if
// CHECK:       return
func.func @genp_with_branch_apply(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %layout = lego.gen_p [%c4, %c4] apply (%a: index, %b: index) {
    %c2 = arith.constant 2 : index
    %sum = arith.addi %a, %b : index
    %cmp = arith.cmpi slt, %sum, %c2 : index
    %res = scf.if %cmp -> (index) {
      scf.yield %a : index
    } else {
      scf.yield %b : index
    }
    lego.yield %res : index
  } inv (%flat: index) {
    %z = arith.constant 0 : index
    lego.yield %z, %z : index, index
  } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}
