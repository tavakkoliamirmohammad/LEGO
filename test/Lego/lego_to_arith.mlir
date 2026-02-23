// RUN: lego-opt --lego-normalization --lego-to-arith %s | FileCheck %s

// ============================================================================
// Row
// ============================================================================

// CHECK-LABEL: func.func @row_2d_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[I]], %[[C8]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[J]], %[[MUL]] : index
// CHECK:       return %[[RES]] : index
func.func @row_2d_apply(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c4, %c8] : !lego.layout
  %f = lego.apply %r(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @row_3d_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C12:.*]] = arith.constant 12 : index
// CHECK:       %[[T2:.*]] = arith.muli %[[J]], %[[C4]] : index
// CHECK:       %[[S1:.*]] = arith.addi %[[K]], %[[T2]] : index
// CHECK:       %[[T1:.*]] = arith.muli %[[I]], %[[C12]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[S1]], %[[T1]] : index
// CHECK:       return %[[RES]] : index
func.func @row_3d_apply(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %r = lego.row [%c2, %c3, %c4] : !lego.layout
  %f = lego.apply %r(%i, %j, %k) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @row_4d_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index, %[[L:.*]]: index)
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C20:.*]] = arith.constant 20 : index
// CHECK-DAG:   %[[C60:.*]] = arith.constant 60 : index
// CHECK:       %[[T4:.*]] = arith.muli %[[K]], %[[C5]] : index
// CHECK:       %[[S1:.*]] = arith.addi %[[L]], %[[T4]] : index
// CHECK:       %[[T2:.*]] = arith.muli %[[J]], %[[C20]] : index
// CHECK:       %[[S2:.*]] = arith.addi %[[S1]], %[[T2]] : index
// CHECK:       %[[T1:.*]] = arith.muli %[[I]], %[[C60]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[S2]], %[[T1]] : index
// CHECK:       return %[[RES]] : index
func.func @row_4d_apply(%i: index, %j: index, %k: index, %l: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %r = lego.row [%c2, %c3, %c4, %c5] : !lego.layout
  %f = lego.apply %r(%i, %j, %k, %l) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @row_2d_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:   %[[I:.*]] = arith.divui %[[F]], %[[C8]] : index
// CHECK-DAG:   %[[J:.*]] = arith.remui %[[F]], %[[C8]] : index
// CHECK:       return %[[I]], %[[J]] : index, index
func.func @row_2d_inv(%f: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c4, %c8] : !lego.layout
  %i, %j = lego.apply_inverse %r(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// CHECK-LABEL: func.func @row_3d_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C12:.*]] = arith.constant 12 : index
// CHECK:       %[[I:.*]] = arith.divui %[[F]], %[[C12]] : index
// CHECK:       %[[R1:.*]] = arith.remui %[[F]], %[[C12]] : index
// CHECK:       %[[J:.*]] = arith.divui %[[R1]], %[[C4]] : index
// CHECK:       %[[K:.*]] = arith.remui %[[R1]], %[[C4]] : index
// CHECK:       return %[[I]], %[[J]], %[[K]] : index, index, index
func.func @row_3d_inv(%f: index) -> (index, index, index) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %r = lego.row [%c2, %c3, %c4] : !lego.layout
  %i, %j, %k = lego.apply_inverse %r(%f) : !lego.layout -> index, index, index
  return %i, %j, %k : index, index, index
}

// ============================================================================
// Col
// ============================================================================

// CHECK-LABEL: func.func @col_2d_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[MUL:.*]] = arith.muli %[[J]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[I]], %[[MUL]] : index
// CHECK:       return %[[RES]] : index
func.func @col_2d_apply(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c = lego.col [%c4, %c8] : !lego.layout
  %f = lego.apply %c(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @col_3d_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[T1:.*]] = arith.muli %[[K]], %[[C6]] : index
// CHECK-DAG:   %[[T2:.*]] = arith.muli %[[J]], %[[C2]] : index
// CHECK-DAG:   %[[T3:.*]] = arith.addi %[[I]], %[[T2]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T3]], %[[T1]] : index
// CHECK:       return %[[RES]] : index
func.func @col_3d_apply(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c = lego.col [%c2, %c3, %c4] : !lego.layout
  %f = lego.apply %c(%i, %j, %k) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @col_2d_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[D:.*]] = arith.divui %[[F]], %[[C4]] : index
// CHECK:       %[[R:.*]] = arith.remui %[[F]], %[[C4]] : index
// CHECK:       return %[[R]], %[[D]] : index, index
func.func @col_2d_inv(%f: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c = lego.col [%c4, %c8] : !lego.layout
  %i, %j = lego.apply_inverse %c(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// ============================================================================
// RegP
// ============================================================================

// CHECK-LABEL: func.func @regp_identity
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[I]], %[[C8]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[J]], %[[MUL]] : index
// CHECK:       return %[[RES]] : index
func.func @regp_identity(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rp = lego.reg_p perm [0, 1] dims [%c4, %c8] : !lego.layout
  %f = lego.apply %rp(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @regp_transpose
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[J]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[I]], %[[MUL]] : index
// CHECK:       return %[[RES]] : index
func.func @regp_transpose(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rp = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  %f = lego.apply %rp(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @regp_transpose_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[D:.*]] = arith.divui %[[F]], %[[C4]] : index
// CHECK:       %[[R:.*]] = arith.remui %[[F]], %[[C4]] : index
// CHECK:       return %[[R]], %[[D]] : index, index
func.func @regp_transpose_inv(%f: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rp = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  %i, %j = lego.apply_inverse %rp(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// CHECK-LABEL: func.func @regp_3d_cyclic
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK:       %[[T2:.*]] = arith.muli %[[I]], %[[C3]] : index
// CHECK:       %[[T3:.*]] = arith.addi %[[J]], %[[T2]] : index
// CHECK:       %[[T1:.*]] = arith.muli %[[K]], %[[C6]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T3]], %[[T1]] : index
// CHECK:       return %[[RES]] : index
func.func @regp_3d_cyclic(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %rp = lego.reg_p perm [2, 0, 1] dims [%c2, %c3, %c4] : !lego.layout
  %f = lego.apply %rp(%i, %j, %k) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @regp_3d_cyclic_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK:       %[[A:.*]] = arith.divui %[[F]], %[[C6]] : index
// CHECK:       %[[R6:.*]] = arith.remui %[[F]], %[[C6]] : index
// CHECK:       %[[B:.*]] = arith.divui %[[R6]], %[[C3]] : index
// CHECK:       %[[C:.*]] = arith.remui %[[R6]], %[[C3]] : index
// CHECK:       return %[[B]], %[[C]], %[[A]] : index, index, index
func.func @regp_3d_cyclic_inv(%f: index) -> (index, index, index) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %rp = lego.reg_p perm [2, 0, 1] dims [%c2, %c3, %c4] : !lego.layout
  %i, %j, %k = lego.apply_inverse %rp(%f) : !lego.layout -> index, index, index
  return %i, %j, %k : index, index, index
}

// CHECK-LABEL: func.func @regp_1d
// CHECK-SAME:  (%[[I:.*]]: index)
// CHECK:       return %[[I]] : index
func.func @regp_1d(%i: index) -> index {
  %c10 = arith.constant 10 : index
  %rp = lego.reg_p perm [0] dims [%c10] : !lego.layout
  %f = lego.apply %rp(%i) : !lego.layout
  return %f : index
}

// ============================================================================
// OrderBy
// ============================================================================

// CHECK-LABEL: func.func @orderby_simple
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[I]], %[[C8]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[MUL]], %[[J]] : index
// CHECK:       return %[[RES]] : index
func.func @orderby_simple(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %p1 = lego.reg_p perm [0] dims [%c4] : !lego.layout
  %c8 = arith.constant 8 : index
  %p2 = lego.reg_p perm [0] dims [%c8] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %f = lego.apply %ob(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @orderby_simple_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[R8:.*]] = arith.remui %[[F]], %[[C8]] : index
// CHECK:       %[[D8:.*]] = arith.divui %[[F]], %[[C8]] : index
// CHECK:       %[[R4:.*]] = arith.remui %[[D8]], %[[C4]] : index
// CHECK:       return %[[R4]], %[[R8]] : index, index
func.func @orderby_simple_inv(%f: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %p1 = lego.reg_p perm [0] dims [%c4] : !lego.layout
  %c8 = arith.constant 8 : index
  %p2 = lego.reg_p perm [0] dims [%c8] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %i, %j = lego.apply_inverse %ob(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// CHECK-LABEL: func.func @orderby_3blocks
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[T1:.*]] = arith.muli %{{.*}}, %[[C3]] : index
// CHECK:       %[[T2:.*]] = arith.addi %[[T1]], %{{.*}} : index
// CHECK:       %[[T3:.*]] = arith.muli %[[T2]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T3]], %[[K]] : index
// CHECK:       return %[[RES]] : index
func.func @orderby_3blocks(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %p1 = lego.reg_p perm [0] dims [%c2] : !lego.layout
  %c3 = arith.constant 3 : index
  %p2 = lego.reg_p perm [0] dims [%c3] : !lego.layout
  %c4 = arith.constant 4 : index
  %p3 = lego.reg_p perm [0] dims [%c4] : !lego.layout
  %ob = lego.order_by(%p1, %p2, %p3) : !lego.layout
  %f = lego.apply %ob(%i, %j, %k) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @orderby_mixed
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[T1:.*]] = arith.muli %{{.*}}, %[[C2]] : index
// CHECK:       %[[T2:.*]] = arith.addi %{{.*}}, %[[T1]] : index
// CHECK:       %[[T3:.*]] = arith.muli %[[T2]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T3]], %{{.*}} : index
// CHECK:       return %[[RES]] : index
func.func @orderby_mixed(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %p1 = lego.reg_p perm [1, 0] dims [%c2, %c3] : !lego.layout
  %c4 = arith.constant 4 : index
  %p2 = lego.reg_p perm [0] dims [%c4] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %f = lego.apply %ob(%i, %j, %k) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @orderby_single_block
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[J]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[I]], %[[MUL]] : index
// CHECK:       return %[[RES]] : index
func.func @orderby_single_block(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rp = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  %ob = lego.order_by(%rp) : !lego.layout
  %f = lego.apply %ob(%i, %j) : !lego.layout
  return %f : index
}

// ============================================================================
// GroupBy
// ============================================================================

// CHECK-LABEL: func.func @groupby_identity
// CHECK:       return
func.func @groupby_identity(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %p1 = lego.reg_p perm [0] dims [%c4] : !lego.layout
  %c8 = arith.constant 8 : index
  %p2 = lego.reg_p perm [0] dims [%c8] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %gb = lego.group_by [%c4, %c8](%ob) : !lego.layout
  %f = lego.apply %gb(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @groupby_transpose
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[FLAT:.*]] = arith.muli %[[I]], %[[C8]] : index
// CHECK:       %[[FLAT2:.*]] = arith.addi %[[J]], %[[FLAT]] : index
// CHECK:       %[[A:.*]] = arith.divui %[[FLAT2]], %[[C8]] : index
// CHECK:       %[[B:.*]] = arith.remui %[[FLAT2]], %[[C8]] : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[B]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[A]], %[[MUL]] : index
// CHECK:       return %[[RES]] : index
func.func @groupby_transpose(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rp = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  %ob = lego.order_by(%rp) : !lego.layout
  %gb = lego.group_by [%c4, %c8](%ob) : !lego.layout
  %f = lego.apply %gb(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @groupby_transpose_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK:       %[[R32:.*]] = arith.remui %[[F]], %[[C32]] : index
// CHECK:       %[[D4:.*]] = arith.divui %[[R32]], %[[C4]] : index
// CHECK:       %[[R4:.*]] = arith.remui %[[R32]], %[[C4]] : index
// CHECK:       %[[M8:.*]] = arith.muli %[[R4]], %[[C8]] : index
// CHECK:       %[[A:.*]] = arith.addi %[[D4]], %[[M8]] : index
// CHECK:       %[[RI:.*]] = arith.divui %[[A]], %[[C8]] : index
// CHECK:       %[[RJ:.*]] = arith.remui %[[A]], %[[C8]] : index
// CHECK:       return %[[RI]], %[[RJ]] : index, index
func.func @groupby_transpose_inv(%f: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %rp = lego.reg_p perm [1, 0] dims [%c4, %c8] : !lego.layout
  %ob = lego.order_by(%rp) : !lego.layout
  %gb = lego.group_by [%c4, %c8](%ob) : !lego.layout
  %i, %j = lego.apply_inverse %gb(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// CHECK-LABEL: func.func @groupby_multi_obj
// CHECK-SAME:  (%[[I:.*]]: index)
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK:       return
func.func @groupby_multi_obj(%i: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %rp1 = lego.reg_p perm [1, 0] dims [%c2, %c3] : !lego.layout
  %ob1 = lego.order_by(%rp1) : !lego.layout
  %rp2 = lego.reg_p perm [0, 1] dims [%c2, %c3] : !lego.layout
  %ob2 = lego.order_by(%rp2) : !lego.layout
  %c6 = arith.constant 6 : index
  %gb = lego.group_by [%c6](%ob1, %ob2) : !lego.layout
  %f = lego.apply %gb(%i) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @groupby_identity_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK:       return
func.func @groupby_identity_inv(%f: index) -> (index, index) {
  %c4 = arith.constant 4 : index
  %p1 = lego.reg_p perm [0] dims [%c4] : !lego.layout
  %c8 = arith.constant 8 : index
  %p2 = lego.reg_p perm [0] dims [%c8] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %gb = lego.group_by [%c4, %c8](%ob) : !lego.layout
  %i, %j = lego.apply_inverse %gb(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// ============================================================================
// TileBy
// ============================================================================

// CHECK-LABEL: func.func @tileby_1d_apply
// CHECK-SAME:  (%[[IT:.*]]: index, %[[IB:.*]]: index)
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK:       %[[T:.*]] = arith.muli %[[IT]], %[[C16]] : index
// CHECK:       %[[FLAT:.*]] = arith.addi %[[IB]], %[[T]] : index
// CHECK:       arith.divui %[[FLAT]], %[[C16]] : index
// CHECK:       arith.remui %[[FLAT]], %[[C16]] : index
// CHECK:       %[[FINAL:.*]] = arith.addi 
// CHECK:       return %[[FINAL]] : index
func.func @tileby_1d_apply(%it: index, %ib: index) -> index {
  %c64 = arith.constant 64 : index
  %inner = lego.row [%c64] : !lego.layout
  // TileBy expects OrderBy input
  %ob = lego.order_by(%inner) : !lego.layout
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %tb = lego.tile_by %ob tile_dims [[%c4], [%c16]] : !lego.layout
  %f = lego.apply %tb(%it, %ib) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @tileby_1d_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK:       arith.remui %[[F]], %{{.*}} : index
// CHECK:       %[[REM:.*]] = arith.remui %{{.*}} : index
// CHECK:       %[[IT:.*]] = arith.divui %[[REM]], %[[C16]] : index
// CHECK:       %[[IB:.*]] = arith.remui %[[REM]], %[[C16]] : index
// CHECK:       return %{{.*}}, %{{.*}} : index, index
func.func @tileby_1d_inv(%f: index) -> (index, index) {
  %c64 = arith.constant 64 : index
  %inner = lego.row [%c64] : !lego.layout
  %ob = lego.order_by(%inner) : !lego.layout
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %tb = lego.tile_by %ob tile_dims [[%c4], [%c16]] : !lego.layout
  %it, %ib = lego.apply_inverse %tb(%f) : !lego.layout -> index, index
  return %it, %ib : index, index
}

// ============================================================================
// Nested / Composed layouts
// ============================================================================

// CHECK-LABEL: func.func @nested_orderby
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK:       %[[T1:.*]] = arith.muli %{{.*}}, %[[C3]] : index
// CHECK:       %[[T2:.*]] = arith.addi %[[T1]], %{{.*}} : index
// CHECK:       %[[T3:.*]] = arith.muli %[[T2]], %[[C5]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T3]], %{{.*}} : index
// CHECK:       return %[[RES]] : index
func.func @nested_orderby(%i: index, %j: index, %k: index) -> index {
  %c2 = arith.constant 2 : index
  %inner_p1 = lego.reg_p perm [0] dims [%c2] : !lego.layout
  %c3 = arith.constant 3 : index
  %inner_p2 = lego.reg_p perm [0] dims [%c3] : !lego.layout
  %inner_ob = lego.order_by(%inner_p1, %inner_p2) : !lego.layout
  %c5 = arith.constant 5 : index
  %outer_p = lego.reg_p perm [0] dims [%c5] : !lego.layout
  %outer_ob = lego.order_by(%inner_ob, %outer_p) : !lego.layout
  %f = lego.apply %outer_ob(%i, %j, %k) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @groupby_with_mixed_orderby
// CHECK:       arith.muli
// CHECK:       return
func.func @groupby_with_mixed_orderby(%i: index, %j: index) -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %rp1 = lego.reg_p perm [1, 0] dims [%c2, %c3] : !lego.layout
  %c4 = arith.constant 4 : index
  %rp2 = lego.reg_p perm [0] dims [%c4] : !lego.layout
  %ob = lego.order_by(%rp1, %rp2) : !lego.layout
  %c6 = arith.constant 6 : index
  %gb = lego.group_by [%c6, %c4](%ob) : !lego.layout
  %f = lego.apply %gb(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @row_vs_col
// CHECK:       arith.muli
// CHECK:       arith.muli
// CHECK:       return
func.func @row_vs_col(%i: index, %j: index) -> (index, index) {
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %row = lego.row [%c3, %c5] : !lego.layout
  %col = lego.col [%c3, %c5] : !lego.layout
  %frow = lego.apply %row(%i, %j) : !lego.layout
  %fcol = lego.apply %col(%i, %j) : !lego.layout
  return %frow, %fcol : index, index
}
