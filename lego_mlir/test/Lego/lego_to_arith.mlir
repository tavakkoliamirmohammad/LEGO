// RUN: lego-opt --lego-to-arith %s | FileCheck %s

// ============================================================================
// Row — N-D Row-Major (identity permutation)
// Python: Row(*dims) = RegP(dims, [0, 1, ..., d-1])
// apply(i0, ..., iN) = flatten_index(idx, dims)
// inv(flat) = unflatten_index(flat, dims)
// ============================================================================

// --- Row 2D: Row(4, 8).apply(i, j) = i*8 + j ---
// CHECK-LABEL: func.func @row_2d_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[I]], %[[C8]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[MUL]], %[[J]] : index
// CHECK:       return %[[RES]] : index
func.func @row_2d_apply(%i: index, %j: index) -> index {
  %r = lego.row [4, 8] : !lego.layout
  %f = lego.apply %r(%i, %j) : !lego.layout
  return %f : index
}

// --- Row 3D: Row(2, 3, 4).apply(i, j, k) = i*12 + j*4 + k ---
// CHECK-LABEL: func.func @row_3d_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C12:.*]] = arith.constant 12 : index
// CHECK:       %[[T1:.*]] = arith.muli %[[I]], %[[C12]] : index
// CHECK:       %[[T2:.*]] = arith.muli %[[J]], %[[C4]] : index
// CHECK:       %[[T3:.*]] = arith.addi %[[T1]], %[[T2]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T3]], %[[K]] : index
// CHECK:       return %[[RES]] : index
func.func @row_3d_apply(%i: index, %j: index, %k: index) -> index {
  %r = lego.row [2, 3, 4] : !lego.layout
  %f = lego.apply %r(%i, %j, %k) : !lego.layout
  return %f : index
}

// --- Row 4D: Row(2, 3, 4, 5).apply(i, j, k, l) = i*60 + j*20 + k*5 + l ---
// CHECK-LABEL: func.func @row_4d_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index, %[[L:.*]]: index)
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C20:.*]] = arith.constant 20 : index
// CHECK-DAG:   %[[C60:.*]] = arith.constant 60 : index
// CHECK:       %[[T1:.*]] = arith.muli %[[I]], %[[C60]] : index
// CHECK:       %[[T2:.*]] = arith.muli %[[J]], %[[C20]] : index
// CHECK:       %[[T3:.*]] = arith.addi %[[T1]], %[[T2]] : index
// CHECK:       %[[T4:.*]] = arith.muli %[[K]], %[[C5]] : index
// CHECK:       %[[T5:.*]] = arith.addi %[[T3]], %[[T4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T5]], %[[L]] : index
// CHECK:       return %[[RES]] : index
func.func @row_4d_apply(%i: index, %j: index, %k: index, %l: index) -> index {
  %r = lego.row [2, 3, 4, 5] : !lego.layout
  %f = lego.apply %r(%i, %j, %k, %l) : !lego.layout
  return %f : index
}

// --- Row 2D inverse: inv(f) = (f/8, f%8) ---
// CHECK-LABEL: func.func @row_2d_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[I:.*]] = arith.divui %[[F]], %[[C8]] : index
// CHECK:       %[[J:.*]] = arith.remui %[[F]], %[[C8]] : index
// CHECK:       return %[[I]], %[[J]] : index, index
func.func @row_2d_inv(%f: index) -> (index, index) {
  %r = lego.row [4, 8] : !lego.layout
  %i, %j = lego.apply_inverse %r(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// --- Row 3D inverse: inv(f) = (f/12, (f%12)/4, f%4) ---
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
  %r = lego.row [2, 3, 4] : !lego.layout
  %i, %j, %k = lego.apply_inverse %r(%f) : !lego.layout -> index, index, index
  return %i, %j, %k : index, index, index
}

// ============================================================================
// Col — N-D Column-Major (reversed permutation)
// Python: Col(*dims) = RegP(dims, [d-1, d-2, ..., 0])
// Col(n, m).apply(i, j) = j*n + i
// Col(a, b, c).apply(i, j, k) = k*(a*b) + j*a + i
// ============================================================================

// --- Col 2D: Col(4, 8).apply(i, j) = j*4 + i ---
// CHECK-LABEL: func.func @col_2d_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[J]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[MUL]], %[[I]] : index
// CHECK:       return %[[RES]] : index
func.func @col_2d_apply(%i: index, %j: index) -> index {
  %c = lego.col [4, 8] : !lego.layout
  %f = lego.apply %c(%i, %j) : !lego.layout
  return %f : index
}

// --- Col 3D: Col(2, 3, 4).apply(i, j, k) = k*6 + j*2 + i ---
// sigma(dims,[2,1,0]) = [4,3,2], sigma(idx,[2,1,0]) = (k,j,i)
// flatten((k,j,i), (4,3,2)) = k*6 + j*2 + i
// CHECK-LABEL: func.func @col_3d_apply
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK:       %[[T1:.*]] = arith.muli %[[K]], %[[C6]] : index
// CHECK:       %[[T2:.*]] = arith.muli %[[J]], %[[C2]] : index
// CHECK:       %[[T3:.*]] = arith.addi %[[T1]], %[[T2]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T3]], %[[I]] : index
// CHECK:       return %[[RES]] : index
func.func @col_3d_apply(%i: index, %j: index, %k: index) -> index {
  %c = lego.col [2, 3, 4] : !lego.layout
  %f = lego.apply %c(%i, %j, %k) : !lego.layout
  return %f : index
}

// --- Col 2D inverse: unflatten(f, sigma([4,8],[1,0])) = unflatten(f,[8,4])
// then inv_sigma([1,0]) => return (f%4, f/4) ---
// CHECK-LABEL: func.func @col_2d_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[D:.*]] = arith.divui %[[F]], %[[C4]] : index
// CHECK:       %[[R:.*]] = arith.remui %[[F]], %[[C4]] : index
// CHECK:       return %[[R]], %[[D]] : index, index
func.func @col_2d_inv(%f: index) -> (index, index) {
  %c = lego.col [4, 8] : !lego.layout
  %i, %j = lego.apply_inverse %c(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// ============================================================================
// RegP — Regular Permutation
// Python: RegP(dims, perm)
// apply(idx) = flatten(sigma(idx, perm), sigma(dims, perm))
// inv(flat)  = sigma(unflatten(flat, sigma(dims, perm)), inverse_perm)
// ============================================================================

// --- RegP identity: RegP([4,8], [0,1]).apply(i,j) = i*8 + j (same as Row) ---
// CHECK-LABEL: func.func @regp_identity
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[I]], %[[C8]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[MUL]], %[[J]] : index
// CHECK:       return %[[RES]] : index
func.func @regp_identity(%i: index, %j: index) -> index {
  %rp = lego.reg_p perm [0, 1] dims [4, 8] : !lego.layout
  %f = lego.apply %rp(%i, %j) : !lego.layout
  return %f : index
}

// --- RegP transpose: RegP([4,8], [1,0]).apply(i,j) = j*4 + i ---
// CHECK-LABEL: func.func @regp_transpose
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[J]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[MUL]], %[[I]] : index
// CHECK:       return %[[RES]] : index
func.func @regp_transpose(%i: index, %j: index) -> index {
  %rp = lego.reg_p perm [1, 0] dims [4, 8] : !lego.layout
  %f = lego.apply %rp(%i, %j) : !lego.layout
  return %f : index
}

// --- RegP transpose inverse: inv(f) = (f%4, f/4) ---
// CHECK-LABEL: func.func @regp_transpose_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[D:.*]] = arith.divui %[[F]], %[[C4]] : index
// CHECK:       %[[R:.*]] = arith.remui %[[F]], %[[C4]] : index
// CHECK:       return %[[R]], %[[D]] : index, index
func.func @regp_transpose_inv(%f: index) -> (index, index) {
  %rp = lego.reg_p perm [1, 0] dims [4, 8] : !lego.layout
  %i, %j = lego.apply_inverse %rp(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// --- RegP 3D cyclic: RegP([2,3,4], [2,0,1]).apply(i,j,k)
// sigma(idx,[2,0,1]) = (k,i,j), sigma(dims,[2,0,1]) = (4,2,3)
// flatten((k,i,j),(4,2,3)) = k*6 + i*3 + j ---
// CHECK-LABEL: func.func @regp_3d_cyclic
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK:       %[[T1:.*]] = arith.muli %[[K]], %[[C6]] : index
// CHECK:       %[[T2:.*]] = arith.muli %[[I]], %[[C3]] : index
// CHECK:       %[[T3:.*]] = arith.addi %[[T1]], %[[T2]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T3]], %[[J]] : index
// CHECK:       return %[[RES]] : index
func.func @regp_3d_cyclic(%i: index, %j: index, %k: index) -> index {
  %rp = lego.reg_p perm [2, 0, 1] dims [2, 3, 4] : !lego.layout
  %f = lego.apply %rp(%i, %j, %k) : !lego.layout
  return %f : index
}

// --- RegP 3D cyclic inverse ---
// sigma(dims,[2,0,1]) = (4,2,3). unflatten(f,(4,2,3)) = (f/6, (f%6)/3, f%3)
// inv_perm([2,0,1]) = [1,2,0], so sigma((a,b,c),[1,2,0]) = (b,c,a)
// result: ((f%6)/3, f%3, f/6)
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
  %rp = lego.reg_p perm [2, 0, 1] dims [2, 3, 4] : !lego.layout
  %i, %j, %k = lego.apply_inverse %rp(%f) : !lego.layout -> index, index, index
  return %i, %j, %k : index, index, index
}

// --- RegP 1D: trivial, RegP([10], [0]).apply(i) = i ---
// CHECK-LABEL: func.func @regp_1d
// CHECK-SAME:  (%[[I:.*]]: index)
// CHECK:       return %[[I]] : index
func.func @regp_1d(%i: index) -> index {
  %rp = lego.reg_p perm [0] dims [10] : !lego.layout
  %f = lego.apply %rp(%i) : !lego.layout
  return %f : index
}

// ============================================================================
// OrderBy — Sequence of permutation blocks
// Python: OrderBy(p1, p2, ...)
// apply: flat = 0; for each perm: flat = flat * |perm| + perm.apply(slice)
// inv:   reverse iterate, mod/div to extract sub-index, perm.inv(sub)
// ============================================================================

// --- OrderBy two 1D blocks: OrderBy(RegP([4],[0]), RegP([8],[0]))
// apply(i, j) = i*8 + j ---
// CHECK-LABEL: func.func @orderby_simple
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[I]], %[[C8]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[MUL]], %[[J]] : index
// CHECK:       return %[[RES]] : index
func.func @orderby_simple(%i: index, %j: index) -> index {
  %p1 = lego.reg_p perm [0] dims [4] : !lego.layout
  %p2 = lego.reg_p perm [0] dims [8] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %f = lego.apply %ob(%i, %j) : !lego.layout
  return %f : index
}

// --- OrderBy inverse ---
// CHECK-LABEL: func.func @orderby_simple_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[R8:.*]] = arith.remui %[[F]], %[[C8]] : index
// CHECK:       %[[D8:.*]] = arith.divui %[[F]], %[[C8]] : index
// CHECK:       %[[R4:.*]] = arith.remui %[[D8]], %[[C4]] : index
// CHECK:       return %[[R4]], %[[R8]] : index, index
func.func @orderby_simple_inv(%f: index) -> (index, index) {
  %p1 = lego.reg_p perm [0] dims [4] : !lego.layout
  %p2 = lego.reg_p perm [0] dims [8] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %i, %j = lego.apply_inverse %ob(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// --- OrderBy three blocks: OrderBy(RegP([2],[0]), RegP([3],[0]), RegP([4],[0]))
// apply(i, j, k) = (i*3 + j)*4 + k = i*12 + j*4 + k ---
// CHECK-LABEL: func.func @orderby_3blocks
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[T1:.*]] = arith.muli %[[I]], %[[C3]] : index
// CHECK:       %[[T2:.*]] = arith.addi %[[T1]], %[[J]] : index
// CHECK:       %[[T3:.*]] = arith.muli %[[T2]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T3]], %[[K]] : index
// CHECK:       return %[[RES]] : index
func.func @orderby_3blocks(%i: index, %j: index, %k: index) -> index {
  %p1 = lego.reg_p perm [0] dims [2] : !lego.layout
  %p2 = lego.reg_p perm [0] dims [3] : !lego.layout
  %p3 = lego.reg_p perm [0] dims [4] : !lego.layout
  %ob = lego.order_by(%p1, %p2, %p3) : !lego.layout
  %f = lego.apply %ob(%i, %j, %k) : !lego.layout
  return %f : index
}

// --- OrderBy with 2D sub-block: OrderBy(RegP([2,3],[1,0]), RegP([4],[0]))
// Block 1: RegP([2,3],[1,0]).apply(i,j) = j*2+i (size=6)
// Block 2: RegP([4],[0]).apply(k) = k (size=4)
// flat = (j*2+i)*4 + k ---
// CHECK-LABEL: func.func @orderby_mixed
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[T1:.*]] = arith.muli %[[J]], %[[C2]] : index
// CHECK:       %[[T2:.*]] = arith.addi %[[T1]], %[[I]] : index
// CHECK:       %[[T3:.*]] = arith.muli %[[T2]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T3]], %[[K]] : index
// CHECK:       return %[[RES]] : index
func.func @orderby_mixed(%i: index, %j: index, %k: index) -> index {
  %p1 = lego.reg_p perm [1, 0] dims [2, 3] : !lego.layout
  %p2 = lego.reg_p perm [0] dims [4] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %f = lego.apply %ob(%i, %j, %k) : !lego.layout
  return %f : index
}

// --- OrderBy with single 2D block: same as RegP applied directly ---
// CHECK-LABEL: func.func @orderby_single_block
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[J]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[MUL]], %[[I]] : index
// CHECK:       return %[[RES]] : index
func.func @orderby_single_block(%i: index, %j: index) -> index {
  %rp = lego.reg_p perm [1, 0] dims [4, 8] : !lego.layout
  %ob = lego.order_by(%rp) : !lego.layout
  %f = lego.apply %ob(%i, %j) : !lego.layout
  return %f : index
}

// ============================================================================
// GroupBy — Groups dimensions and applies a sequence of layout objects
// Python: GroupBy(group_dims, objects)
// apply(*idx):
//   current = flatten(idx, group_dims)
//   for obj in reversed(objects): unflatten(current, obj.dims()) → obj.apply
// inv(flat):
//   for obj in objects: obj.inv(current) → flatten into obj.dims()
//   unflatten(result, group_dims)
// ============================================================================

// --- GroupBy identity: GroupBy([4,8], [OrderBy(RegP([4],[0]),RegP([8],[0]))])
// identity transform: flatten → unflatten(same dims) → re-apply = same ---
// CHECK-LABEL: func.func @groupby_identity
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK:       return
func.func @groupby_identity(%i: index, %j: index) -> index {
  %p1 = lego.reg_p perm [0] dims [4] : !lego.layout
  %p2 = lego.reg_p perm [0] dims [8] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %gb = lego.group_by [4, 8](%ob) : !lego.layout
  %f = lego.apply %gb(%i, %j) : !lego.layout
  return %f : index
}

// --- GroupBy with transpose: GroupBy([4,8], [OrderBy(RegP([4,8],[1,0]))])
// apply(i, j):
//   current = i*8 + j
//   unflatten(current, [4,8]) = (current/8, current%8)
//   RegP([4,8],[1,0]).apply(a,b) = b*4+a
//   result = j*4 + i ---
// CHECK-LABEL: func.func @groupby_transpose
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:       %[[FLAT:.*]] = arith.muli %[[I]], %[[C8]] : index
// CHECK:       %[[FLAT2:.*]] = arith.addi %[[FLAT]], %[[J]] : index
// CHECK:       %[[A:.*]] = arith.divui %[[FLAT2]], %[[C8]] : index
// CHECK:       %[[B:.*]] = arith.remui %[[FLAT2]], %[[C8]] : index
// CHECK:       %[[MUL:.*]] = arith.muli %[[B]], %[[C4]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[MUL]], %[[A]] : index
// CHECK:       return %[[RES]] : index
func.func @groupby_transpose(%i: index, %j: index) -> index {
  %rp = lego.reg_p perm [1, 0] dims [4, 8] : !lego.layout
  %ob = lego.order_by(%rp) : !lego.layout
  %gb = lego.group_by [4, 8](%ob) : !lego.layout
  %f = lego.apply %gb(%i, %j) : !lego.layout
  return %f : index
}

// --- GroupBy transpose inverse ---
// inv(f):
//   obj = OrderBy(RegP([4,8],[1,0])), obj.inv(f):
//     unflatten(f, sigma([4,8],[1,0])) = unflatten(f, [8,4])
//     = (f/4, f%4), inv_sigma=(f%4, f/4)
//   flatten((f%4, f/4), [4,8]) = (f%4)*8 + f/4
//   unflatten(result, [4,8])
// CHECK-LABEL: func.func @groupby_transpose_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
// CHECK:       %[[R32:.*]] = arith.remui %[[F]], %[[C32]] : index
// CHECK:       %[[D4:.*]] = arith.divui %[[R32]], %[[C4]] : index
// CHECK:       %[[R4:.*]] = arith.remui %[[R32]], %[[C4]] : index
// CHECK:       %[[M8:.*]] = arith.muli %[[R4]], %[[C8]] : index
// CHECK:       %[[A:.*]] = arith.addi %[[M8]], %[[D4]] : index
// CHECK:       %[[RI:.*]] = arith.divui %[[A]], %[[C8]] : index
// CHECK:       %[[RJ:.*]] = arith.remui %[[A]], %[[C8]] : index
// CHECK:       return %[[RI]], %[[RJ]] : index, index
func.func @groupby_transpose_inv(%f: index) -> (index, index) {
  %rp = lego.reg_p perm [1, 0] dims [4, 8] : !lego.layout
  %ob = lego.order_by(%rp) : !lego.layout
  %gb = lego.group_by [4, 8](%ob) : !lego.layout
  %i, %j = lego.apply_inverse %gb(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// --- GroupBy with two objects: GroupBy([6], [O1_transpose, O2_identity])
// apply(i):
//   current = i (flatten 1D is identity)
//   reversed: first O2_identity (no-op), then O1_transpose
//   O2 = OrderBy(RegP([2,3],[0,1])):
//     unflatten(i, [2,3]) = (i/3, i%3)
//     RegP identity: i/3*3 + i%3 = i (no-op)
//   O1 = OrderBy(RegP([2,3],[1,0])):
//     unflatten(i, [2,3]) = (i/3, i%3)
//     RegP transpose: (i%3)*2 + (i/3) ---
// CHECK-LABEL: func.func @groupby_multi_obj
// CHECK-SAME:  (%[[I:.*]]: index)
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK:       return
func.func @groupby_multi_obj(%i: index) -> index {
  %rp1 = lego.reg_p perm [1, 0] dims [2, 3] : !lego.layout
  %ob1 = lego.order_by(%rp1) : !lego.layout
  %rp2 = lego.reg_p perm [0, 1] dims [2, 3] : !lego.layout
  %ob2 = lego.order_by(%rp2) : !lego.layout
  %gb = lego.group_by [6](%ob1, %ob2) : !lego.layout
  %f = lego.apply %gb(%i) : !lego.layout
  return %f : index
}

// --- GroupBy identity inverse (round-trip check) ---
// CHECK-LABEL: func.func @groupby_identity_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK:       return
func.func @groupby_identity_inv(%f: index) -> (index, index) {
  %p1 = lego.reg_p perm [0] dims [4] : !lego.layout
  %p2 = lego.reg_p perm [0] dims [8] : !lego.layout
  %ob = lego.order_by(%p1, %p2) : !lego.layout
  %gb = lego.group_by [4, 8](%ob) : !lego.layout
  %i, %j = lego.apply_inverse %gb(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}

// ============================================================================
// TileBy — Tile dimensions: Python-style multi-level tiling
// Creates GroupBy([tile_dims], ...) which flattens the input split indices.
// ============================================================================

// --- TileBy 1D: TileBy([4], [16]). d=1. q=2. ---
// Input indices: (i_t, i_b).
// Flatten with [4, 16] -> i_t * 16 + i_b.
// Interleave (d=1, q=2) -> [0, 1] (identity).
// Inner layout Row(64) -> identity.
// Result: i_t * 16 + i_b.

// CHECK-LABEL: func.func @tileby_1d_apply
// CHECK-SAME:  (%[[IT:.*]]: index, %[[IB:.*]]: index)
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK:       %[[T:.*]] = arith.muli %[[IT]], %[[C16]] : index
// CHECK:       %[[FLAT:.*]] = arith.addi %[[T]], %[[IB]] : index
// CHECK:       return %[[FLAT]] : index
func.func @tileby_1d_apply(%it: index, %ib: index) -> index {
  %inner = lego.row [64] : !lego.layout
  // TileBy expects OrderBy input
  %ob = lego.order_by(%inner) : !lego.layout
  %tb = lego.tile_by %ob tile_dims [[4], [16]] : !lego.layout
  %f = lego.apply %tb(%it, %ib) : !lego.layout
  return %f : index
}

// --- TileBy inverse ---
// CHECK-LABEL: func.func @tileby_1d_inv
// CHECK-SAME:  (%[[F:.*]]: index)
// CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
// CHECK:       %[[IT:.*]] = arith.divui %[[F]], %[[C16]] : index
// CHECK:       %[[IB:.*]] = arith.remui %[[F]], %[[C16]] : index
// CHECK:       return %[[IT]], %[[IB]] : index, index
func.func @tileby_1d_inv(%f: index) -> (index, index) {
  %inner = lego.row [64] : !lego.layout
  %ob = lego.order_by(%inner) : !lego.layout
  %tb = lego.tile_by %ob tile_dims [[4], [16]] : !lego.layout
  %it, %ib = lego.apply_inverse %tb(%f) : !lego.layout -> index, index
  return %it, %ib : index, index
}

// ============================================================================
// Nested / Composed layouts
// ============================================================================

// --- OrderBy nested in OrderBy: OrderBy contains an OrderBy sub-block ---
// Outer: OrderBy(inner_orderby, RegP([5],[0]))
// inner_orderby = OrderBy(RegP([2],[0]), RegP([3],[0]))
// inner_orderby.apply(i,j) = i*3 + j (size=6)
// Outer.apply(i,j,k) = (i*3+j)*5 + k = i*15 + j*5 + k
// CHECK-LABEL: func.func @nested_orderby
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index, %[[K:.*]]: index)
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK:       %[[T1:.*]] = arith.muli %[[I]], %[[C3]] : index
// CHECK:       %[[T2:.*]] = arith.addi %[[T1]], %[[J]] : index
// CHECK:       %[[T3:.*]] = arith.muli %[[T2]], %[[C5]] : index
// CHECK:       %[[RES:.*]] = arith.addi %[[T3]], %[[K]] : index
// CHECK:       return %[[RES]] : index
func.func @nested_orderby(%i: index, %j: index, %k: index) -> index {
  %inner_p1 = lego.reg_p perm [0] dims [2] : !lego.layout
  %inner_p2 = lego.reg_p perm [0] dims [3] : !lego.layout
  %inner_ob = lego.order_by(%inner_p1, %inner_p2) : !lego.layout
  %outer_p = lego.reg_p perm [0] dims [5] : !lego.layout
  %outer_ob = lego.order_by(%inner_ob, %outer_p) : !lego.layout
  %f = lego.apply %outer_ob(%i, %j, %k) : !lego.layout
  return %f : index
}

// --- GroupBy with OrderBy that contains a transposed 2D sub-block ---
// GroupBy([6,4], [OrderBy(RegP([2,3],[1,0]), RegP([4],[0]))])
// apply(i, j):
//   current = flatten((i,j), [6,4]) = i*4 + j
//   unflatten(current, [2,3,4]) where obj dims = [2,3,4]
//     => (current/12, (current%12)/4, current%4)
//   OrderBy: block1=RegP([2,3],[1,0]).apply(a,b) = b*2+a (size=6)
//            block2=RegP([4],[0]).apply(c) = c (size=4)
//     => (b*2+a)*4 + c
// CHECK-LABEL: func.func @groupby_with_mixed_orderby
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK:       arith.muli
// CHECK:       return
func.func @groupby_with_mixed_orderby(%i: index, %j: index) -> index {
  %rp1 = lego.reg_p perm [1, 0] dims [2, 3] : !lego.layout
  %rp2 = lego.reg_p perm [0] dims [4] : !lego.layout
  %ob = lego.order_by(%rp1, %rp2) : !lego.layout
  %gb = lego.group_by [6, 4](%ob) : !lego.layout
  %f = lego.apply %gb(%i, %j) : !lego.layout
  return %f : index
}

// --- Row and Col produce complementary results for same dims ---
// Row([3,5]).apply(1,2) vs Col([3,5]).apply(1,2)
// Row: 1*5+2 = 7, Col: 2*3+1 = 7 ... wait both are 7 for this input
// Let's verify separation: Row(i,j) = i*5+j, Col(i,j) = j*3+i
// CHECK-LABEL: func.func @row_vs_col
// CHECK-SAME:  (%[[I:.*]]: index, %[[J:.*]]: index)
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK:       %[[ROW:.*]] = arith.muli %[[I]], %[[C5]] : index
// CHECK:       %[[COL:.*]] = arith.muli %[[J]], %[[C3]] : index
// CHECK:       return
func.func @row_vs_col(%i: index, %j: index) -> (index, index) {
  %row = lego.row [3, 5] : !lego.layout
  %col = lego.col [3, 5] : !lego.layout
  %frow = lego.apply %row(%i, %j) : !lego.layout
  %fcol = lego.apply %col(%i, %j) : !lego.layout
  return %frow, %fcol : index, index
}
