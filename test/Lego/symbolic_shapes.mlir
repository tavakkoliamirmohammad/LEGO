// RUN: lego-opt -lego-lower %s | FileCheck %s

// CHECK-LABEL: func.func @row_symbolic
// CHECK-SAME:  (%[[RI:.*]]: index, %[[RJ:.*]]: index, %[[RD0:.*]]: index, %[[RD1:.*]]: index)
// CHECK:       %[[RT0:.*]] = arith.muli %[[RI]], %[[RD1]] : index
// CHECK:       %[[RT1:.*]] = arith.addi %[[RJ]], %[[RT0]] : index
// CHECK:       return %[[RT1]] : index
func.func @row_symbolic(%i: index, %j: index, %d0: index, %d1: index) -> index {
  %r = lego.row [%d0, %d1] : !lego.layout
  %f = lego.apply %r(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @col_symbolic
// CHECK-SAME:  (%[[CI:.*]]: index, %[[CJ:.*]]: index, %[[CD0:.*]]: index, %[[CD1:.*]]: index)
// CHECK:       %[[CT0:.*]] = arith.muli %[[CJ]], %[[CD0]] : index
// CHECK:       %[[CT1:.*]] = arith.addi %[[CI]], %[[CT0]] : index
// CHECK:       return %[[CT1]] : index
func.func @col_symbolic(%i: index, %j: index, %d0: index, %d1: index) -> index {
  %c = lego.col [%d0, %d1] : !lego.layout
  %f = lego.apply %c(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @reg_p_symbolic
// CHECK-SAME:  (%[[RPI:.*]]: index, %[[RPJ:.*]]: index, %[[RPD0:.*]]: index, %[[RPD1:.*]]: index)
// CHECK:       %[[RPT0:.*]] = arith.muli %[[RPJ]], %[[RPD0]] : index
// CHECK:       %[[RPT1:.*]] = arith.addi %[[RPI]], %[[RPT0]] : index
// CHECK:       return %[[RPT1]] : index
func.func @reg_p_symbolic(%i: index, %j: index, %d0: index, %d1: index) -> index {
  %rp = lego.reg_p perm [1, 0] dims [%d0, %d1] : !lego.layout
  %f = lego.apply %rp(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @gen_p_symbolic
// CHECK-SAME:  (%[[GI:.*]]: index, %[[GJ:.*]]: index, %[[GD0:.*]]: index, %[[GD1:.*]]: index)
// CHECK:       %[[GMUL:.*]] = arith.muli %[[GI]], %[[GD1]] : index
// CHECK:       %[[GRES:.*]] = arith.addi %[[GMUL]], %[[GJ]] : index
// CHECK:       return %[[GRES]] : index
func.func @gen_p_symbolic(%i: index, %j: index, %d0: index, %d1: index) -> index {
  %layout = lego.gen_p [%d0, %d1]
    apply (%arg0: index, %arg1: index) {
      %mul = arith.muli %arg0, %d1 : index
      %res = arith.addi %mul, %arg1 : index
      lego.yield %res : index
    }
    inv (%flat: index) {
      %ii = arith.divui %flat, %d1 : index
      %jj = arith.remui %flat, %d1 : index
      lego.yield %ii, %jj : index, index
    } : !lego.layout
  %f = lego.apply %layout(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @tile_by_symbolic
// CHECK-SAME:  (%[[TI:.*]]: index, %[[TJ:.*]]: index, %[[TD0:.*]]: index, %[[TD1:.*]]: index)
// CHECK:       %[[TMUL:.*]] = arith.muli %[[TI]], %[[TD1]] : index
// CHECK:       %[[TRES:.*]] = arith.addi %[[TJ]], %[[TMUL]] : index
// CHECK:       return %[[TRES]] : index
func.func @tile_by_symbolic(%i: index, %j: index, %d0: index, %d1: index) -> index {
  %r = lego.row [%d0, %d1] : !lego.layout
  %ob = lego.order_by(%r) : !lego.layout
  %tb = lego.tile_by %ob tile_dims [[%d0, %d1]] : !lego.layout
  %f = lego.apply %tb(%i, %j) : !lego.layout
  return %f : index
}

// CHECK-LABEL: func.func @apply_inverse_row_symbolic
// CHECK-SAME:  (%[[AIFLAT:.*]]: index, %[[AID0:.*]]: index, %[[AID1:.*]]: index)
// CHECK:       %[[AII:.*]] = arith.divui %[[AIFLAT]], %[[AID1]] : index
// CHECK:       %[[AIJ:.*]] = arith.remui %[[AIFLAT]], %[[AID1]] : index
// CHECK:       return %[[AII]], %[[AIJ]] : index, index
func.func @apply_inverse_row_symbolic(%f: index, %d0: index, %d1: index) -> (index, index) {
  %r = lego.row [%d0, %d1] : !lego.layout
  %i, %j = lego.apply_inverse %r(%f) : !lego.layout -> index, index
  return %i, %j : index, index
}
