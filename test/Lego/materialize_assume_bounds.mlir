// RUN: lego-opt %s -lego-materialize-assume-bounds -split-input-file | FileCheck %s --check-prefix=MAT
// RUN: lego-opt %s -lego-materialize-assume-bounds -lego-materialize-assume-bounds='cleanup=true' -split-input-file | FileCheck %s --check-prefix=RT

// Standalone tests for the lego-materialize-assume-bounds pass.
// MAT prefix:  after materialization (remui inserted, assume ops erased)
// RT  prefix:  after materialization + cleanup round-trip (remui removed)

// -----

// assume_bounds with ub: inserts remui
// MAT-LABEL: func.func @ub_only
// MAT-SAME:  (%[[X:.*]]: index, %[[D:.*]]: index)
// MAT:       %[[REM:.*]] = arith.remui %[[X]], %[[D]] {{.*}} : index
// MAT-NOT:   lego.assume_bounds
// MAT:       return %[[REM]] : index

// RT-LABEL: func.func @ub_only
// RT-SAME:  (%[[X:.*]]: index, %[[D:.*]]: index)
// RT-NOT:   arith.remui
// RT:       return %[[X]] : index
func.func @ub_only(%x: index, %d: index) -> index {
  lego.assume_bounds %x ub: %d
  return %x : index
}

// -----

// assume_bounds with lb+ub: ub materializes, lb is discarded
// MAT-LABEL: func.func @lb_and_ub
// MAT-SAME:  (%[[X:.*]]: index, %[[LB:.*]]: index, %[[UB:.*]]: index)
// MAT:       %[[REM:.*]] = arith.remui %[[X]], %[[UB]] {{.*}} : index
// MAT-NOT:   lego.assume_bounds
// MAT:       return %[[REM]] : index

// RT-LABEL: func.func @lb_and_ub
// RT-NOT:   arith.remui
// RT:       return %{{.*}} : index
func.func @lb_and_ub(%x: index, %lb: index, %ub: index) -> index {
  lego.assume_bounds %x lb: %lb ub: %ub
  return %x : index
}

// -----

// lb-only: erased without inserting remui
// MAT-LABEL: func.func @lb_only
// MAT-SAME:  (%[[X:.*]]: index, %[[LB:.*]]: index)
// MAT-NOT:   lego.assume_bounds
// MAT-NOT:   arith.remui
// MAT:       return %[[X]] : index

// RT-LABEL: func.func @lb_only
// RT-NOT:   lego.assume_bounds
// RT-NOT:   arith.remui
// RT:       return %{{.*}} : index
func.func @lb_only(%x: index, %lb: index) -> index {
  lego.assume_bounds %x lb: %lb
  return %x : index
}

// -----

// Already-lowered form: lego.assume(cmpi slt, block_arg, d)
// Normalization lowers assume_bounds to cmpi + lego.assume.  The pass
// also handles this form when %x is a block argument.
// MAT-LABEL: func.func @lowered_form
// MAT-SAME:  (%[[X:.*]]: index, %[[D:.*]]: index)
// MAT:       %[[REM:.*]] = arith.remui %[[X]], %[[D]] {{.*}} : index
// MAT-NOT:   lego.assume
// MAT:       return %[[REM]] : index

// RT-LABEL: func.func @lowered_form
// RT-NOT:   arith.remui
// RT:       return %{{.*}} : index
func.func @lowered_form(%x: index, %d: index) -> index {
  %cmp = arith.cmpi slt, %x, %d : index
  lego.assume %cmp
  return %x : index
}

// -----

// Skip computed values: only block args get remui
// When the cmpi LHS is computed (not a block argument), the pass skips it.
// MAT-LABEL: func.func @skip_computed
// MAT-SAME:  (%[[A:.*]]: index, %[[B:.*]]: index, %[[D:.*]]: index)
// MAT:       %[[SUM:.*]] = arith.addi %[[A]], %[[B]] : index
// MAT:       lego.assume
// MAT:       return %[[SUM]] : index

// RT-LABEL: func.func @skip_computed
// RT:       arith.addi
// RT:       return
func.func @skip_computed(%a: index, %b: index, %d: index) -> index {
  %sum = arith.addi %a, %b : index
  %cmp = arith.cmpi slt, %sum, %d : index
  lego.assume %cmp
  return %sum : index
}
