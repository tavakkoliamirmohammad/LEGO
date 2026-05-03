// RUN: lego-opt %s --lego-to-arm-sve | FileCheck %s

// ARM SVE pipeline IR-shape verification test (R15).
//
// This test validates the *shape* of the emitted LLVM IR for the
// --lego-to-arm-sve pipeline: that LEGO dialect ops are lowered, vector ops
// appear, and the function signature survives lowering.
//
// NOTE: This test verifies IR shape ONLY.  Runtime validation requires ARM SVE
// hardware (Neoverse V1/V2, Apple M4, etc.) which is not present on this CHPC
// node (Intel Xeon Gold 6330).  The emitted fixed-width vectors (2xf64, 4xf32
// at vscale=1) are correct for SVE targets: the LLVM AArch64 backend legalizes
// them to full SVE width when +sve is specified at llc time.
//
// Vector lane widths (SVE vscale=1 = 16-byte NEON-equivalent):
//   f64: 16/8 = 2 lanes → vector<2xf64>
//   f32: 16/4 = 4 lanes → vector<4xf32>

// CHECK-LABEL: llvm.func @saxpy_sve
// CHECK-NOT: lego.
// CHECK-NOT: scf.for
// CHECK: vector
// CHECK: llvm.return
func.func @saxpy_sve(%a: f64, %X: memref<?xf64>, %Y: memref<?xf64>, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %N step %c1 {
    %xi = memref.load %X[%i] : memref<?xf64>
    %yi = memref.load %Y[%i] : memref<?xf64>
    %p  = arith.mulf %a, %xi : f64
    %s  = arith.addf %p, %yi : f64
    memref.store %s, %Y[%i] : memref<?xf64>
  }
  return
}
