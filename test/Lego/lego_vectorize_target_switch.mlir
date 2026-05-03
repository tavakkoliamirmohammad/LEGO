// RUN: lego-opt %s --lego-vectorize='target=neon'   | FileCheck %s --check-prefix=NEON
// RUN: lego-opt %s --lego-vectorize='target=avx2'   | FileCheck %s --check-prefix=AVX2
// RUN: lego-opt %s --lego-vectorize='target=avx512' | FileCheck %s --check-prefix=AVX512

// R15: portability sanity check — same MLIR input, three different vector widths.
// The vectorizer reads target from the pass option; no AVX-specific widths hardcoded.
// All decisions go through getRegisterLanesForType(target, elemBytes).
//
// f64 (8 bytes): neon=2 lanes, avx2=4 lanes, avx512=8 lanes.

// NEON-LABEL: func.func @target_switch
// NEON: vector<2xf64>

// AVX2-LABEL: func.func @target_switch
// AVX2: vector<4xf64>

// AVX512-LABEL: func.func @target_switch
// AVX512: vector<8xf64>

func.func @target_switch(%A: memref<1024xf64>, %B: memref<1024xf64>) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1024 step %c1 {
    %v = memref.load %A[%i] : memref<1024xf64>
    memref.store %v, %B[%i] : memref<1024xf64>
  }
  return
}
