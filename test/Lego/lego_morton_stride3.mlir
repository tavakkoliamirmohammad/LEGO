// RUN: lego-opt %s --lego-strength-reduction | FileCheck %s

// Stride-3 (3-D Morton) bit-spread recogniser test.
//
// The matcher in lib/Lego/LegoArithSimplification.cpp (RecognizeBitSpread)
// detects a per-bit decomposition of the form
//     out = sum over i of:  ((src >> i) & 1) << (offset + stride*i)
// and rewrites it to the Hacker's Delight bit-magic emit.  Stride 2 has
// long been supported (2-D Morton); this file pins down stride 3 (3-D).
//
// The kernel below spreads 3 bits (positions 0..2 of %x) to lane 1 of a
// 3-D Morton index — i.e. the y-coordinate in (x,y,z) Morton — so the
// output bits land at positions 1, 4, 7.  After the rewrite the body
// must contain the stride-3 stage-mask emit (constants 0xC30C30C3 and
// 0x49249249) and NOT the stride-2 mask (0x55555555).

// CHECK-LABEL: func.func @morton_stride3_lane0
// 0xC30C30C3 = -1022611261  (signed i32 representation of 3272356035)
// 0x49249249 =  1227133513
// CHECK: arith.constant 1227133513
// CHECK: arith.constant -1022611261
// CHECK-NOT: arith.constant 1431655765   // 0x55555555 (stride-2 stage)
func.func @morton_stride3_lane0(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c6 = arith.constant 6 : i32
  %c9 = arith.constant 9 : i32
  // bit 0 of x → position 0
  %b0  = arith.andi %x, %c1 : i32
  // bit 1 of x → position 3
  %sx1 = arith.shrui %x, %c1 : i32
  %b1  = arith.andi %sx1, %c1 : i32
  %t1  = arith.shli %b1, %c3 : i32
  // bit 2 of x → position 6
  %sx2 = arith.shrui %x, %c2 : i32
  %b2  = arith.andi %sx2, %c1 : i32
  %t2  = arith.shli %b2, %c6 : i32
  // bit 3 of x → position 9
  %sx3 = arith.shrui %x, %c3 : i32
  %b3  = arith.andi %sx3, %c1 : i32
  %t3  = arith.shli %b3, %c9 : i32
  %s0  = arith.addi %b0, %t1 : i32
  %s1  = arith.addi %s0, %t2 : i32
  %s2  = arith.addi %s1, %t3 : i32
  return %s2 : i32
}
