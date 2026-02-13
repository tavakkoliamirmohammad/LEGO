// RUN: lego-opt %s | FileCheck %s

module {
  // CHECK: "lego.row"
  %0 = "lego.row"() {n = 128 : i32, m = 128 : i32} : () -> !lego.layout
}
