// RUN: lego-opt %s | FileCheck %s

module {
  // CHECK: %layout = lego.row 128 : i32, 128 : i32 : !lego.layout
  %layout = lego.row 128 : i32, 128 : i32 : !lego.layout
}
