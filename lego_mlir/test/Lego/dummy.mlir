// RUN: lego-opt %s | FileCheck %s

module {
  // CHECK: %layout = lego.row 128, 128 : !lego.layout
  %layout = lego.row 128, 128 : !lego.layout
}
