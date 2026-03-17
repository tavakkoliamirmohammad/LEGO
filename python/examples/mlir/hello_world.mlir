// Run: lego-opt hello_world.mlir -lego-lower
func.func @hello_world(%i: index, %j: index) -> index {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %r = lego.row [%c4, %c8] : !lego.layout
  %f = lego.apply %r(%i, %j) : !lego.layout
  return %f : index
}
