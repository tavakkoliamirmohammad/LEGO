#ifndef LEGO_OPS
#define LEGO_OPS

include "LegoDialect.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// LEGO Dialect Operations
//===----------------------------------------------------------------------===//

class Lego_Op<string mnemonic, list<Trait> traits = []> :
    Op<Lego_Dialect, mnemonic, traits>;

def RowOp : Lego_Op<"row", [Pure]> {
    let summary = "Creates a Row-Major layout";
    let arguments = (ins I32Attr:$n, I32Attr:$m);
    let results = (outs Lego_LayoutType:$result);
    // TODO: Add assembly format
}

def TileByOp : Lego_Op<"tile_by", [Pure]> {
    let summary = "Creates a Tiled layout";
    let arguments = (ins I32ArrayAttr:$tensor_dims, I32ArrayAttr:$tile_sizes);
    let results = (outs Lego_LayoutType:$result);
}

//===----------------------------------------------------------------------===//
// LEGO Transform Operations
//===----------------------------------------------------------------------===//

def ApplyLayoutOp : Op<Lego_Dialect, "lego.apply_layout",
    [TransformOpInterface,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Applies a LEGO layout to a target operation";
  let description = [{
    Applies the specified LEGO layout to the target operation.
    The target operation must be compatible with the layout (e.g., a linalg.matmul).
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       Lego_LayoutType:$layout);
  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat = "$target `,` $layout attr-dict `:` `(` type($target) `,` type($layout) `)` `->` type($transformed)";
  
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure apply(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::transform::TransformResults &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // LEGO_OPS
