#ifndef LEGO_OPS
#define LEGO_OPS

include "LegoDialect.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LEGO Dialect Operations
//===----------------------------------------------------------------------===//

class Lego_Op<string mnemonic, list<Trait> traits = []> :
    Op<Lego_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Layout Primitives
//===----------------------------------------------------------------------===//

def YieldOp : Lego_Op<"yield", [Pure, Terminator,
    ParentOneOf<["GenPOp"]>]> {
  let summary = "Yield a value from a lego region";
  let description = [{
    Terminates a region in `lego.gen_p`, yielding the computed flat index.
  }];
  let arguments = (ins Variadic<Index>:$values);
  let assemblyFormat = "$values attr-dict `:` type($values)";
  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>
  ];
}

def GenPOp : Lego_Op<"gen_p", [Pure,
    SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Generic Layout Block with User-Defined Logic";
  let description = [{
    Crucial for complex layouts like `antidiag` where the transformation cannot be expressed
    by a simple affine map. The region contains the Logic to compute the transformed indices.
    
    The region takes the input indices as arguments and yields the transformed flat index
    using `lego.yield`.

    Example:
    ```mlir
    %layout = lego.gen_p [4, 4] ({
    ^bb0(%i: index, %j: index):
      %c1 = arith.constant 1 : index
      %sum = arith.addi %i, %j : index
      lego.yield %sum : index
    }) : !lego.layout
    ```
  }];
  let arguments = (ins I64ArrayAttr:$dims);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Lego_LayoutType:$result);
  let assemblyFormat = "$dims $body attr-dict `:` type($result)";
}

def RegPOp : Lego_Op<"reg_p", [Pure]> {
  let summary = "Regular Permutation Block";
  let description = [{
    Creates a layout block defined by a permutation vector.
    Equivalent to reordering the dimensions according to `perm`.
  }];
  let arguments = (ins I64ArrayAttr:$perm, I64ArrayAttr:$dims);
  let results = (outs Lego_LayoutType:$result);
  let assemblyFormat = "`perm` $perm `dims` $dims attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Combinators and Higher-Level Layouts
//===----------------------------------------------------------------------===//

def OrderByOp : Lego_Op<"order_by", [Pure]> {
  let summary = "Applies a sequence of permutation blocks";
  let description = [{
    Represents a layout composed of a sequence of permutation blocks.
    Each block consumes a subset of the input indices and produces a flat sub-index.
    The final flat index is computed by combining sub-indices in mixed-radix form.

    Example: `lego.order_by(%l1, %l2) : !lego.layout`
  }];
  let arguments = (ins Variadic<Lego_LayoutType>:$perms);
  let results = (outs Lego_LayoutType:$result);
  let assemblyFormat = "`(` $perms `)` attr-dict `:` type($result)";
}

def GroupByOp : Lego_Op<"group_by", [Pure]> {
  let summary = "Groups dimensions and applies a sequence of layout objects";
  let description = [{
    Groups the input dimensions according to `group_dims` and then applies the
    sequence of layout objects. During apply, objects are iterated in reverse:
    for each object, unflatten the current flat index into the object's dims,
    then apply the object. During inv, objects are iterated forward.

    Example: `lego.group_by [4, 5](%ob1, %ob2) : !lego.layout`
  }];
  let arguments = (ins I64ArrayAttr:$group_dims, Variadic<Lego_LayoutType>:$objects);
  let results = (outs Lego_LayoutType:$result);
  let assemblyFormat = "$group_dims `(` $objects `)` attr-dict `:` type($result)";
}

def TileByOp : Lego_Op<"tile_by", [Pure]> {
    let summary = "Creates a Tiled layout";
    let arguments = (ins Lego_LayoutType:$input, I64ArrayAttr:$tile_sizes);
    let results = (outs Lego_LayoutType:$result);
    let assemblyFormat = "$input `sizes` $tile_sizes attr-dict `:` type($result)";
}

def RowOp : Lego_Op<"row", [Pure]> {
    let summary = "Creates a Row-Major layout (identity permutation)";
    let description = [{
      Row(*dims) = RegP(dims, [0, 1, ..., d-1]).
      Sugar for RegP with identity permutation.
    }];
    let arguments = (ins I64ArrayAttr:$dims);
    let results = (outs Lego_LayoutType:$result);
    let assemblyFormat = "$dims attr-dict `:` type($result)";
}

def ColOp : Lego_Op<"col", [Pure]> {
    let summary = "Creates a Column-Major layout (reversed permutation)";
    let description = [{
      Col(*dims) = RegP(dims, [d-1, d-2, ..., 0]).
      Sugar for RegP with reversed identity permutation.
    }];
    let arguments = (ins I64ArrayAttr:$dims);
    let results = (outs Lego_LayoutType:$result);
    let assemblyFormat = "$dims attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Application Operations (Runtime)
//===----------------------------------------------------------------------===//

def ApplyOp : Lego_Op<"apply", [Pure]> {
  let summary = "Applies a layout to indices to get a flat index";
  let description = [{
    Given a layout and a set of indices, returns the flattened index.
    This corresponds to `layout.apply(idx)`.
  }];
  let arguments = (ins Lego_LayoutType:$layout, Variadic<Index>:$indices);
  let results = (outs Index:$flat_index);
  let assemblyFormat = "$layout `(` $indices `)` attr-dict `:` type($layout)";
}

def ApplyInverseOp : Lego_Op<"apply_inverse", [Pure]> {
  let summary = "Applies the inverse of a layout to a flat index";
  let description = [{
    Given a layout and a flat index, returns the original multi-dimensional indices.
    This corresponds to `layout.inv(flat_idx)`.
  }];
  let arguments = (ins Lego_LayoutType:$layout, Index:$flat_index);
  let results = (outs Variadic<Index>:$indices);
  let assemblyFormat = "$layout `(` $flat_index `)` attr-dict `:` type($layout) `->` type($indices)";
}

//===----------------------------------------------------------------------===//
// LEGO Transform Operations (Transform Dialect)
//===----------------------------------------------------------------------===//

def ApplyLayoutTransformOp : Op<Lego_Dialect, "lego.transform_loop",
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
