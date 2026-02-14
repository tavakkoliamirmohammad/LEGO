# LEGO MLIR Dialect

The LEGO (Layouts for Efficient GPU Optimization) dialect provides operations to define and apply data layout transformations. It treats layouts as first-class citizens in MLIR, enabling complex index manipulations through a composable and high-level abstraction.

## Overview

The core idea of the LEGO dialect is to separate the *definition* of a layout from its *application*. A layout defines a bidirectional mapping between N-dimensional logical indices and a 1-dimensional physical (flat) index.

- **Forward Map (`apply`)**: $(i_0, i_1, \dots, i_N) \to \text{flat\_index}$
- **Inverse Map (`apply_inverse`)**: $\text{flat\_index} \to (i_0, i_1, \dots, i_N)$

Layouts can be composed, nested, and tiled to express sophisticated memory access patterns optimized for GPUs.

## Operations

### Layout Primitives

#### `lego.row`
Creates a row-major layout (standard C-style).
```mlir
%layout = lego.row [10, 20] : !lego.layout
```
This is syntactic sugar for `lego.reg_p` with an identity permutation `[0, 1]`.

#### `lego.col`
Creates a column-major layout (Fortran-style).
```mlir
%layout = lego.col [10, 20] : !lego.layout
```
This is syntactic sugar for `lego.reg_p` with a reversed permutation `[1, 0]`.

#### `lego.reg_p` (Regular Permutation)
Defines a layout by permuting the input dimensions.
```mlir
// Transpose: (i, j) -> (j, i)
%layout = lego.reg_p perm [1, 0] dims [10, 20] : !lego.layout
```
The flattening logic follows the permuted dimensions.

#### `lego.gen_p` (Generic Layout)
Allows defining arbitrary forward and inverse logic using regions. This is useful for layouts that cannot be expressed as simple permutations, such as anti-diagonal layouts.
```mlir
%layout = lego.gen_p [4, 4]
  apply (%i: index, %j: index) {
    %sum = arith.addi %i, %j : index
    lego.yield %sum : index
  }
  inv (%flat: index) {
    // Inverse logic yielding 2 indices
    %c4 = arith.constant 4 : index
    %i = arith.divui %flat, %c4 : index
    %j = arith.remui %flat, %c4 : index
    lego.yield %i, %j : index, index
  } : !lego.layout
```

### Layout Combinators

#### `lego.order_by`
sequences a list of layout objects. It combines them in a mixed-radix fashion.
```mlir
%l1 = lego.row [4] : !lego.layout
%l2 = lego.row [8] : !lego.layout
%combined = lego.order_by(%l1, %l2) : !lego.layout
```
The first layout acts as the least significant chunk of the flat index.

#### `lego.group_by`
Groups input dimensions and applies a sequence of layout objects to them.
```mlir
// Input dims: (d0, d1, d2, d3) -> grouped into [16, 32]
%layout = lego.group_by [16, 32] (%obj1, %obj2) : !lego.layout
```
- **Apply (Forward)**: Iterates objects in **reverse** order. Unflattens the current flat index into the object's dimensions, then applies the object.
- **Inverse**: Iterates objects in **forward** order. Applies the inverse of the object, then reflattens.

#### `lego.tile_by`
A high-level tiling operation that desugars into `lego.group_by` and `lego.order_by` structures.
```mlir
%base = lego.row [128, 128] : !lego.layout
// Tiles the base layout into tiles of size [16, 16] with outer dims [8, 8]
%tiled = lego.tile_by %base tile_dims [[16, 16], [8, 8]] : !lego.layout
```
This operation is automatically desugared by the `LegoDesugarPass`.

### Application Operations

#### `lego.apply`
Computes the flat index for a given set of N-D indices using the specified layout.
```mlir
%flat = lego.apply %layout (%i, %j) : !lego.layout
```

#### `lego.apply_inverse`
Computes the N-D indices from a flat index using the specified layout.
```mlir
%i, %j = lego.apply_inverse %layout (%flat) : !lego.layout -> index, index
```

### Transform Operations

#### `lego.transform_loop`
Used within the transform dialect to apply a layout to a target operation (e.g., a loop nest or `linalg.matmul`).

## Compilation Pipeline

1.  **Desugaring (`LegoDesugarPass`)**:
    -   `lego.row` $\to$ `lego.reg_p` (identity perm)
    -   `lego.col` $\to$ `lego.reg_p` (reversed perm)
    -   `lego.tile_by` $\to$ Complex chain of `lego.group_by`, `lego.order_by`, and reshuffling `lego.reg_p` ops.

2.  **Lowering to Arith (`LegoToArithPass`)**:
    -   `lego.apply` is lowered to a sequence of `arith.muli` and `arith.addi`.
    -   `lego.apply_inverse` is lowered to a sequence of `arith.divui` and `arith.remui`.
    -   `lego.gen_p` bodies are inlined.

## Semantics

### Flattening
A multi-dimensional index $(i_0, \dots, i_N)$ with dimensions $(D_0, \dots, D_N)$ is flattened to:
$$ \text{flat} = \sum_{k=0}^{N} \left( i_k \times \prod_{j=k+1}^{N} D_j \right) $$
This is a standard row-major packing where the last dimension is contiguous.

### Unflattening
A flat index is unflattened by computing moduli and divisions in reverse order of the stride products.
