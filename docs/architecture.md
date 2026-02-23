# Lego MLIR Architecture

This document provides a detailed overview of the **Lego MLIR Dialect**, a specialized MLIR dialect designed for representing and transforming tensor layouts through composable primitives.

## 1. Overview

The Lego MLIR dialect (`lego`) provides a high-level abstraction for defining complex memory layouts and coordinate transformations. It is designed to be:

-   **Composable**: Layouts are built from small, reusable primitives.
-   **Invertible**: Every layout definition supports both forward (`apply`) and inverse (`apply_inverse`) mappings.
-   **Lowerable**: High-level layout operations can be compiled down to standard arithmetic (`arith`) and affine (`affine`) operations for efficient execution.

## 2. Directory Structure

The project follows a standard LLVM/MLIR project structure:

-   `include/Lego/`: Public headers and TableGen definitions.
    -   `LegoOps.td`: Defines the dialect operations.
    -   `Passes.td`: Defines the transformation passes.
    -   `LegoDialect.td`: Defines the Lego dialect itself.
-   `lib/Lego/`: Implementation of the dialect and passes.
    -   `LegoOps.cpp`: C++ implementation of operation logic (verifiers, printers).
    -   `LegoToArith.cpp`: Implementations of the lowering pass to standard arithmetic.
    -   `LegoNormalization.cpp`: Implementation of the normalization pass (simplifying sugar ops).
-   `python/`: Python bindings for the dialect.
-   `test/`: MLIR-based tests (`lit` tests).

## 3. Core Concepts

### 3.1. Layout Primitives

The core of the dialect is the `!lego.layout` type, which represents a mapping between an N-dimensional index space and a 1-dimensional flat index space.

#### `reg_p` (Regular Permutation)
The most fundamental primitive. It represents a reordering of dimensions followed by a flattening.
-   **Args**: `perm` (permutation vector), `dims` (input dimensions).
-   **Semantics**:
    -   **Apply**: Permutes input indices according to `perm`, then flattens them using mixed-radix packing.
    -   **Inverse**: Unflattens the index into the permuted shape, then applies the inverse permutation.

#### `gen_p` (Generic Permutation/Layout)
Allows strictly user-defined logic for layouts that cannot be expressed as simple permutations (e.g., antidiagonal layouts).
-   **Regions**:
    -   `apply`: Takes N indices, returns 1 flat index.
    -   `inv`: Takes 1 flat index, returns N indices.
-   **Verification**: The `lego-verify-consistency` pass checks if `apply` and `inv` are mathematically consistent.

**Example**:
```mlir
%layout = lego.gen_p [4, 4]
  apply {
  ^bb0(%i: index, %j: index):
    %sum = arith.addi %i, %j : index
    lego.yield %sum : index
  }
  inv {
  ^bb0(%flat: index):
    %c4 = arith.constant 4 : index
    %i = arith.divui %flat, %c4 : index
    %j = arith.remui %flat, %c4 : index
    lego.yield %i, %j : index, index
  } : !lego.layout
```

### 3.2. Combinators

Operators that combine existing layouts into more complex ones.

#### `order_by`
Sequences multiple layout blocks. It partitions the input indices among the provided sub-layouts.
-   **Args**: List of `!lego.layout` values.
-   **Semantics**:
    -   Splits the input indices: The first $k_1$ indices go to the first layout, the next $k_2$ to the second, etc.
    -   Computes logical flat indices for each sub-layout.
    -   Combines these logical indices using mixed-radix packing (stride is the product of the volume of subsequent layouts).

#### `group_by`
Groups input dimensions before applying sub-layouts. This allows treating a block of dimensions as a single unit or "digit" in a larger structure.
-   **Args**: `group_dims` (dimensions of the groups), `objects` (list of layouts).
-   **Semantics**:
    -   **Apply**: Iterates through `objects` in **reverse order**.
    -   **Inverse**: Iterates through `objects` in **forward order**.

### 3.3. Syntactic Sugar

These operations provide convenient shorthands for common patterns but are internally rewritten to primitives during the `lego-normalization` pass.

-   **`row`**: Creates a row-major layout.
    -   *Desugars to*: `RegP` with identity permutation `[0, 1, ..., N-1]`.
-   **`col`**: Creates a column-major layout.
    -   *Desugars to*: `RegP` with reversed permutation `[N-1, ..., 0]`.
-   **`tile_by`**: Applies a tiling transformation.
    -   *Desugars to*: A sequence of `GroupBy` and `OrderBy` operations that conceptually split and reorder dimensions to achieve tiling.

## 4. Compilation Flow

The typical compilation flow for Lego MLIR is:

1.  **Parsing/Construction**: User writes or generates Lego MLIR code (often via Python bindings).
2.  **Normalization (`lego-normalization`)**:
    -   Input: Mix of Primitives ("RegP") and Sugar ("TileBy", "Row").
    -   Action: Rewrites `TileBy`, `Row`, `Col` into explicit `GroupBy`, `OrderBy`, and `RegP` chains.
    -   Output: IR containing only `GroupBy`, `OrderBy`, `RegP`, `GenP`.
3.  **Lowering (`lego-to-arith`)**:
    -   Input: Lego operations (`apply`, `apply_inverse`) and definitions.
    -   Action: Expands `apply(layout, indices)` into a sequence of `arith.muli`, `arith.addi`, `arith.remui`, `arith.divui` operations.
    -   Output: Standard MLIR (Arith, SCF, Affine) without any `lego` operations (layout definitions become dead code or are removed).

## 5. Lowering Logic Details

### Flattening (Forward)
Implemented in `LegoToArith.cpp` via `flattenIndex`.
Formula: $Flat = i_N + d_N \times (i_{N-1} + d_{N-1} \times (...))$
This corresponds to a standard mixed-radix packing (generalized row-major).

### Unflattening (Inverse)
Implemented in `LegoToArith.cpp` via `unflattenIndex`.
Inverse of flattening logic using integer division and modulus:
For each dimension $k$ (from 0 to $N$):
-   $Stride = \prod_{j=k+1}^{N} d_j$
-   $i_k = Current / Stride$
-   $Current = Current \% Stride$

### Layout-Specific Lowering
-   **RegP**: Permutes indices $\rightarrow$ Flatten using permuted strides.
-   **OrderBy**: Slices indices for each sub-layout $\rightarrow$ Apply sub-layout $\rightarrow$ Combine results as if they were digits in a mixed-radix system.
-   **GroupBy**: Flattens input indices to a single intermediate $\rightarrow$ Unflattens into "object" domains $\rightarrow$ Applies layout objects recursively.

## 6. Integration

The dialect is designed to integrate with the Transform Dialect (`lego.transform_loop`) to allow Layout Transformations to be applied to other operations (like `linalg` ops) within a transformation script.

-   **`lego.transform_loop`**: A transform op that applies a Lego layout concept to a target operation, potentially tiling or reordering its loops.

## Python Bindings
## 7. Arithmetic Simplification & Optimization

The `lego-arith-simplification` pass implements specialized algebraic identities to optimize the arithmetic operations generated during lowering. These simplifications are crucial for generating clean and efficient code, especially for index calculations.

### key Algebraic Identities

The pass focuses on simplifying expressions involving integer division (`divui`) and remainder (`remui`), which are common in layout transformations.

1.  **Modulo Simplification**:
    -   Identity: `(q * d + r) % d -> r % d`
    -   Rationale: Removes unnecessary multiplication and addition when the term is a multiple of the divisor.

2.  **Division Simplification**:
    -   Identity: `(q * d + r) / d -> q + (r / d)`
    -   Rationale: Extracts the quotient component directly. In unsigned arithmetic, if $r < d$, this simplifies further to just $q$.

3.  **Constant Distributivity**:
    -   Identity: `(x + c) / d -> (x / d) + (c / d)` (when `c` is a multiple of `d`)
    -   Rationale: Distributes division over addition for constant terms that are multiples of the divisor.

4.  **Reconstruction**:
    -   Identity: `(x / d) * d + (x % d) -> x`
    -   Rationale: Reconstructs the original value from its quotient and remainder parts, reversing the decomposition.


### Interaction with Standard MLIR Canonicalizations

The `lego-arith-simplification` pass is designed to work in tandem with standard MLIR `arith` dialect canonicalizations.

-   **Complementary Roles**: Lego specific simplifications handle high-level algebraic identities that are often missed by standard passes because they require understanding the specific structure of index calculations (e.g., `(q*d + r)/d`).
-   **Cleanup**: Once Lego simplifications are applied, they often expose further opportunities for standard canonicalizations. For example, rewriting `(q * d + r) / d` to `q + (r / d)` allows standard canonicalization to fold `(r / d)` to `0` if it can prove `r < d` (e.g. from known ranges or constants).
-   **Pipeline Order**: It is recommended to run `lego-arith-simplification` before standard `canonicalize` passes to maximize the simplification impact.

### Integer Range Optimizations (`--int-range-optimizations`)

The standard MLIR pass `--int-range-optimizations` is highly recommended in the pipeline:

-   **Value Range Analysis**: It performs data-flow analysis to determine the possible range of values for each SSA value.
-   **Dead Code Elimination**: By proving that certain conditions are always false (e.g., `remui %x, %c` where `%x < %c` implies result is `%x`), it can eliminate dead code branches or simplify operations.
-   **Synergy**: Lego's simplifications (like `(q*d+r)/d -> q`) often rely on the assumption that `r < d`. While `lego-arith-simplification` performs the algebraic rewrite, `--int-range-optimizations` can provide the proof that `r` is within bounds, enabling further folding of any residual terms.
