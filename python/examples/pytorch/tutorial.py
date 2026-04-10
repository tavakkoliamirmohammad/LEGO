"""
LEGO PyTorch Integration Tutorial
==================================

LEGO is a layout algebra compiler. In plain terms: it lets you change HOW
your tensors are stored in memory -- column-major, tiled, Z-curve, swizzled --
without rewriting your model. Why does that matter?

The Problem
-----------
GPU memory is accessed in 128-byte cache lines. If your matmul reads matrix A
row-by-row but A is stored column-by-column, every load is a cache miss. The
standard fix is to manually transpose, tile, or swizzle your data -- but that
means rewriting your indexing, your autograd, your serialization, everything.

What LEGO Does
--------------
LEGO separates the *logical* view of your data (what your math sees) from the
*physical* layout (how bytes sit in memory). You annotate a tensor with a
layout, and LEGO handles the index remapping transparently through:

  - Pointwise ops (add, relu, etc.) -- work directly on physical data (zero cost)
  - Reductions (sum, mean) -- automatically inverse-remap when needed
  - Matmul (mm, bmm, addmm) -- Triton kernel reads through the layout's index table
  - torch.compile -- layout metadata propagates through the FX graph

The net effect: you get better memory access patterns with ONE line of code.

Run this tutorial:
    cd LEGO/python
    python examples/pytorch/tutorial.py
"""

import torch
import numpy as np
import lego
from lego.frontends.python_mlir import ColMajor, RowMajor, TiledPermute


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ============================================================================
# 1. Virtual annotation -- zero-cost layout tagging
# ============================================================================

section("1. Virtual Annotation (metadata only, no data movement)")

x = torch.randn(4, 4)
layout = ColMajor((4, 4))

# annotate() attaches layout metadata. No data is copied or moved.
ax = lego.annotate(x, layout)

print(f"Original tensor:  type={type(x).__name__}, data_ptr={x.data_ptr()}")
print(f"Annotated tensor: type={type(ax).__name__}, data_ptr={ax._data.data_ptr()}")
print(f"Same memory? {x.data_ptr() == ax._data.data_ptr()}")
print(f"Layout: {ax.lego_layout}")
print()

# All standard ops work -- layout propagates through pointwise ops:
y = torch.relu(ax) + ax * 2.0
print(f"After relu + mul: still a LegoTensor? {isinstance(y, lego.LegoTensor)}")
print(f"Layout preserved? {y.lego_layout is layout}")


# ============================================================================
# 2. Physical rearrangement -- data movement with autograd
# ============================================================================

section("2. Physical Rearrangement (actual data permutation)")

x = torch.arange(16, dtype=torch.float32).reshape(4, 4)
print(f"Original (row-major):\n{x}\n")

# rearrange() physically permutes the data to column-major order.
layout = ColMajor((4, 4))
rx = lego.rearrange(x, layout)

print(f"After rearrange to ColMajor:")
print(f"  Physical data (what's in memory):\n  {rx._data}\n")
print(f"  Notice: column 0 of original [0,4,8,12] is now row 0 of physical data.")
print(f"  This means sequential memory reads now traverse columns -- better for")
print(f"  column-reduction kernels.\n")

# Pointwise ops work directly on physical data (no remapping needed):
doubled = rx * 2.0
print(f"rx * 2.0 works on physical data directly: {isinstance(doubled, lego.LegoTensor)}")

# Reductions automatically inverse-remap when needed:
col_sum = torch.sum(rx, dim=0)
expected = torch.sum(x, dim=0)
print(f"sum(rx, dim=0) = {col_sum}")
print(f"sum(x,  dim=0) = {expected}")
print(f"Correct? {torch.allclose(col_sum, expected)}")

# Autograd works through nn.Module parameters (the main use case):
linear = torch.nn.Linear(4, 2, bias=False)
ax_grad = lego.annotate(torch.randn(4, 4), layout)
loss = linear(ax_grad).sum()
loss.backward()
print(f"\nGradient flows to model params: linear.weight.grad exists? {linear.weight.grad is not None}")
print(f"Gradient shape: {linear.weight.grad.shape}")


# ============================================================================
# 3. Tiled layouts -- the real win for GPU kernels
# ============================================================================

section("3. Tiled Layouts (the real performance win)")

# TiledPermute stores data in 8x8 tiles. Within each tile, elements are
# contiguous in memory. This means a GPU thread block loading one tile
# gets perfect cache locality.

shape = (32, 32)
tile = (8, 8)
layout = TiledPermute(shape, tile_shape=tile)

x = torch.randn(*shape)
rx = lego.rearrange(x, layout)

print(f"Shape: {shape}, Tile: {tile}")
print(f"Original: 32x32 row-major -- a warp reading an 8x8 block touches 8 cache lines")
print(f"Tiled:    32x32 tiled 8x8 -- a warp reading an 8x8 block touches 1 cache line")
print(f"")

# Annotate (virtual) preserves correctness for all ops including matmul:
ax = lego.annotate(x, layout)
b = torch.randn(32, 16)
result = torch.mm(ax, b)
expected = torch.mm(x, b)
print(f"mm(annotated_A, B) correct? {torch.allclose(result, expected, atol=1e-5)}")
print(f"On GPU, lego::mm dispatches to a Triton kernel that reads through")
print(f"the layout's index table -- so even physical rearrangements work.")


# ============================================================================
# 4. Triton-accelerated matmul on GPU
# ============================================================================

section("4. Layout-Aware Triton Matmul (GPU)")

if torch.cuda.is_available():
    try:
        import triton
        from lego.torch.triton_kernels import triton_lego_mm

        M, K, N = 256, 128, 256
        layout = TiledPermute((M, K), tile_shape=(32, 32))

        a_data = torch.randn(M, K, device="cuda")
        b = torch.randn(K, N, device="cuda")

        # Standard matmul
        expected = torch.mm(a_data, b)

        # LEGO: rearrange A to tiled layout, then matmul with layout-aware kernel
        from lego.backend.compiler import LayoutCompiler
        compiler = LayoutCompiler(layout._layout, layout._shape, "i64")
        fwd, _ = compiler.get_permutation_table()
        fwd_t = torch.from_numpy(np.ascontiguousarray(fwd)).to("cuda")
        a_tiled = a_data.reshape(-1)[fwd_t].reshape(M, K)

        result = triton_lego_mm(a_tiled, b, a_layout=layout)

        print(f"Standard mm:    shape={expected.shape}")
        print(f"LEGO Triton mm: shape={result.shape}")
        max_err = (result - expected).abs().max().item()
        print(f"Max error: {max_err:.6f}")
        print(f"Correct (atol=0.05)? {torch.allclose(result, expected, atol=0.05)}")
        print(f"(Tiled Triton accumulation has slightly different FP rounding than cuBLAS)")
        print()
        print(f"The Triton kernel reads A through the layout's inverse permutation")
        print(f"table, so it sees logical-order data even though A is physically tiled.")
        print(f"In a real workload, the tiled layout means the kernel's memory access")
        print(f"pattern has better spatial locality.")

    except ImportError:
        print("Triton not installed -- skipping GPU demo.")
        print("Install with: pip install triton")
else:
    print("No CUDA GPU available -- skipping GPU demo.")
    print("The Triton kernel runs on NVIDIA GPUs with Triton installed.")


# ============================================================================
# 5. torch.compile integration
# ============================================================================

section("5. torch.compile(backend='lego')")

import lego.torch.compile  # registers the backend

layout = ColMajor((8, 8))
x = torch.randn(8, 8)

def model(a):
    y = torch.relu(a)
    y = y * 2.0
    return y + a

compiled = torch.compile(model, backend="lego")
result = compiled(lego.annotate(x, layout))
expected = model(x)

print(f"Eager result matches compiled? {torch.allclose(result, expected)}")
print(f"Layout metadata propagates through the FX graph,")
print(f"so inductor can generate layout-aware code.")


# ============================================================================
# 6. Works with nn.Module -- no code changes needed
# ============================================================================

section("6. Drop-in with nn.Module")

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MLP()
layout = RowMajor((16, 32))

# Just annotate the input -- the model code doesn't change at all
x = lego.annotate(torch.randn(16, 32), layout)
y = model(x)
loss = y.sum()
loss.backward()

print(f"Input: LegoTensor with {layout}")
print(f"Output shape: {y.shape}")
print(f"Gradients computed: fc1.weight.grad exists? {model.fc1.weight.grad is not None}")
print(f"                    fc2.weight.grad exists? {model.fc2.weight.grad is not None}")
print()
print(f"The model code is UNCHANGED. LEGO handles layout propagation,")
print(f"dispatch to layout-aware ops, and gradient flow automatically.")


# ============================================================================
# 7. Autotune -- find the best layout automatically
# ============================================================================

section("7. Autotune")

from lego.torch.autotune import autotune

layout = autotune(
    shape=(128, 128),
    tile_candidates=[(8, 8), (16, 16), (32, 32), (64, 64)],
    device="cpu",
    n_iters=10,
)

print(f"Best layout for (128, 128): {layout}")
print(f"Tile shape: {layout._shape}")
print()
print(f"autotune benchmarks each tile configuration and picks the fastest.")
print(f"On GPU, it uses proper CUDA synchronization for accurate timing.")


# ============================================================================
# Summary
# ============================================================================

section("Summary: What LEGO Gives You")

print("""
  1. ONE LINE to change memory layout:
       rx = lego.rearrange(tensor, TiledPermute((M,N), tile_shape=(64,64)))

  2. ALL OPS JUST WORK -- pointwise, reductions, matmul, autograd, compile

  3. TRITON KERNELS on GPU -- layout-aware matmul reads through permutation
     tables for correct results on physically-rearranged data

  4. ZERO MODEL CHANGES -- annotate inputs, everything else propagates

  5. AUTOTUNE -- grid search over tile sizes to find the optimal layout

  The pitch: you get the memory access patterns of hand-tuned CUDA kernels
  with the ergonomics of standard PyTorch.
""")
