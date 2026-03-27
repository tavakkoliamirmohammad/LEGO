"""
LEGO PyTorch Integration

Provides autograd-compatible layout transforms via torch.autograd.Function
and a torch.library custom op for torch.compile support.
Also provides strided permutation representation for RegP-based layouts.
"""

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ============================================================================
# Permutation Representations
# ============================================================================

class PermutationRepr:
    """Base class for permutation representations."""

    def apply_numpy(self, arr):
        """Apply permutation to a flat numpy array: output[i] = arr[perm[i]]."""
        raise NotImplementedError

    def apply_inverse_numpy(self, arr):
        """Apply inverse permutation to a flat numpy array."""
        raise NotImplementedError

    def to_dense(self):
        """Return (fwd, inv) as dense int64 numpy arrays."""
        raise NotImplementedError


class DensePermutation(PermutationRepr):
    """Standard O(numel) dense permutation table."""

    def __init__(self, fwd, inv):
        self._fwd = np.ascontiguousarray(fwd, dtype=np.int64)
        self._inv = np.ascontiguousarray(inv, dtype=np.int64)

    def apply_numpy(self, arr):
        return arr.ravel()[self._fwd]

    def apply_inverse_numpy(self, arr):
        return arr.ravel()[self._inv]

    def to_dense(self):
        return (self._fwd, self._inv)

    @property
    def fwd(self):
        return self._fwd

    @property
    def inv(self):
        return self._inv


class StridedPermutation(PermutationRepr):
    """O(rank) strided permutation for RegP-based layouts.

    For a RegP with perm_vector, the permutation is a linear function
    of multi-dimensional indices. Instead of storing O(numel) int64 array,
    we store (shape, perm_vector) and compute indices on the fly.

    The forward permutation maps logical flat index -> physical flat index
    by unflatting, permuting axes, and reflatting.
    """

    def __init__(self, shape, perm_vector):
        self._shape = tuple(shape)
        self._perm_vector = list(perm_vector)
        self._numel = 1
        for s in shape:
            self._numel *= int(s)
        # Precompute strides for both orderings
        self._logical_strides = self._compute_strides(self._shape)
        permuted_shape = tuple(self._shape[p] for p in self._perm_vector)
        self._physical_strides = self._compute_strides(permuted_shape)
        # Inverse perm vector
        self._inv_perm_vector = [0] * len(self._perm_vector)
        for i, p in enumerate(self._perm_vector):
            self._inv_perm_vector[p] = i

    @staticmethod
    def _compute_strides(shape):
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * int(shape[i + 1])
        return strides

    def _unflatten(self, flat_idx, shape, strides):
        """Convert flat index to multi-dim indices."""
        indices = []
        rem = flat_idx
        for s, st in zip(shape, strides):
            indices.append(rem // st)
            rem = rem % st
        return indices

    def apply_flat(self, flat_idx):
        """Compute physical flat index from logical flat index."""
        indices = self._unflatten(flat_idx, self._shape, self._logical_strides)
        permuted = [indices[p] for p in self._perm_vector]
        permuted_shape = tuple(self._shape[p] for p in self._perm_vector)
        result = 0
        for idx, st in zip(permuted, self._physical_strides):
            result += idx * st
        return result

    def apply_inverse_flat(self, flat_idx):
        """Compute logical flat index from physical flat index."""
        permuted_shape = tuple(self._shape[p] for p in self._perm_vector)
        indices = self._unflatten(flat_idx, permuted_shape, self._physical_strides)
        unpermuted = [indices[self._inv_perm_vector[i]] for i in range(len(self._shape))]
        result = 0
        for idx, st in zip(unpermuted, self._logical_strides):
            result += idx * st
        return result

    def apply_numpy(self, arr):
        fwd, _ = self.to_dense()
        return arr.ravel()[fwd]

    def apply_inverse_numpy(self, arr):
        _, inv = self.to_dense()
        return arr.ravel()[inv]

    def to_dense(self):
        """Materialize the full dense permutation tables."""
        fwd = np.empty(self._numel, dtype=np.int64)
        inv = np.empty(self._numel, dtype=np.int64)
        for i in range(self._numel):
            j = self.apply_flat(i)
            fwd[i] = j
            inv[j] = i
        return (fwd, inv)

    @property
    def shape(self):
        return self._shape

    @property
    def perm_vector(self):
        return self._perm_vector


if _HAS_TORCH:

    class _LegoPermuteFunction(torch.autograd.Function):
        """Autograd function for LEGO layout permutations.

        Forward: applies gather-based permutation ``output[i] = input[perm[i]]``.
        Backward: applies the inverse permutation (since layouts are bijective).

        Usage:
          For layout transform (logical->physical):
            call with (tensor, fwd_perm, inv_perm)
          For inverse transform (physical->logical):
            call with (tensor, inv_perm, fwd_perm)

        Invariant: inv_perm[fwd_perm[i]] == i for all i.
        """

        @staticmethod
        def forward(ctx, input_tensor, perm, inv_perm):
            ctx.save_for_backward(inv_perm)
            flat = input_tensor.contiguous().view(-1)
            return flat[perm].view(input_tensor.shape)

        @staticmethod
        def backward(ctx, grad_output):
            inv_perm, = ctx.saved_tensors
            flat_grad = grad_output.contiguous().view(-1)
            return flat_grad[inv_perm].view(grad_output.shape), None, None

    # ========================================================================
    # torch.library custom ops for torch.compile support
    # ========================================================================

    torch.library.define("lego::permute", "(Tensor x, Tensor perm) -> Tensor")

    def _permute_impl(x, perm):
        return x.contiguous().view(-1)[perm].view(x.shape)

    torch.library.impl("lego::permute", "cpu")(_permute_impl)
    torch.library.impl("lego::permute", "cuda")(_permute_impl)

    @torch.library.register_fake("lego::permute")
    def _permute_fake(x, perm):
        return x.new_empty(x.shape)

    # ========================================================================
    # Autograd for lego::permute custom op (4E)
    # ========================================================================

    def _permute_backward(ctx, grad_output):
        perm, = ctx.saved_tensors
        numel = perm.shape[0]
        inv_perm = torch.empty_like(perm)
        inv_perm.scatter_(0, perm, torch.arange(numel, device=perm.device))
        flat_grad = grad_output.contiguous().view(-1)
        return flat_grad[inv_perm].view(grad_output.shape), None

    def _permute_setup_context(ctx, inputs, output):
        x, perm = inputs
        ctx.save_for_backward(perm)

    torch.library.register_autograd(
        "lego::permute", _permute_backward, setup_context=_permute_setup_context
    )

    # ========================================================================
    # Strided permute custom op
    # ========================================================================

    torch.library.define(
        "lego::permute_strided",
        "(Tensor x, int[] shape, int[] perm_vector) -> Tensor"
    )

    def _permute_strided_impl(x, shape, perm_vector):
        """Apply RegP-style strided permutation without materializing table."""
        # Use reshape + permute which is zero-copy for contiguous tensors
        md = x.contiguous().view(shape)
        return md.permute(perm_vector).contiguous().view(x.shape)

    torch.library.impl("lego::permute_strided", "cpu")(_permute_strided_impl)
    torch.library.impl("lego::permute_strided", "cuda")(_permute_strided_impl)

    @torch.library.register_fake("lego::permute_strided")
    def _permute_strided_fake(x, shape, perm_vector):
        return x.new_empty(x.shape)

else:
    _LegoPermuteFunction = None
