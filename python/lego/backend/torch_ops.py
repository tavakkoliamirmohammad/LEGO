"""
LEGO PyTorch Integration

Provides autograd-compatible layout transforms via torch.autograd.Function
and a torch.library custom op for torch.compile support.
"""

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:

    class _LegoPermuteFunction(torch.autograd.Function):
        """Autograd function for LEGO layout permutations.

        Forward: applies perm to flatten the input.
        Backward: applies inv_perm (since layouts are bijective).

        For forward transforms, call with (tensor, fwd_perm, inv_perm).
        For inverse transforms, call with (tensor, inv_perm, fwd_perm).
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

    # Backward compat aliases
    LegoTransformFunction = _LegoPermuteFunction
    LegoInverseTransformFunction = _LegoPermuteFunction

    # ========================================================================
    # torch.library custom op for torch.compile support
    # ========================================================================

    torch.library.define("lego::permute", "(Tensor x, Tensor perm) -> Tensor")

    def _permute_impl(x, perm):
        return x.contiguous().view(-1)[perm].view(x.shape)

    torch.library.impl("lego::permute", "cpu")(_permute_impl)
    torch.library.impl("lego::permute", "cuda")(_permute_impl)

    @torch.library.register_fake("lego::permute")
    def _permute_fake(x, perm):
        return x.new_empty(x.shape)

else:
    _LegoPermuteFunction = None
    LegoTransformFunction = None
    LegoInverseTransformFunction = None
