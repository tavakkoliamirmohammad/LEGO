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

    class LegoTransformFunction(torch.autograd.Function):
        """Autograd function for LEGO layout transforms.

        Forward: applies the layout transformation via permutation table.
        Backward: applies the inverse (since layouts are bijective).
        """

        @staticmethod
        def forward(ctx, input_tensor, compiler, fwd_perm=None, inv_perm=None):
            ctx.compiler = compiler
            device = input_tensor.device

            if fwd_perm is not None:
                fwd_t = fwd_perm
                inv_t = inv_perm
            else:
                fwd_np, inv_np = compiler.get_permutation_table()
                fwd_t = torch.from_numpy(fwd_np).to(device)
                inv_t = torch.from_numpy(inv_np).to(device)

            ctx.save_for_backward(inv_t)
            flat = input_tensor.contiguous().view(-1)
            return flat[fwd_t].view(input_tensor.shape)

        @staticmethod
        def backward(ctx, grad_output):
            inv_t, = ctx.saved_tensors
            flat_grad = grad_output.contiguous().view(-1)
            return flat_grad[inv_t].view(grad_output.shape), None, None, None

    class LegoInverseTransformFunction(torch.autograd.Function):
        """Autograd function for LEGO inverse layout transforms."""

        @staticmethod
        def forward(ctx, input_tensor, compiler, fwd_perm=None, inv_perm=None):
            ctx.compiler = compiler
            device = input_tensor.device

            if inv_perm is not None:
                inv_t = inv_perm
                fwd_t = fwd_perm
            else:
                fwd_np, inv_np = compiler.get_permutation_table()
                fwd_t = torch.from_numpy(fwd_np).to(device)
                inv_t = torch.from_numpy(inv_np).to(device)

            ctx.save_for_backward(fwd_t)
            flat = input_tensor.contiguous().view(-1)
            return flat[inv_t].view(input_tensor.shape)

        @staticmethod
        def backward(ctx, grad_output):
            fwd_t, = ctx.saved_tensors
            flat_grad = grad_output.contiguous().view(-1)
            return flat_grad[fwd_t].view(grad_output.shape), None, None, None

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
    # Stubs when PyTorch is not available
    LegoTransformFunction = None
    LegoInverseTransformFunction = None
