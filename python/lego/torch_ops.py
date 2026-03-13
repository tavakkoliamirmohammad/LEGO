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

        Forward: applies the layout transformation (permutation).
        Backward: applies the inverse (since layouts are bijective).
        """

        @staticmethod
        def forward(ctx, input_tensor, compiler):
            ctx.compiler = compiler
            src = input_tensor.contiguous()

            if src.is_cuda:
                np_src = src.cpu().numpy()
            else:
                np_src = src.numpy()

            np_dst = compiler.transform_numpy(np_src)

            if src.is_cuda:
                output = torch.from_numpy(np_dst).to(input_tensor.device)
            else:
                output = torch.from_numpy(np_dst)

            return output

        @staticmethod
        def backward(ctx, grad_output):
            compiler = ctx.compiler
            grad = grad_output.contiguous()

            if grad.is_cuda:
                np_grad = grad.cpu().numpy()
            else:
                np_grad = grad.numpy()

            np_out = compiler.inverse_transform_numpy(np_grad)

            if grad.is_cuda:
                result = torch.from_numpy(np_out).to(grad_output.device)
            else:
                result = torch.from_numpy(np_out)

            return result, None

    class LegoInverseTransformFunction(torch.autograd.Function):
        """Autograd function for LEGO inverse layout transforms."""

        @staticmethod
        def forward(ctx, input_tensor, compiler):
            ctx.compiler = compiler
            src = input_tensor.contiguous()

            if src.is_cuda:
                np_src = src.cpu().numpy()
            else:
                np_src = src.numpy()

            np_dst = compiler.inverse_transform_numpy(np_src)

            if src.is_cuda:
                output = torch.from_numpy(np_dst).to(input_tensor.device)
            else:
                output = torch.from_numpy(np_dst)

            return output

        @staticmethod
        def backward(ctx, grad_output):
            compiler = ctx.compiler
            grad = grad_output.contiguous()

            if grad.is_cuda:
                np_grad = grad.cpu().numpy()
            else:
                np_grad = grad.numpy()

            np_out = compiler.transform_numpy(np_grad)

            if grad.is_cuda:
                result = torch.from_numpy(np_out).to(grad_output.device)
            else:
                result = torch.from_numpy(np_out)

            return result, None

else:
    # Stubs when PyTorch is not available
    LegoTransformFunction = None
    LegoInverseTransformFunction = None
