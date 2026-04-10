"""
LEGO Layer 1: Kernel Codegen via torch.library

Registers ``lego::mm`` and ``lego::bmm`` as custom ops with:
  - torch.library registration (CPU + CUDA)
  - Fake-tensor impl (shape/dtype inference for torch.compile)
  - Autograd formula (forward + backward)

Eager mode uses standard matmul.  Phase 1+ will replace the impl
with Triton kernels generated from LEGO layout algebra.
"""

import torch

# ============================================================================
# lego::mm
# ============================================================================

@torch.library.custom_op("lego::mm", mutates_args=())
def lego_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Layout-aware matrix multiply (eager fallback)."""
    return torch.mm(a, b)


@lego_mm.register_fake
def _mm_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)


def _mm_setup_ctx(ctx, inputs, output):
    a, b = inputs
    ctx.save_for_backward(a, b)


def _mm_backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a = grad @ b.t() if ctx.needs_input_grad[0] else None
    grad_b = a.t() @ grad if ctx.needs_input_grad[1] else None
    return grad_a, grad_b


lego_mm.register_autograd(_mm_backward, setup_context=_mm_setup_ctx)


# ============================================================================
# lego::bmm
# ============================================================================

@torch.library.custom_op("lego::bmm", mutates_args=())
def lego_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Layout-aware batched matrix multiply (eager fallback)."""
    return torch.bmm(a, b)


@lego_bmm.register_fake
def _bmm_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        a.shape[0], a.shape[1], b.shape[2], dtype=a.dtype, device=a.device
    )


def _bmm_setup_ctx(ctx, inputs, output):
    a, b = inputs
    ctx.save_for_backward(a, b)


def _bmm_backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a = torch.bmm(grad, b.transpose(1, 2)) if ctx.needs_input_grad[0] else None
    grad_b = torch.bmm(a.transpose(1, 2), grad) if ctx.needs_input_grad[1] else None
    return grad_a, grad_b


lego_bmm.register_autograd(_bmm_backward, setup_context=_bmm_setup_ctx)


# ============================================================================
# lego::addmm  (bias + a @ b — used by nn.Linear)
# ============================================================================

@torch.library.custom_op("lego::addmm", mutates_args=())
def lego_addmm(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Layout-aware addmm (eager fallback)."""
    return torch.addmm(bias, a, b)


@lego_addmm.register_fake
def _addmm_fake(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)


def _addmm_setup_ctx(ctx, inputs, output):
    bias, a, b = inputs
    ctx.save_for_backward(a, b)


def _addmm_backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_bias = grad.sum(0) if ctx.needs_input_grad[0] else None
    grad_a = grad @ b.t() if ctx.needs_input_grad[1] else None
    grad_b = a.t() @ grad if ctx.needs_input_grad[2] else None
    return grad_bias, grad_a, grad_b


lego_addmm.register_autograd(_addmm_backward, setup_context=_addmm_setup_ctx)


# ============================================================================
# lego::permute — general layout permutation (torch.compile-safe)
# ============================================================================

@torch.library.custom_op("lego::permute", mutates_args=())
def lego_permute(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply gather-based permutation: output[i] = input[perm[i]]."""
    return x.contiguous().view(-1)[perm].view(x.shape)


@lego_permute.register_fake
def _permute_fake(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _permute_setup_ctx(ctx, inputs, output):
    x, perm = inputs
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.shape[0], device=perm.device)
    ctx.save_for_backward(inv_perm)


def _permute_backward(ctx, grad):
    (inv_perm,) = ctx.saved_tensors
    grad_x = grad.contiguous().view(-1)[inv_perm].view(grad.shape)
    return grad_x, None


lego_permute.register_autograd(_permute_backward, setup_context=_permute_setup_ctx)


# ============================================================================
# User-facing decorator
# ============================================================================

def torch_op(qualname, *, mutates_args=(), fake=None):
    """Register a LEGO custom op via torch.library.

    Usage::

        @lego.torch_op("lego::my_kernel")
        def my_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            ...

        # With explicit fake-tensor impl for ops with custom shape logic:
        @lego.torch_op("lego::my_kernel", fake=lambda a, b: torch.empty(a.shape[0], b.shape[1]))
        def my_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            ...
    """
    def decorator(fn):
        op = torch.library.custom_op(qualname, mutates_args=mutates_args)(fn)

        fake_fn = fake if fake is not None else fn

        @op.register_fake
        def _fake(*args, **kwargs):
            return fake_fn(*args, **kwargs)

        return op
    return decorator
