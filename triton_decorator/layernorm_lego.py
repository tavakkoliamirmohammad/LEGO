import torch
import triton
import triton.language as tl
import sympy as sp
import lego
from lego.lego import *
from lego import jit as lego_jit

@lego_jit
@triton.jit
def _layer_norm_fwd_fused(
    X, Y, W, B, Mean, Rstd,
    M, N, eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    L_XY = OrderBy(Row(M, N)).TileBy([M, N / BLOCK_SIZE_N], [1, BLOCK_SIZE_N])
    L_M = OrderBy(Row(M)).TileBy([M], [1])
    L_W = OrderBy(Row(1, N)).TileBy([1, N / BLOCK_SIZE_N], [1, BLOCK_SIZE_N])

    _mean = tl.zeros([1, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_x = L_XY[pid, k, 0, 0:BLOCK_SIZE_N]
        mask = (k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]) < N
        a = tl.load(X + offset_x, mask=mask, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean) / N
    
    _var = tl.zeros([1, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_x = L_XY[pid, k, 0, 0:BLOCK_SIZE_N]
        mask = (k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]) < N
        x = tl.load(X + offset_x, mask=mask, other=0.).to(tl.float32)
        x = tl.where(mask, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var) / N
    rstd = 1 / tl.sqrt(var + eps)
    
    tl.store(Mean + L_M[pid, 0], mean)
    tl.store(Rstd + L_M[pid, 0], rstd)
    
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        mask = (k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :]) < N
        offset_w = L_W[0, k, 0, 0:BLOCK_SIZE_N]
        w = tl.load(W + offset_w, mask=mask)
        b = tl.load(B + offset_w, mask=mask)
        offset_x = L_XY[pid, k, 0, 0:BLOCK_SIZE_N]
        x = tl.load(X + offset_x, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + offset_x, y, mask=mask)

@lego_jit
@triton.jit
def _layer_norm_bwd_dx_fused(DX, DY, DW, DB, X, W, Mean, Rstd, Lock, stride, M, N, GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    L_XY = OrderBy(Row(M, N)).TileBy([M, N / BLOCK_SIZE_N], [1, BLOCK_SIZE_N])
    L_M = OrderBy(Row(M)).TileBy([M], [1])
    L_W = OrderBy(Row(1, N)).TileBy([1, N / BLOCK_SIZE_N], [1, BLOCK_SIZE_N])
    W_BWD_L = OrderBy(Row(1, BLOCK_SIZE_N)).TileBy([1], [BLOCK_SIZE_N])

    mask = tl.arange(0, BLOCK_SIZE_N)[None, :] < N
    lock_id = pid % GROUP_SIZE_M
    Lock_ptr = Lock + lock_id
    Count = Lock_ptr + GROUP_SIZE_M
    
    offset_dwdb = L_XY[lock_id, 0, 0, 0:BLOCK_SIZE_N]
    DW_ptrs = DW + offset_dwdb
    DB_ptrs = DB + offset_dwdb
    
    offset_x = L_XY[pid, 0, 0, 0:BLOCK_SIZE_N]
    x = tl.load(X + offset_x, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + offset_x, mask=mask, other=0).to(tl.float32)
    
    offset_w = W_BWD_L[0, 0:BLOCK_SIZE_N]
    w = tl.load(W + offset_w, mask=mask).to(tl.float32)
    
    mean = tl.load(Mean + L_M[pid, 0])
    rstd = tl.load(Rstd + L_M[pid, 0])
    
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy) / N
    c2 = tl.sum(wdy) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    
    tl.store(DX + offset_x, dx, mask=mask)
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock_ptr, 0, 1) == 1:
        pass
    count = tl.load(Count)
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW_ptrs, mask=mask)
        partial_db += tl.load(DB_ptrs, mask=mask)
    tl.store(DW_ptrs, partial_dw, mask=mask)
    tl.store(DB_ptrs, partial_db, mask=mask)
    tl.atomic_xchg(Lock_ptr, 0)

@triton.jit
def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)

class LEGOLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        y = torch.empty_like(x)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
        rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _layer_norm_fwd_fused[(M, )](
            x_arg, y, weight, bias, mean, rstd,
            M, N, eps,
            BLOCK_SIZE_N=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        N = w.shape[0]
        GROUP_SIZE_M = 64
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        _dw = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        _db = torch.zeros((GROUP_SIZE_M, N), dtype=x.dtype, device=w.device)
        dw = torch.empty((N, ), dtype=w.dtype, device=w.device)
        db = torch.empty((N, ), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M, )](
            dx, dy, _dw, _db, x, w, m, v, locks,
            x_arg.stride(0), M, N, # Fixed missing stride argument
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=ctx.num_warps)
        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db, min(GROUP_SIZE_M, M), N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128, num_ctas=1)
        return dx, None, dw, db, None

lego_layer_norm = LEGOLayerNorm.apply

def test_lego_layer_norm(M, N, dtype, eps=1e-5, device='cuda'):
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    y_tri = lego_layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    assert torch.allclose(y_tri, y_ref, atol=2e-2, rtol=1e-2)
    assert torch.allclose(dx_tri, dx_ref, atol=2e-2, rtol=1e-2)
    assert torch.allclose(db_tri, db_ref, atol=2e-2, rtol=1e-2)
    assert torch.allclose(dw_tri, dw_ref, atol=2e-2, rtol=1e-2)
    print(f"✅ LayerNorm match for M={M}, N={N}")

if __name__ == "__main__":
    test_lego_layer_norm(1151, 8192, torch.float16)
