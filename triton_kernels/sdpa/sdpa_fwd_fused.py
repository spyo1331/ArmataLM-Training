from typing import Any
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import gc

DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = 'cuda'
DTYPE  = torch.float32

@triton.jit
def _attn_fwd_inner(
        Q,O,L,M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_qo, scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
        DIAGONAL: tl.constexpr,
        offsets_QO_N: tl.constexpr, offsets_KV_N: tl.constexpr,
        Dh: tl.constexpr, N: tl.constexpr
):
    if DIAGONAL:
        lo = block_index_qo * BLOCK_SIZE_QO
        hi = (block_index_qo + 1) * BLOCK_SIZE_QO
        lo = tl.multiple_of(lo, BLOCK_SIZE_QO)
    else:
        lo,hi = 0, block_index_qo * BLOCK_SIZE_QO

    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo

    for start_KV in range(lo,hi, BLOCK_SIZE_KV):
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)
        mask_kv_n = offsets_KV_N < N
        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_kv_n[None, :], other=0.)
        S = tl.dot(Q, K_T) * scale
        if DIAGONAL:
            causal_mask = offsets_QO_N[:, None] >= offsets_KV_N[None, :]
            S = tl.where(causal_mask, S, -1.0e6)

        M_new = tl.maximum(M, tl.max(S, axis=1))
        S -= M_new[:, None]
        P = tl.exp2(S)
        L_new = tl.sum(P, axis=1)
        alpha = tl.exp2(M - M_new)
        L = L * alpha + L_new

        V = tl.load(V_ptr + V_offsets, mask=mask_kv_n[:,None], other=0.0)
        O *= alpha[:, None]
        O = tl.dot(P,V, acc=O)

        M = M_new
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV

    return O,L,M




@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_QO": BQO, "BLOCK_SIZE_KV": BKV},
            num_stages=num_stages, num_warps=num_warps)
        for BQO in [16, 32, 64, 128]
        for BKV in [16, 32, 64, 128]
        if BKV <= BQO                      
        for num_stages in [2, 3, 5]
        for num_warps in [2, 4, 8]         
    ],
    key=['N', 'Dh'],                       
)
@triton.jit
def attn_fwd(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
        scale,
        stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
        stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
        stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
        stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
        stride_LSE_B, stride_LSE_H, stride_LSE_N,
        B,
        H,
        N,
        Dh: tl.constexpr,
        BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr
):
    rln2: tl.constexpr = 1.4426950408889643
    scale *= rln2

    tl.static_assert(BLOCK_SIZE_QO % BLOCK_SIZE_KV == 0)

    index_BH = tl.program_id(axis=1)
    index_B = index_BH // H
    index_H = index_BH % H
    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H

    block_index_QO  = tl.program_id(0)
    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_Dh = tl.arange(0, Dh)

    Q_offsets = offsets_QO_N[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh
    K_T_offsets = offsets_Dh[:, None] * stride_K_Dh + offsets_KV_N[None, :] * stride_K_N
    V_offsets = offsets_KV_N[:, None] * stride_V_N + offsets_Dh[None, :] * stride_V_Dh

    mask_QO_N = offsets_QO_N < N
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:, None], other=0.0)
    M = tl.full(shape=[BLOCK_SIZE_QO], value=-1e6, dtype=tl.float32)
    L = tl.full(shape=[BLOCK_SIZE_QO], value=1.0, dtype=tl.float32)
    O = tl.zeros([BLOCK_SIZE_QO, Dh], dtype=tl.float32)

    O,L,M = _attn_fwd_inner(
        Q,O,L,M,K_ptr, V_ptr,K_T_offsets,
        V_offsets,block_index_QO, 
        scale, 
        stride_K_N, stride_V_N, 
        BLOCK_SIZE_QO, BLOCK_SIZE_KV, 
        False, 
        offsets_QO_N,offsets_KV_N, 
        Dh,N
    )

    O,L,M = _attn_fwd_inner(
        Q,O,L,M,K_ptr, V_ptr,K_T_offsets,
        V_offsets,block_index_QO, 
        scale, 
        stride_K_N, stride_V_N, 
        BLOCK_SIZE_QO, BLOCK_SIZE_KV, 
        True, 
        offsets_QO_N,offsets_KV_N, 
        Dh,N
        )


    O = O/L[:, None]
    LSE = M + tl.math.log2(L)

    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=LSE_mask)

    O_offsets = offsets_QO_N[:, None] * stride_O_N + offsets_Dh[None, :] * stride_O_Dh
    tl.store(O_ptr + O_offsets, O, mask=mask_QO_N[:, None])

    

class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale):
        B,H,N,Dh = q.shape
        assert q.shape == k.shape == v.shape
        assert q.device == k.device == v.device

        O = torch.empty_like(q)
        LSE = torch.empty((B,H, N), device=q.device, dtype=torch.float32)
        grid = lambda meta: (
            triton.cdiv(N, meta['BLOCK_SIZE_QO']),
            B * H
        )

        attn_fwd[grid](
            q,k,v, O, LSE,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),B,H,N,Dh,
        )

        ctx.save_for_backward(q,k,v,O,LSE)
        ctx.grid = grid
        ctx.B, ctx.H, ctx.N, ctx.head_dim = B,H,N,Dh

        return O

triton_attn = TritonFlashAttention.apply

def test_FA_kernel(B,H,N,Dh, device=DEVICE, atol=5e-3):
    q = torch.randn(size=(B,H,N,Dh), dtype=torch.float32, device=device, requires_grad=True) - 0.5
    k = torch.randn(size=(B,H,N,Dh), dtype=torch.float32, device=device, requires_grad=True) - 0.5
    v = torch.randn(size=(B,H,N,Dh), dtype=torch.float32, device=device, requires_grad=True) - 0.5
    scale = 1/ math.sqrt(Dh)
    tri_out = triton_attn(q,k,v,scale)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q,k,v, is_causal=True, scale=scale)
    torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0.0)
    print('passed fwd')


def attn_flops(B, H, N, Dh, causal=True):
    f = 4.0 * B * H * N * N * Dh      # QK^T + P@V
    return f * 0.5 if causal else f


# ================= ХЕЛПЕРЫ =================
def make_qkv(B, H, N, Dh, dtype=None, requires_grad=False):
    dtype = dtype or DTYPE
    kw = dict(device=DEVICE, dtype=dtype, requires_grad=requires_grad)
    return (torch.randn(B, H, N, Dh, **kw),
            torch.randn(B, H, N, Dh, **kw),
            torch.randn(B, H, N, Dh, **kw))


def torch_attn(q, k, v, scale):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


def _safe_bench(fn, flops):
    """do_bench с защитой от OOM (актуально для SDPA-math на больших N)."""
    try:
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        return flops / (ms * 1e-3) / 1e12
    except RuntimeError:
        torch.cuda.empty_cache()
        return float('nan')


def detect_sdpa_backend(B=1, H=1, N=128, Dh=64, dtype=None):
    """Какой backend реально использует SDPA для данного dtype."""
    dtype = dtype or DTYPE
    q, k, v = make_qkv(B, H, N, Dh, dtype=dtype)
    s = 1 / math.sqrt(Dh)
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except ImportError:
        return "unknown"
    for name, be in [('flash', SDPBackend.FLASH_ATTENTION),
                     ('mem_efficient', SDPBackend.EFFICIENT_ATTENTION),
                     ('math', SDPBackend.MATH)]:
        try:
            with sdpa_kernel([be]):
                F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=s)
            return name
        except Exception:
            continue
    return "none"


# ================= TFLOPS: по длине последовательности =================
@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['N'], x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
    x_log=True, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch SDPA', 'Triton FA'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='attn_fwd_tflops_by_N',
    args={'B': 4, 'H': 8, 'Dh': 64},
))
def bench_tflops_by_N(B, H, N, Dh, provider):
    q, k, v = make_qkv(B, H, N, Dh)
    s = 1 / math.sqrt(Dh)
    fn = (lambda: torch_attn(q, k, v, s)) if provider == 'torch' else (lambda: triton_attn(q, k, v, s))
    return _safe_bench(fn, attn_flops(B, H, N, Dh))


# ================= TFLOPS: по размерности головы =================
@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['Dh'], x_vals=[32, 64, 128],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch SDPA', 'Triton FA'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='attn_fwd_tflops_by_Dh',
    args={'B': 4, 'H': 8, 'N': 2048},
))
def bench_tflops_by_Dh(B, H, N, Dh, provider):
    q, k, v = make_qkv(B, H, N, Dh)
    s = 1 / math.sqrt(Dh)
    fn = (lambda: torch_attn(q, k, v, s)) if provider == 'torch' else (lambda: triton_attn(q, k, v, s))
    return _safe_bench(fn, attn_flops(B, H, N, Dh))


# ================= TFLOPS: по батчу =================
@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['B'], x_vals=[1, 2, 4, 8, 16, 32],
    x_log=True, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch SDPA', 'Triton FA'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='attn_fwd_tflops_by_B',
    args={'H': 8, 'N': 1024, 'Dh': 64},
))
def bench_tflops_by_B(B, H, N, Dh, provider):
    q, k, v = make_qkv(B, H, N, Dh)
    s = 1 / math.sqrt(Dh)
    fn = (lambda: torch_attn(q, k, v, s)) if provider == 'torch' else (lambda: triton_attn(q, k, v, s))
    return _safe_bench(fn, attn_flops(B, H, N, Dh))


# ================= ПАМЯТЬ =================
def measure_memory(fn, warmup=5, iters=20):
    for _ in range(warmup):
        out = fn(); del out
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    for _ in range(iters):
        out = fn()
        torch.cuda.synchronize()
        del out
    peak = torch.cuda.max_memory_allocated()
    return base / 2**20, peak / 2**20, (peak - base) / 2**20


def _print_mem_row(label, m_pt, m_tri):
    print(f"{label}")
    print(f"  PyTorch : peak={m_pt[1]:9.1f} MB  delta={m_pt[2]:9.1f} MB")
    print(f"  Triton  : peak={m_tri[1]:9.1f} MB  delta={m_tri[2]:9.1f} MB")
    d = m_pt[2] - m_tri[2]
    pct = 100 * d / m_pt[2] if m_pt[2] else 0.0
    print(f"  → экономия {d:+.1f} MB ({pct:+.1f}%)")


def bench_memory(B=4, H=8, N=2048, Dh=64, dtype=None):
    dtype = dtype or DTYPE
    s = 1 / math.sqrt(Dh)
    print(f"\nMemory: B={B}, H={H}, N={N}, Dh={Dh}, dtype={dtype}")
    print(f"SDPA backend для этого dtype: {detect_sdpa_backend(dtype=dtype)}")
    print("-" * 74)

    # ---- inference: no_grad ----
    q, k, v = make_qkv(B, H, N, Dh, dtype=dtype, requires_grad=False)
    with torch.no_grad():
        m_pt  = measure_memory(lambda: torch_attn(q, k, v, s))
        m_tri = measure_memory(lambda: triton_attn(q, k, v, s))
    _print_mem_row("Forward (inference, no_grad):", m_pt, m_tri)
    lse_mb = B * H * N * 4 / 2**20
    print(f"  (LSE у Triton: {lse_mb:.2f} MB — накладные расходы на backward)")
    del q, k, v; gc.collect(); torch.cuda.empty_cache()

    # ---- training forward: граф удерживает сохранённые тензоры ----
    q, k, v = make_qkv(B, H, N, Dh, dtype=dtype, requires_grad=True)
    m_pt  = measure_memory(lambda: torch_attn(q, k, v, s))
    m_tri = measure_memory(lambda: triton_attn(q, k, v, s))
    _print_mem_row("\nForward (training, save_for_backward):", m_pt, m_tri)
    del q, k, v; gc.collect(); torch.cuda.empty_cache()

    # ---- backward не реализован ----
    print("\nForward + Backward: пропущено — у TritonFlashAttention нет backward().")


# ================= КОРРЕКТНОСТЬ (гейт перед бенчмарком) =================
def check_correctness(B=2, H=2, N=69, Dh=64, dtype=None, atol=2e-2):
    dtype = dtype or DTYPE
    torch.manual_seed(0)
    q, k, v = make_qkv(B, H, N, Dh, dtype=dtype)
    s = 1 / math.sqrt(Dh)
    ref = torch_attn(q.float(), k.float(), v.float(), s)
    got = triton_attn(q, k, v, s).float()
    err = (ref - got).abs().max().item()
    ok = err < atol
    print(f"Correctness (B={B},H={H},N={N},Dh={Dh},{dtype}): max_err={err:.3e} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


# ================= ЗАПУСК =================
if __name__ == "__main__":
    for dtype in [torch.float32]:
        DTYPE = dtype
        print("\n" + "=" * 74)
        print(f"DTYPE = {dtype}")
        print("=" * 74)

        if not check_correctness():
            print("Сначала почините кернел (см. фильтр конфигов BKV <= BQO).")
            continue

        bench_memory(B=4, H=8, N=2048, Dh=64, dtype=dtype)

        tag = str(dtype).split('.')[-1]
        bench_tflops_by_N.run(show_plots=True, print_data=True)
        bench_tflops_by_Dh.run(show_plots=True, print_data=True)
        bench_tflops_by_B.run(show_plots=True, print_data=True)