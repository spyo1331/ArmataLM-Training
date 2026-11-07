import triton
import triton.language as tl
from swiglu_ffn_triton_forward import configs_for_swiglu, sigmoid_fast_exp2
import torch
from triton_kernels.helper_functions import make_tensors_contiguous

def configs_for_wgrad():
    """Для wgrad-ядер (контракция по M — длинный цикл): больше BLOCK_SIZE_K."""
    return [
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,  'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    ]

@triton.autotune(configs=configs_for_swiglu(), key=['N_exp', 'K_out'])
@triton.heuristics({"EVEN_K": lambda args: args["K_out"] % args["BLOCK_SIZE_K"] == 0})
@triton.jit
def _bwd_dh_du_dv(
    go_ptr, d_ptr, u_ptr, v_ptr, du_ptr, dv_ptr,
    B, M, N_exp, K_out,
    stride_gob, stride_gom, stride_gok,
    stride_dn,  stride_dk,
    stride_ub,  stride_um,  stride_un,
    stride_vb,  stride_vm,  stride_vn,
    stride_dub, stride_dum, stride_dun,
    stride_dvb, stride_dvm, stride_dvn,
    IS_FP32: tl.constexpr,
    EVEN_K:  tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    PID   = tl.program_id(0)
    PID_B = tl.program_id(1)

    num_pid_m = tl.cdiv(M,     BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N_exp, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = PID // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (PID % num_pid_in_group) % group_size_m
    pid_n = (PID % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    go_ptrs = go_ptr + PID_B * stride_gob + offs_m[:, None] * stride_gom + offs_k[None, :] * stride_gok
    d_ptrs  = d_ptr  + offs_k[:, None] * stride_dk  + offs_n[None, :] * stride_dn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K_out, BLOCK_SIZE_K)):
        k_left = K_out - k * BLOCK_SIZE_K
        if EVEN_K:
            go = tl.load(go_ptrs); dt = tl.load(d_ptrs)
        else:
            go = tl.load(go_ptrs, mask=offs_k[None, :] < k_left, other=0.0)
            dt = tl.load(d_ptrs,  mask=offs_k[:, None] < k_left, other=0.0)
        if IS_FP32:
            acc = tl.dot(go, dt, acc=acc, input_precision='tf32')
        else:
            acc = tl.dot(go, dt, acc=acc)
        go_ptrs += stride_gok * BLOCK_SIZE_K
        d_ptrs  += stride_dk  * BLOCK_SIZE_K

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_exp)
    u = tl.load(u_ptr + PID_B * stride_ub + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un,
                mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + PID_B * stride_vb + offs_m[:, None] * stride_vm + offs_n[None, :] * stride_vn,
                mask=mask, other=0.0).to(tl.float32)

    s = sigmoid_fast_exp2(u)
    silu_u = u * s
    dsilu  = s * (1.0 + u * (1.0 - s))

    du = acc * v * dsilu
    dv = acc * silu_u

    tl.store(du_ptr + PID_B * stride_dub + offs_m[:, None] * stride_dum + offs_n[None, :] * stride_dun,
             du.to(du_ptr.dtype.element_ty), mask=mask)
    tl.store(dv_ptr + PID_B * stride_dvb + offs_m[:, None] * stride_dvm + offs_n[None, :] * stride_dvn,
             dv.to(dv_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=configs_for_wgrad(), key=['N_exp', 'K_out', 'M'])
@triton.heuristics({"EVEN_K": lambda args: args["M"] % args["BLOCK_SIZE_K"] == 0})
@triton.jit
def _bwd_dd(
    h_ptr, go_ptr, dd_ptr,
    N_exp, K_out, M,
    stride_hm,  stride_hn,
    stride_gom, stride_gok,
    stride_ddn, stride_ddk,
    IS_FP32: tl.constexpr,
    EVEN_K:  tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):

    PID = tl.program_id(0)
    num_pid_n = tl.cdiv(N_exp, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K_out, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = PID // num_pid_in_group
    first_pid_n = group_id * GROUP_SIZE_M
    group_size_n = min(num_pid_n - first_pid_n, GROUP_SIZE_M)
    pid_n = first_pid_n + (PID % num_pid_in_group) % group_size_n
    pid_k = (PID % num_pid_in_group) // group_size_n

    offs_n = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_K)

    h_ptrs  = h_ptr  + offs_m[None, :] * stride_hm  + offs_n[:, None] * stride_hn
    go_ptrs = go_ptr + offs_m[:, None] * stride_gom + offs_k[None, :] * stride_gok

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        m_left = M - m * BLOCK_SIZE_K
        if EVEN_K:
            ht = tl.load(h_ptrs); go = tl.load(go_ptrs)
        else:
            ht = tl.load(h_ptrs,  mask=offs_m[None, :] < m_left, other=0.0)
            go = tl.load(go_ptrs, mask=offs_m[:, None] < m_left, other=0.0)
        if IS_FP32:
            acc = tl.dot(ht, go, acc=acc, input_precision='tf32')
        else:
            acc = tl.dot(ht, go, acc=acc)
        h_ptrs  += stride_hm  * BLOCK_SIZE_K
        go_ptrs += stride_gom * BLOCK_SIZE_K

    mask = (offs_n[:, None] < N_exp) & (offs_k[None, :] < K_out)
    tl.store(dd_ptr + offs_n[:, None] * stride_ddn + offs_k[None, :] * stride_ddk,
             acc.to(dd_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=configs_for_swiglu(), key=['M', 'K_in', 'N_exp'])
@triton.heuristics({"EVEN_K": lambda args: args["N_exp"] % args["BLOCK_SIZE_K"] == 0})
@triton.jit
def _bwd_dx(
    du_ptr, dv_ptr, w_ptr, g_ptr, dx_ptr,
    B, M, K_in, N_exp,
    stride_dub, stride_dum, stride_dun,
    stride_dvb, stride_dvm, stride_dvn,
    stride_wk,  stride_wn,
    stride_gk,  stride_gn,
    stride_dxb, stride_dxm, stride_dxk,
    IS_FP32: tl.constexpr,
    EVEN_K:  tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    PID   = tl.program_id(0)
    PID_B = tl.program_id(1)

    num_pid_m = tl.cdiv(M,    BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(K_in, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = PID // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (PID % num_pid_in_group) % group_size_m
    pid_n = (PID % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    du_ptrs = du_ptr + PID_B * stride_dub + offs_m[:, None] * stride_dum + offs_k[None, :] * stride_dun
    dv_ptrs = dv_ptr + PID_B * stride_dvb + offs_m[:, None] * stride_dvm + offs_k[None, :] * stride_dvn
    w_ptrs  = w_ptr  + offs_k[:, None] * stride_wn  + offs_n[None, :] * stride_wk
    g_ptrs  = g_ptr  + offs_k[:, None] * stride_gn  + offs_n[None, :] * stride_gk

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(N_exp, BLOCK_SIZE_K)):
        k_left = N_exp - k * BLOCK_SIZE_K
        if EVEN_K:
            du = tl.load(du_ptrs); dv = tl.load(dv_ptrs)
            wt = tl.load(w_ptrs);  gt = tl.load(g_ptrs)
        else:
            du = tl.load(du_ptrs, mask=offs_k[None, :] < k_left, other=0.0)
            dv = tl.load(dv_ptrs, mask=offs_k[None, :] < k_left, other=0.0)
            wt = tl.load(w_ptrs,  mask=offs_k[:, None] < k_left, other=0.0)
            gt = tl.load(g_ptrs,  mask=offs_k[:, None] < k_left, other=0.0)
        if IS_FP32:
            acc = tl.dot(du, wt, acc=acc, input_precision='tf32')
            acc = tl.dot(dv, gt, acc=acc, input_precision='tf32')
        else:
            acc = tl.dot(du, wt, acc=acc)
            acc = tl.dot(dv, gt, acc=acc)
        du_ptrs += stride_dun * BLOCK_SIZE_K
        dv_ptrs += stride_dvn * BLOCK_SIZE_K
        w_ptrs  += stride_wn  * BLOCK_SIZE_K
        g_ptrs  += stride_gn  * BLOCK_SIZE_K

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < K_in)
    tl.store(dx_ptr + PID_B * stride_dxb + offs_m[:, None] * stride_dxm + offs_n[None, :] * stride_dxk,
             acc.to(dx_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=configs_for_wgrad(), key=['K_in', 'N_exp'])
@triton.heuristics({"EVEN_K": lambda args: args["M"] % args["BLOCK_SIZE_K"] == 0})
@triton.jit
def _bwd_dw_dg(
    x_ptr, du_ptr, dv_ptr, dw_ptr, dg_ptr,
    K_in, N_exp, M,
    stride_xm,  stride_xk,
    stride_dum, stride_dun,
    stride_dvm, stride_dvn,
    stride_dwk, stride_dwn,
    stride_dgk, stride_dgn,
    IS_FP32: tl.constexpr,
    EVEN_K:  tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):

    PID = tl.program_id(0)
    num_pid_k = tl.cdiv(K_in,  BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N_exp, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = PID // num_pid_in_group
    first_pid_k = group_id * GROUP_SIZE_M
    group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE_M)
    pid_k = first_pid_k + (PID % num_pid_in_group) % group_size_k
    pid_n = (PID % num_pid_in_group) // group_size_k

    offs_k = pid_k * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs  = x_ptr  + offs_m[None, :] * stride_xm  + offs_k[:, None] * stride_xk
    du_ptrs = du_ptr + offs_m[:, None] * stride_dum + offs_n[None, :] * stride_dun
    dv_ptrs = dv_ptr + offs_m[:, None] * stride_dvm + offs_n[None, :] * stride_dvn

    acc_w = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        m_left = M - m * BLOCK_SIZE_K
        if EVEN_K:
            xt = tl.load(x_ptrs); du = tl.load(du_ptrs); dv = tl.load(dv_ptrs)
        else:
            xt = tl.load(x_ptrs,  mask=offs_m[None, :] < m_left, other=0.0)
            du = tl.load(du_ptrs, mask=offs_m[:, None] < m_left, other=0.0)
            dv = tl.load(dv_ptrs, mask=offs_m[:, None] < m_left, other=0.0)
        if IS_FP32:
            acc_w = tl.dot(xt, du, acc=acc_w, input_precision='tf32')
            acc_g = tl.dot(xt, dv, acc=acc_g, input_precision='tf32')
        else:
            acc_w = tl.dot(xt, du, acc=acc_w)
            acc_g = tl.dot(xt, dv, acc=acc_g)
        x_ptrs  += stride_xm  * BLOCK_SIZE_K
        du_ptrs += stride_dum * BLOCK_SIZE_K
        dv_ptrs += stride_dvm * BLOCK_SIZE_K

    mask = (offs_k[:, None] < K_in) & (offs_n[None, :] < N_exp)
    tl.store(dw_ptr + offs_k[:, None] * stride_dwk + offs_n[None, :] * stride_dwn,
             acc_w.to(dw_ptr.dtype.element_ty), mask=mask)
    tl.store(dg_ptr + offs_k[:, None] * stride_dgk + offs_n[None, :] * stride_dgn,
             acc_g.to(dg_ptr.dtype.element_ty), mask=mask)



@make_tensors_contiguous
def SwiGLU_ffn_bw(go, x, w, g, d, u, v, h):
    """
    Backward pass для SwiGLU FFN. Работает только с 3D тензорами [B, M, *].

    Args:
        go: grad_output,        [B, M, K_out]
        x:  вход,               [B, M, K_in]
        w:  веса up projection, [K_in, N_exp]
        g:  веса gate,          [K_in, N_exp]
        d:  веса down proj,     [N_exp, K_out]
        u, v, h: промежуточные, [B, M, N_exp]

    Возвращает:
        dx: [B, M, K_in]
        dw: [K_in, N_exp]
        dg: [K_in, N_exp]
        dd: [N_exp, K_out]
    """

    B, M, K_in = x.shape
    N_exp = w.shape[-1]
    K_out = d.shape[-1]
    IS_FP32 = (x.dtype == torch.float32)


    du = torch.empty_like(u)
    dv = torch.empty_like(v)
    grid1 = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N_exp, meta['BLOCK_SIZE_N']),
        B,
    )
    _bwd_dh_du_dv[grid1](
        go, d, u, v, du, dv,
        B, M, N_exp, K_out,
        go.stride(0), go.stride(1), go.stride(2),
        d.stride(0), d.stride(1),
        u.stride(0), u.stride(1), u.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        du.stride(0), du.stride(1), du.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        IS_FP32=IS_FP32,
    )


    h_flat = h.reshape(B * M, N_exp)
    go_flat = go.reshape(B * M, K_out)
    dd = torch.empty_like(d)
    grid2 = lambda meta: (
        triton.cdiv(N_exp, meta['BLOCK_SIZE_M']) *
        triton.cdiv(K_out, meta['BLOCK_SIZE_N']),
    )
    _bwd_dd[grid2](
        h_flat, go_flat, dd,
        N_exp, K_out, B * M,
        h_flat.stride(0), h_flat.stride(1),
        go_flat.stride(0), go_flat.stride(1),
        dd.stride(0), dd.stride(1),
        IS_FP32=IS_FP32,
    )

    # ---- 3. dx (batch-aware) ----
    dx = torch.empty_like(x)
    grid3 = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) *
        triton.cdiv(K_in, meta['BLOCK_SIZE_N']),
        B,
    )
    _bwd_dx[grid3](
        du, dv, w, g, dx,
        B, M, K_in, N_exp,
        du.stride(0), du.stride(1), du.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        w.stride(0), w.stride(1),
        g.stride(0), g.stride(1),
        dx.stride(0), dx.stride(1), dx.stride(2),
        IS_FP32=IS_FP32,
    )

    # ---- 4. dw, dg (flatten trick) ----
    x_flat = x.reshape(B * M, K_in)
    du_flat = du.reshape(B * M, N_exp)
    dv_flat = dv.reshape(B * M, N_exp)
    dw = torch.empty_like(w)
    dg = torch.empty_like(g)
    grid4 = lambda meta: (
        triton.cdiv(K_in, meta['BLOCK_SIZE_M']) *
        triton.cdiv(N_exp, meta['BLOCK_SIZE_N']),
    )
    _bwd_dw_dg[grid4](
        x_flat, du_flat, dv_flat, dw, dg,
        K_in, N_exp, B * M,
        x_flat.stride(0), x_flat.stride(1),
        du_flat.stride(0), du_flat.stride(1),
        dv_flat.stride(0), dv_flat.stride(1),
        dw.stride(0), dw.stride(1),
        dg.stride(0), dg.stride(1),
        IS_FP32=IS_FP32,
    )

    return dx, dw, dg, dd