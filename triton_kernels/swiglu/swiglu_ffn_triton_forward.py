import torch
import triton.language as tl
import triton
import matplotlib


DEVICE = triton.runtime.driver.active.get_active_torch_device()
DTYPE = torch.float32

_converter_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16
}




def configs_for_swiglu(pre_hook=None):
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8},  num_stages=3, num_warps=8, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,  'GROUP_SIZE_M': 8},  num_stages=4, num_warps=4, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,  'GROUP_SIZE_M': 8},  num_stages=4, num_warps=4, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32,  'GROUP_SIZE_M': 8},  num_stages=4, num_warps=4, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,  'GROUP_SIZE_M': 8},  num_stages=4, num_warps=4, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32,  'GROUP_SIZE_M': 8},  num_stages=4, num_warps=4, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32,  'GROUP_SIZE_M': 8},  num_stages=5, num_warps=2, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32,  'GROUP_SIZE_M': 8},  num_stages=5, num_warps=2, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4, pre_hook=pre_hook),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4, pre_hook=pre_hook),
    ]


@triton.jit
def sigmoid_fast_exp2(x):
    LOG2E: tl.constexpr = 1.4426950408889634
    exp2_val = tl.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;",
        "=r, r",
        [-x * LOG2E],
        dtype=tl.float32,
        is_pure=True,
        pack=1
    )
    return 1.0 / (1.0 + exp2_val)

@triton.autotune(configs=configs_for_swiglu(), key=['M','N','K'])
@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0})
@triton.jit
def _up_proj_swiglu_ffn_forward(
        x_ptr, w_ptr, g_ptr, o_ptr, u_ptr, v_ptr,
        B, M, N, K,
        stride_xb, stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_gk, stride_gn,
        stride_ob, stride_om, stride_on,
        stride_ub, stride_um, stride_un,
        stride_vb, stride_vm, stride_vn,
        EVEN_K: tl.constexpr,
        CACHE_UV: tl.constexpr,
        OUTPUT_DTYPE: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    PID = tl.program_id(0)
    PID_B = tl.program_id(1)
    num_pids_along_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pids_along_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pids_in_group = GROUP_SIZE_M * num_pids_along_n
    group_ID = PID // num_pids_in_group
    first_pid_group_m = group_ID * GROUP_SIZE_M
    corrected_group_size_m = min(num_pids_along_m - first_pid_group_m, GROUP_SIZE_M)

    pid_m = first_pid_group_m + (PID % num_pids_in_group) % corrected_group_size_m
    pid_n = (PID % num_pids_in_group) // corrected_group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + stride_xb * PID_B + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    g_ptrs = g_ptr + offs_k[:, None] * stride_gk + offs_n[None, :] * stride_gn

    acc_w = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_left = K - k * BLOCK_SIZE_K
        if not EVEN_K:
            x = tl.load(x_ptrs, mask=offs_k[None, :] < k_left, other=0.0)
            w = tl.load(w_ptrs, mask=offs_k[:, None] < k_left, other=0.0)
            g = tl.load(g_ptrs, mask=offs_k[:, None] < k_left, other=0.0)
        else:
            x = tl.load(x_ptrs)
            w = tl.load(w_ptrs)
            g = tl.load(g_ptrs)

        acc_w = tl.dot(x, w, acc=acc_w)
        acc_g = tl.dot(x, g, acc=acc_g)

        x_ptrs += stride_xk * BLOCK_SIZE_K
        w_ptrs += stride_wk * BLOCK_SIZE_K
        g_ptrs += stride_gk * BLOCK_SIZE_K

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if CACHE_UV:
        u_ptrs = u_ptr + stride_ub * PID_B + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un
        v_ptrs = v_ptr + stride_vb * PID_B + offs_m[:, None] * stride_vm + offs_n[None, :] * stride_vn
        tl.store(u_ptrs, acc_w.to(OUTPUT_DTYPE), mask=mask)
        tl.store(v_ptrs, acc_g.to(OUTPUT_DTYPE), mask=mask)

    silu_activated = acc_w * sigmoid_fast_exp2(acc_w)
    out = silu_activated * acc_g

    o_ptrs = o_ptr + (stride_ob * PID_B + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(o_ptrs, out.to(OUTPUT_DTYPE), mask=mask)


@triton.autotune(configs=configs_for_swiglu(), key=['M','N','K'])
@triton.heuristics({"EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0})
@triton.jit
def _swiglu_ffn_down_proj_forward(
        a_ptr, d_ptr, out_ptr, B,M,N,K,
        stride_ab, stride_am, stride_an,
        stride_dn, stride_dk,
        stride_ob, stride_om, stride_ok,
        OUTPUT_DTYPE: tl.constexpr,
        EVEN_N: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    PID = tl.program_id(0)
    PID_B = tl.program_id(1)
    n_progs_along_m = tl.cdiv(M, BLOCK_SIZE_M)
    n_progs_along_k = tl.cdiv(K, BLOCK_SIZE_K)
    n_progs_in_group = n_progs_along_k * GROUP_SIZE_M
    group_ID = PID // n_progs_in_group
    first_element_m = group_ID * GROUP_SIZE_M
    corrected_group_size_m = min(n_progs_along_m - first_element_m, GROUP_SIZE_M)

    pid_m = first_element_m + (PID%n_progs_in_group)%corrected_group_size_m
    pid_k = (PID%n_progs_in_group)//corrected_group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + (PID_B * stride_ab + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    d_ptrs = d_ptr + (offs_n[:,None] * stride_dn + offs_k[None, :] * stride_dk)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        if EVEN_N:
            a = tl.load(a_ptrs)
            d = tl.load(d_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_n[None, :] < N - n*BLOCK_SIZE_N, other=0.0)
            d = tl.load(d_ptrs, mask=offs_n[:, None] < N - n*BLOCK_SIZE_N, other=0.0)

        acc = tl.dot(a,d, acc=acc)


        a_ptrs += stride_an * BLOCK_SIZE_N
        d_ptrs += stride_dn * BLOCK_SIZE_N

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ok = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    out_ptrs = out_ptr + (PID_B * stride_ob + offs_om[:, None] * stride_om + offs_ok[None, :] * stride_ok)
    out_mask = (offs_om[:,None] < M) & (offs_ok[None, :] < K)
    tl.store(out_ptrs, acc.to(OUTPUT_DTYPE), mask=out_mask)



def triton_dtype_converter(x: torch.Tensor) -> tl.dtype:
    return _converter_dtype[x.dtype]



def SwiGLU_ffn(x, w, g, d, training: bool = False):
    was_2d = (x.ndim == 2)
    if was_2d:
        x = x.unsqueeze(0)

    B, M, K_in = x.shape
    N_exp = w.shape[-1]
    K_out = d.shape[-1]
    OUTPUT_DTYPE = triton_dtype_converter(x)

    h = torch.empty((B, M, N_exp), device=x.device, dtype=x.dtype)

    if training:
        u = torch.empty_like(h); v = torch.empty_like(h)
        u_arg, v_arg = u, v
        su = (u.stride(0), u.stride(1), u.stride(2))
        sv = (v.stride(0), v.stride(1), v.stride(2))
    else:
        u_arg = v_arg = x
        su = (0, 0, 0); sv = (0, 0, 0)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N_exp, meta['BLOCK_SIZE_N']),
        B,
    )
    _up_proj_swiglu_ffn_forward[grid](
        x, w, g, h, u_arg, v_arg,
        B, M, N_exp, K_in,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1),
        g.stride(0), g.stride(1),
        h.stride(0), h.stride(1), h.stride(2),
        su[0], su[1], su[2],
        sv[0], sv[1], sv[2],
        CACHE_UV=training,
        OUTPUT_DTYPE=OUTPUT_DTYPE,
    )

    y = torch.empty((B, M, K_out), device=x.device, dtype=x.dtype)
    grid1 = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(K_out, meta['BLOCK_SIZE_K']),
        B,
    )
    _swiglu_ffn_down_proj_forward[grid1](
        h, d, y,
        B, M, N_exp, K_out,
        h.stride(0), h.stride(1), h.stride(2),
        d.stride(0), d.stride(1),
        y.stride(0), y.stride(1), y.stride(2),
        OUTPUT_DTYPE=OUTPUT_DTYPE,
    )

    # ---- ГЛАВНОЕ: squeeze только если вход был 2D ----
    if training:
        if was_2d:
            return y.squeeze(0), u.squeeze(0), v.squeeze(0), h.squeeze(0)
        return y, u, v, h
    return y.squeeze(0) if was_2d else y





