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

    



