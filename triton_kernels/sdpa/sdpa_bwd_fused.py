import torch
import os
import triton
import triton.language as tl
import math


DEVICE = triton.runtime.driver.active.get_active_torch_device()




@triton.autotune(
    [
        triton.Config({'PRE_BLOCK_SIZE_ROW': PRE_BLOCK_SIZE_ROW}, num_stages=num_stages, num_warps=num_warps)
        for PRE_BLOCK_SIZE_ROW in [32, 64, 128, 256]
        for num_stages in [3,5,7]
        for num_warps in [2,4,8]
    ], key=['Dh']
)
@triton.jit
def attn_bwd_preprocess(
    O_ptr, dldo_ptr, Delta_ptr,
    stride_O_B,stride_O_H,stride_O_N,stride_O_Dh,
    stride_dldO_B,stride_dldO_H,stride_dldO_N,stride_dldO_Dh,
    stride_delta_b, stride_delta_H, stride_delta_N,
    N, Dh: tl.constexpr,
    PRE_BLOCK_SIZE_ROW: tl.constexpr
):
    index_BH = tl.program_id(axis=1)
    row = tl.program_id(axis=0)

    row_offsets = row * PRE_BLOCK_SIZE_ROW + tl.arange(0, PRE_BLOCK_SIZE_ROW)
    cols_offsets = tl.arange(0, Dh)
    mask = row_offsets < N

    O_ptr += index_BH * stride_O_H
    O_offsets = row_offsets[:, None] * stride_O_N + cols_offsets[None, :] * stride_O_Dh
    O = tl.load(O_ptr + O_offsets, mask=mask[:, None], other=0.0)

    dldo_ptr += index_BH * stride_O_H
    dldO_offsets = row_offsets[:, None] * stride_dldO_N + cols_offsets[None, :] * stride_dldO_Dh
    dldO = tl.load(dldo_ptr + dldO_offsets, mask=mask[:, None], other=0.0)

    Delta = tl.sum(dldO * O, axis=1)
    Delta_ptr += index_BH * stride_delta_H
    tl.store(Delta_ptr + row_offsets, Delta, mask=mask)


@triton.jit
def _attn_backward_KV(
    K,V,dldK, dldV,
    Q_ptr, dldO_ptr, LSE_ptr, Delta,
    stride_N, stride_Dh,
    H,N,Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr,
    start_row, start_col, num_steps, scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr,
):
    offsets_row = start_row + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_col = start_col + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)

    Q_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_row[None, :] * stride_N
    dldO_offsets = offsets_row[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh

    for block_idx in range(num_steps):
        mask_N = offsets_row < N
        Q_T = tl.load(Q_ptr + Q_T_offsets, mask=mask_N[None, :], other=0.0)
        LSE = tl.load(LSE_ptr + offsets_row, mask=mask_N, other=0.0)



        dldO = tl.load(dldO_ptr + dldO_offsets, mask=mask_N[:, None], other=0.0)
        Delta = tl.load(_ptr + offsets, mask=mask_N, other=0.0)

        S_T = tl.dot(K, Q_T)
        P_T = tl.exp2(S_T - LSE[None, :])

        if MASK:
            mask = (offsets_col[:, None] <= offsets_row[None, :])
            P_T = tl.where(mask, P_T, 0.0)
        dldV = tl.dot(P_T, dldO, acc=dldV)
        dldp_T = tl.dot(V, tl.trans(dldO))
        dlds_T = (P_T * (dldp_T - Delta[None, :]) * ln2)
        dldK = tl.dot(dlds_T, tl.trans(Q_T), acc=dldK)

        offsets_row += BLOCK_SIZE_ROW
        Q_ptr += BLOCK_SIZE_ROW * stride_N
        dldO_ptr += BLOCK_SIZE_ROW * stride_N

        return dldK, dldV

@triton.jit
def _attn_backward_Q(
    dldQ, Q, dldO, LSE,
    K_ptr, V_ptr, Delta_ptr,
    stride_N, stride_Dh, H, N, Dh,
    BLOCK_SIZE_ROW2: tl.constexpr, BLOCK_SIZE_COL2: tl.constexpr,
    start_ROW, start_COL, n_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr
):
    offsets_ROW = start_ROW + tl.arange(), BLOCK_SIZE_ROW2
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL2)
    offsets_Dh = tl.arange(0, Dh)

    K_and_V_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_COL[None, :] * stride_N

    Delta = tl.load(Delta_ptr + offsets_ROW, mask=offsets_ROW < N, other=0.0)

    for block_idx in range(n_steps):
        KV_mask = (offsets_COL < N)[None, :]
        K_T = tl.load(K_ptr + K_and_V_T_offsets, mask=KV_mask, other=0.0)
        V_T = tl.load(V_ptr + K_and_V_T_offsets, mask=KV_mask, other=0.0)

        S = tl.dot(Q, K_T)
        P = tl.exp2(S - LSE)

        if MASK:
            P = tl.where(offsets_ROW[:, None] >= offsets_COL[None, :], P, 0.0)

        dldP = tl.dot(dldO, V_T)
        dldS = (P * (dldP - Delta[:, None] * ln2))

        dldQ = tl.dot(dldS, tl.trans(K_T), acc=dldQ)

        offsets_COL += BLOCK_SIZE_COL2
        K_ptr += BLOCK_SIZE_COL2 * stride_N
        V_ptr += BLOCK_SIZE_COL2 * stride_N

    return dldQ


@triton.autotune(
    [
        triton.Config({'BLOCK_SIZE_MACRO': BLOCK_SIZE_MACRO, "BLOCK_SIZE_MICRO": BLOCK_SIZE_MICRO}, num_stages=n_stages, num_warps=n_warps)
        for BLOCK_SIZE_MICRO in [16,32,64]
        for BLOCK_SIZE_MACRO in [32,64,128]
        for n_stages in [3,5,7]
        for n_warps in [4,8,16]
        if BLOCK_SIZE_MACRO > BLOCK_SIZE_MICRO
    ], key=['Dh']
)
@triton.jit
def attn_bwd(Q_ptr, K_ptr, V_ptr, dldO_ptr, dldQ_ptr, dldK_ptr, dldV_ptr,
             LSE_ptr, Delta_ptr, scale, 
             stride_B, stride_H, stride_N, stride_Dh,
             H,N,Dh: tl.constexpr,
             BLOCK_SIZE_MACRO: tl.constexpr, BLOCK_SIZE_MICRO: tl.constexpr):

    ln2: tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889643

    idx_BH = tl.program_id(axis=1)
    idx_B = idx_BH // H
    idx_H = idx_BH % H
    BH_stride = idx_B * stride_B + idx_H * stride_H
    Q_ptr += BH_stride
    K_ptr += BH_stride
    V_ptr += BH_stride
    dldO_ptr += BH_stride
    dldQ_ptr += BH_stride
    dldK_ptr += BH_stride
    dldV_ptr += BH_stride
    BH_stride = idx_BH * N
    LSE_ptr += BH_stride
    Delta_ptr += BH_stride

    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)

    #stage 1
    BLOCK_SIZE_ROW1: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL1: tl.constexpr = BLOCK_SIZE_MACRO

    pid = tl.program_id(0)
    start_col = pid * BLOCK_SIZE_COL1
    start_row = start_col
    num_steps = BLOCK_SIZE_COL1 // BLOCK_SIZE_ROW1

    offsets_col1 = start_col + tl.arange(0, BLOCK_SIZE_COL1)
    offsets_Dh = tl.arange(0, Dh)
    KV_offsets = offsets_col1[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    KV_mask = offsets_col1[:, None] < N
    K = tl.load(K_ptr + KV_offsets, mask=KV_mask, other=0.0)
    V = tl.load(V_ptr + KV_offsets, mask=KV_mask, other=0.0)
    K += scale * rln2

    dldK = tl.zeros((BLOCK_SIZE_COL1, Dh), dtype=tl.float32)
    dldV = tl.zeros((BLOCK_SIZE_COL1, Dh), dtype=tl.float32)


    dldK, dldV = _attn_backward_KV(
        K,V,dldK,dldV, Q_ptr, dldO_ptr, LSE_ptr, Delta_ptr,
        stride_N, stride_Dh, BLOCK_SIZE_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=True
        )

    start_row += BLOCK_SIZE_MACRO
    N_adj = tl.cdiv(N, BLOCK_SIZE_COL1) * BLOCK_SIZE_COL1
    n_steps = (N_adj - start_row) // BLOCK_SIZE_ROW1

    dldK, dldV = _attn_backward_KV(
        K,V,dldK,dldV, Q_ptr, dldO_ptr, LSE_ptr, Delta_ptr,
        stride_N, stride_Dh, BLOCK_SIZE_ROW, start_COL, n_steps,
        scale, ln2, rln2,
        MASK=False
    )

    dldK *= scale * rln2
    tl.store(dldK_ptr + KV_offsets, dldK, mask=KV_mask)
    tl.store(dldV_ptr + KV_offsets, dldV, mask=KV_mask)


    #stage 2
    BLOCK_SIZE_ROW2: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL2: tl.constexpr = BLOCK_SIZE_MICRO

    start_ROW = pid * BLOCK_SIZE_ROW2
    start_COL = start_ROW
    n_steps = BLOCK_SIZE_ROW2 // BLOCK_SIZE_COL2

    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW2)
    QO_offsets = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    mask_ROW = offsets_ROW  < N
    Q = tl.load(Q_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.0)
    Q *= scale * rln2
    dldO = tl.load(dldO_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.0)
    LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_ROW, other=0.0)[:, None]

    dldQ = tl.zeros((BLOCK_SIZE_ROW2, Dh), dtype=tl.float32)
    dldQ = _attn_backward_Q(
        dldQ, Q, dldO, LSE,
        K_ptr, V_ptr, Delta_ptr,
        stride_N, stride_Dh, H, N, Dh,
        BLOCK_SIZE_ROW2, BLOCK_SIZE_COL2,
        start_ROW, start_COL, n_steps,
        scale, ln2, rln2,
        MASK=True
    )

    end_col = start_COL
    start_COL = 0
    n_steps = end_col // BLOCK_SIZE_COL2

    dldQ = _attn_backward_Q(
        dldQ, Q, dldO, LSE,
        K_ptr, V_ptr, Delta_ptr,
        stride_N, stride_Dh, H, N, Dh,
        BLOCK_SIZE_ROW2, BLOCK_SIZE_COL2,
        start_ROW, start_COL, n_steps,
        scale, ln2, rln2,
        MASK=False
    )

    dldQ *= scale * rln2
    tl.store(dldQ_ptr + QO_offsets, dldQ, mask=mask_ROW[:, None])

