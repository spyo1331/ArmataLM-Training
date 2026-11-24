import torch
import os
import triton
import triton.language as tl
import math


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.autotune(
    [
        triton.Config({"PRE_BLOCK_SIZE_ROW": PRE_BLOCK_SIZE_ROW},
                      num_stages=num_stages, num_warps=num_warps)
        for PRE_BLOCK_SIZE_ROW in [32]
        for num_stages in [3]
        for num_warps in [4]
    ],
    key=["Dh"],
)
@triton.jit
def attn_backward_preprocess(
        O_ptr, dLdO_ptr, Delta_ptr,
        stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
        stride_dLdO_B, stride_dLdO_H, stride_dLdO_N, stride_dLdO_Dh,
        stride_Delta_B, stride_Delta_H, stride_Delta_N,
        N, Dh: tl.constexpr,
        PRE_BLOCK_SIZE_ROW: tl.constexpr,
):
    index_BH = tl.program_id(1)
    row = tl.program_id(0)

    row_offsets = row * PRE_BLOCK_SIZE_ROW + tl.arange(0, PRE_BLOCK_SIZE_ROW)
    col_offsets = tl.arange(0, Dh)
    mask = row_offsets < N

    O_ptr += index_BH * stride_O_H
    O_offsets = row_offsets[:, None] * stride_O_N + col_offsets[None, :] * stride_O_Dh
    O = tl.load(O_ptr + O_offsets, mask=mask[:, None], other=0.)

    dLdO_ptr += index_BH * stride_dLdO_H
    dLdO_offsets = row_offsets[:, None] * stride_dLdO_N + col_offsets[None, :] * stride_dLdO_Dh
    dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask=mask[:, None], other=0.)

    Delta = tl.sum(dLdO.to(tl.float32) * O.to(tl.float32), axis=1)
    Delta_ptr += index_BH * stride_Delta_H
    tl.store(Delta_ptr + row_offsets, Delta, mask=mask)


@triton.jit
def _attn_backward_KV(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh: tl.constexpr,
        BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr,
        start_row, start_col, num_steps, scale,
        ln2: tl.constexpr, rln2: tl.constexpr,
        MASK: tl.constexpr,
):
    offsets_row = start_row + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_col = start_col + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)

    Q_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_row[None, :] * stride_N
    dLdO_offsets = offsets_row[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh

    for block_idx in range(num_steps):
        mask_N = offsets_row < N
        Q_T = tl.load(Q_ptr + Q_T_offsets, mask=mask_N[None, :], other=0.0)
        LSE = tl.load(LSE_ptr + offsets_row, mask=mask_N, other=0.0)
        dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask=mask_N[:, None], other=0.0)
        Delta = tl.load(Delta_ptr + offsets_row, mask=mask_N, other=0.0)

        S_T = tl.dot(K, Q_T.to(K.dtype))
        P_T = tl.exp2(S_T - LSE[None, :])

        if MASK:
            causal_mask = (offsets_col[:, None] <= offsets_row[None, :])
            P_T = tl.where(causal_mask, P_T, 0.0)

        dLdV = tl.dot(P_T.to(dLdO.dtype), dLdO, acc=dLdV)

        dLdP_T = tl.dot(V, tl.trans(dLdO.to(V.dtype)))
        dLdS_T = P_T * (dLdP_T - Delta[None, :]) * ln2

        dLdK = tl.dot(dLdS_T.to(Q_T.dtype), tl.trans(Q_T), acc=dLdK)


        offsets_row += BLOCK_SIZE_ROW
        Q_T_offsets += BLOCK_SIZE_ROW * stride_N
        dLdO_offsets += BLOCK_SIZE_ROW * stride_N

    return dLdK, dLdV


@triton.jit
def _attn_backward_Q(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh: tl.constexpr,
        BLOCK_SIZE_ROW: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr,
        start_ROW, start_COL, num_steps,
        scale, ln2: tl.constexpr, rln2: tl.constexpr,
        MASK: tl.constexpr
):
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)

    K_V_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_COL[None, :] * stride_N

    Delta = tl.load(Delta_ptr + offsets_ROW, mask=offsets_ROW < N, other=0.)

    for block_idx in range(num_steps):
        mask_COL = offsets_COL < N
        K_T = tl.load(K_ptr + K_V_T_offsets, mask=mask_COL[None, :], other=0.)
        V_T = tl.load(V_ptr + K_V_T_offsets, mask=mask_COL[None, :], other=0.)

        S = tl.dot(Q, K_T.to(Q.dtype))
        P = tl.exp2(S - LSE)

        if MASK:
            causal_mask = (offsets_ROW[:, None] >= offsets_COL[None, :])
            P = tl.where(causal_mask, P, 0.)

        dLdP = tl.dot(dLdO, V_T.to(dLdO.dtype))
        dLdS = P * (dLdP - Delta[:, None]) * ln2

        dLdQ += tl.dot(dLdS.to(Q.dtype), tl.trans(K_T.to(Q.dtype)))


        offsets_COL += BLOCK_SIZE_COL
        K_V_T_offsets += BLOCK_SIZE_COL * stride_N

    return dLdQ


@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MACRO": BLOCK_SIZE_MACRO, "BLOCK_SIZE_MICRO": BLOCK_SIZE_MICRO},
                      num_stages=num_stages, num_warps=num_warps)
        for BLOCK_SIZE_MICRO in [16]
        for BLOCK_SIZE_MACRO in [32]
        for num_stages in [3]
        for num_warps in [4]
        if BLOCK_SIZE_MACRO > BLOCK_SIZE_MICRO
    ],
    key=["Dh"],
)
@triton.jit
def attn_backward(
        Q_ptr, K_ptr, V_ptr,
        dLdO_ptr, dLdQ_ptr, dLdK_ptr, dLdV_ptr,
        LSE_ptr, Delta_ptr,
        scale,
        stride_B, stride_H, stride_N, stride_Dh,
        H, N, Dh: tl.constexpr,
        BLOCK_SIZE_MICRO: tl.constexpr,
        BLOCK_SIZE_MACRO: tl.constexpr,
):
    ln2: tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634

    idx_batch_head = tl.program_id(1)
    idx_batch = idx_batch_head // H
    idx_head = idx_batch_head % H
    batch_head_jump = idx_batch * stride_B + idx_head * stride_H
    Q_ptr += batch_head_jump
    K_ptr += batch_head_jump
    V_ptr += batch_head_jump
    dLdO_ptr += batch_head_jump
    dLdQ_ptr += batch_head_jump
    dLdK_ptr += batch_head_jump
    dLdV_ptr += batch_head_jump

    batch_head_jump = idx_batch_head * N
    LSE_ptr += batch_head_jump
    Delta_ptr += batch_head_jump

    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)

    BLOCK_SIZE_ROW_1: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL_1: tl.constexpr = BLOCK_SIZE_MACRO

    pid = tl.program_id(0)
    start_COL = pid * BLOCK_SIZE_COL_1
    start_ROW = start_COL
    num_steps = BLOCK_SIZE_COL_1 // BLOCK_SIZE_ROW_1

    offsets_COL_1 = start_COL + tl.arange(0, BLOCK_SIZE_COL_1)
    offsets_Dh = tl.arange(0, Dh)
    KV_offsets = offsets_COL_1[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    KV_mask = offsets_COL_1[:, None] < N
    K = tl.load(K_ptr + KV_offsets, mask=KV_mask, other=0.)
    V = tl.load(V_ptr + KV_offsets, mask=KV_mask, other=0.)

    K *= (scale * rln2).to(K_ptr.dtype.element_ty)

    dLdK = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)
    dLdV = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)


    dLdK, dLdV = _attn_backward_KV(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=True
    )

    start_ROW += BLOCK_SIZE_COL_1

    N_adj = tl.cdiv(N, BLOCK_SIZE_COL_1) * BLOCK_SIZE_COL_1
    num_steps = (N_adj - start_ROW) // BLOCK_SIZE_ROW_1

    # недиагональная часть
    dLdK, dLdV = _attn_backward_KV(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False
    )

    dLdK *= scale * rln2

    tl.store(dLdK_ptr + KV_offsets, dLdK.to(dLdK_ptr.dtype.element_ty), mask=KV_mask)
    tl.store(dLdV_ptr + KV_offsets, dLdV.to(dLdV_ptr.dtype.element_ty), mask=KV_mask)

    BLOCK_SIZE_ROW_2: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL_2: tl.constexpr = BLOCK_SIZE_MICRO

    start_ROW = pid * BLOCK_SIZE_ROW_2
    start_COL = start_ROW
    num_steps = BLOCK_SIZE_ROW_2 // BLOCK_SIZE_COL_2

    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW_2)
    QO_offsets = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    mask_ROW = offsets_ROW < N
    Q = tl.load(Q_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.)
    Q = (Q * (scale * rln2)).to(Q_ptr.dtype.element_ty)
    dLdO = tl.load(dLdO_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.)
    LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_ROW, other=0.)[:, None]

    dLdQ = tl.zeros([BLOCK_SIZE_ROW_2, Dh], dtype=tl.float32)

    dLdQ = _attn_backward_Q(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=True
    )

    end_COL = start_COL
    start_COL = 0
    num_steps = end_COL // BLOCK_SIZE_COL_2

    dLdQ = _attn_backward_Q(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr,
        stride_N, stride_Dh,
        H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2,
        MASK=False
    )

    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + QO_offsets, dLdQ.to(dLdQ_ptr.dtype.element_ty), mask=mask_ROW[:, None])




