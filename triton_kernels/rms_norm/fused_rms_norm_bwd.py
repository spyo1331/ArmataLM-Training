import triton
import triton.language as tl
import torch
from triton_kernels.helper_functions import make_tensors_contiguous


@triton.jit
def _rms_norm_fused_dx(grad_in_ptr, g_ptr, rms_ptr, x_ptr, o_ptr, B,M,N,
                       grad_in_ptr_stride, grad_in_ptr_b_stride, g_stride, g_b_stride, x_stride, x_b_stride, o_stride, o_b_stride,
                       BLOCK_SIZE: tl.constexpr):
    """:argument grad_in_ptr: Поступающий градиент.
    :argument g_ptr: Веса масштаба.
    :argument rms_ptr: Сохраненные RMS значения."""

    row = tl.program_id(0)
    batch = tl.program_id(1)

    grad_in_ptr += (row * grad_in_ptr_stride) + (batch * grad_in_ptr_b_stride)
    x_ptr += (row * x_stride) + (batch * x_b_stride)
    rms_ptr += batch*M + row
    g_ptr += (row * g_stride) + (batch * g_b_stride)
    o_ptr += (row * o_stride) + (batch * o_b_stride)

    cols = tl.arange(0, BLOCK_SIZE)

    gi = tl.load(grad_in_ptr + cols, mask=cols<N, other=0.0)
    x = tl.load(x_ptr + cols, mask=cols<N, other=0.0)
    rms = tl.load(rms_ptr)
    g = tl.load(g_ptr + cols, mask=cols<N, other=0.0)
    r_rms = 1.0/rms
    first_element = (g * gi)*r_rms
    second_element = (x*r_rms*r_rms*r_rms)/N
    third_element = tl.sum(x*g*gi)
    output = first_element - second_element * third_element
    tl.store(o_ptr + cols, output, mask=cols < N)


@triton.jit
def _rms_norm_fused_dg(
        go_ptr, x_ptr, rms_ptr, dg_ptr,
        B, M, N,
        go_stride_m, go_stride_b,
        x_stride_m, x_stride_b,
        rms_stride_m, rms_stride_b,
        BLOCK_SIZE: tl.constexpr,
):

    row = tl.program_id(0)
    batch = tl.program_id(1)


    go_ptr += (row * go_stride_m) + (batch * go_stride_b)
    x_ptr += (row * x_stride_m) + (batch * x_stride_b)
    rms_ptr += (row * rms_stride_m) + (batch * rms_stride_b)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N


    go_val = tl.load(go_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x_val = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rms_val = tl.load(rms_ptr).to(tl.float32)


    partial = go_val * x_val / rms_val


    tl.atomic_add(dg_ptr + cols, partial, mask=mask)

@make_tensors_contiguous
def fused_rms_norm_bwd(go: torch.Tensor, x: torch.Tensor,
                       g: torch.Tensor, rms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    B, M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)

    dx = torch.empty_like(x)
    dg = torch.zeros(N, device=x.device, dtype=torch.float32)


    grid_dx = (M, B)
    _rms_norm_fused_dx[grid_dx](
        go, g, rms, x, dx,
        B, M, N,
        go.stride(1), go.stride(0),
        0, 0,
        x.stride(1), x.stride(0),
        dx.stride(1), dx.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )


    grid_dg = (B * M,)
    _rms_norm_fused_dg[grid_dg](
        go, x, rms, dg,
        B, M, N,
        go.stride(1), go.stride(0),
        x.stride(1), x.stride(0),
        rms.stride(1), rms.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return dx, dg
