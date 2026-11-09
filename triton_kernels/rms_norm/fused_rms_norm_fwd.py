import triton
import triton.language as tl
import torch
from triton_kernels.helper_functions import make_tensors_contiguous

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N'],
)
@triton.jit
def _rms_norm_fused_fwd(x_ptr, g_ptr, o_ptr, rms_o_ptr, eps, B, M, N,
                    stride_x, stride_xb, stride_o, stride_ob,rms_o_stride,rms_stride_b,
                    BLOCK_SIZE: tl.constexpr):

    row = tl.program_id(0)
    batch = tl.program_id(1)

    x_ptr += (row * stride_x) + (batch * stride_xb)
    o_ptr += (row * stride_o) + (batch * stride_ob)
    blocks = tl.arange(0, BLOCK_SIZE)

    rms_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for tile_start_idx in range(0, N, BLOCK_SIZE):
        cols = tile_start_idx + blocks
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0)
        rms_acc += x*x

    sum_rms_acc=tl.sum(rms_acc)
    rms_final = tl.sqrt((1/N) * sum_rms_acc + eps)
    tl.store(rms_o_ptr + (row*rms_o_stride)+(batch*rms_stride_b), rms_final)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for tile_start_idx1 in range(0, N, BLOCK_SIZE):
        cols = tile_start_idx1 + blocks
        x = tl.load(x_ptr + cols, mask=cols < N, other=0.0)
        g = tl.load(g_ptr + cols, mask=cols < N, other=0.0)
        acc += (x/rms_final)*g

        tl.store(o_ptr + cols, acc, mask=cols < N)




@make_tensors_contiguous
def fused_rms_norm_fwd(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, M, N = x.shape

    out = torch.empty_like(x, device=x.device, dtype=x.dtype)
    rms = torch.empty(size=(B,M), device=x.device, dtype=x.dtype)
    BLOCK_SIZE=triton.next_power_of_2(N)


    grid = (M,B)
    _rms_norm_fused_fwd[grid](
        x, gamma, out, rms, eps, B,M,N,
        x.stride(1), x.stride(0), out.stride(1), out.stride(0), rms.stride(1), rms.stride(0),BLOCK_SIZE=BLOCK_SIZE
    )

    return out, rms




