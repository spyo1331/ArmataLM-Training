import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor
from sdpa_fwd_fused import attn_fwd

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