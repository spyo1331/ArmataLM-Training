import torch
import triton
from sdpa_fwd_fused import attn_fwd
from sdpa_bwd_fused import attn_bwd_preprocess, attn_bwd

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
        ctx.B, ctx.H, ctx.N, ctx.Dh = B,H,N,Dh

        return O

    @staticmethod
    def backward(ctx, dldO):
        q,k,v,O, LSE = ctx.saved_tensors
        grid = ctx.grid
        scale = ctx.scale
        B,H,N,Dh=ctx.B, ctx.H, ctx.N, ctx.Dh

        dldq = torch.empty_like(q)
        dldk = torch.empty_like(k)
        dldv = torch.empty_like(v)

        dldO = dldO.contiguous()
        assert q.stride() == k.stride() == v.stride() == O.stride()

        Delta = torch.empty_like(LSE)
        pre_grid = lambda meta: (triton.cdiv(N, meta['PRE_BLOCK_SIZE_ROW']), B*H)
        attn_bwd_preprocess[pre_grid](
            O, dldO, Delta,
            O.stride(0), O.stride(1),O.stride(2),O.stride(3),
            dldO.stride(0), dldO.stride(1),dldO.stride(2),dldO.stride(3),
            Delta.stride(0),Delta.stride(1),Delta.stride(2),
            N, Dh,
        )

        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_MACRO']), B * H)
        attn_bwd[grid](
            q,k,v,
            dldO, dldq, dldk, dldv, LSE, Delta,
            scale,
            q.stride(0),q.stride(1),q.stride(2),q.stride(3),
        )

        return dldq, dldk, dldv, None

trit_attn = TritonFlashAttention.apply
