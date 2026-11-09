import torch
from triton_kernels.rms_norm.fused_rms_norm_fwd import fused_rms_norm_fwd
from triton_kernels.rms_norm.fused_rms_norm_bwd import fused_rms_norm_bwd

class FusedRMSNormalization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, g, eps=1e-6):
        was_2d = (x.ndim == 2)
        if was_2d:
            x = x.unsqueeze(0)

        out, rms = fused_rms_norm_fwd(x, g, eps)

        ctx.save_for_backward(x, g, rms)
        ctx.was_2d = was_2d

        return out.squeeze(0) if was_2d else out

    @staticmethod
    def backward(ctx, go):
        x, g, rms = ctx.saved_tensors

        if ctx.was_2d:
            go = go.unsqueeze(0)

        dx, dg = fused_rms_norm_bwd(go, x, g, rms)

        if ctx.was_2d:
            dx = dx.squeeze(0)

        return dx, dg, None