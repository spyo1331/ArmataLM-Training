from swiglu_ffn_triton_forward import configs_for_swiglu, triton_dtype_converter, _up_proj_swiglu_ffn_forward, _swiglu_ffn_down_proj_forward, SwiGLU_ffn
from swiglu_ffn_backward import SwiGLU_ffn_bw
import torch




class SwiGLUFFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, g, d, cache_uv: bool = True):
        was_2d = (x.ndim == 2)
        if was_2d:
            x = x.unsqueeze(0)


        x3 = x.contiguous()
        w  = w.contiguous()
        g  = g.contiguous()
        d  = d.contiguous()


        if cache_uv:
            y, u, v, h = SwiGLU_ffn(x3, w, g, d, training=True)

            ctx.save_for_backward(x3, w, g, d, u, v, h)
        else:
            y = SwiGLU_ffn(x3, w, g, d, training=False)
            ctx.save_for_backward(x3, w, g, d)

        ctx.was_2d   = was_2d
        ctx.cache_uv = cache_uv
        return y.squeeze(0) if was_2d else y




    @staticmethod
    def backward(ctx, go):
        was_2d = ctx.was_2d
        cache_uv = ctx.cache_uv


        if cache_uv:
            x3, w, g, d, u, v, h = ctx.saved_tensors
        else:
            x3, w, g, d = ctx.saved_tensors
            with torch.no_grad():
                u = x3 @ w
                v = x3 @ g
                h = torch.nn.functional.silu(u) * v


        if was_2d:
            go = go.unsqueeze(0)


        dx, dw, dg, dd = SwiGLU_ffn_bw(go, x3, w, g, d, u, v, h)


        if was_2d:
            dx = dx.squeeze(0)

        return dx, dw, dg, dd, None



