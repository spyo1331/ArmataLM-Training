import torch
import triton
import math
from triton_kernels import trit_attn
DEVICE = 'cuda'



def test_FA(B,H,N,Dh, device=DEVICE, atol=5e-3):
    q = torch.randn((B,H,N,Dh), dtype=torch.float32, device=device, requires_grad=True)
    k = torch.randn((B,H,N,Dh), dtype=torch.float32, device=device, requires_grad=True)
    v = torch.randn((B,H,N,Dh), dtype=torch.float32, device=device, requires_grad=True)
    scale = 1/math.sqrt(Dh)
    tri_out = trit_attn(q,k,v,scale)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q,k,v, is_causal=True, scale=scale)
    torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0.0)
    print('passed fwd')

    dldout = 0.1 * torch.randn_like(q)

    tri_out.backward(dldout, retain_graph=True)
    dldq_tri, dldk_tri, dldv_tri = [_.grad.clone() for _ in [q,k,v]]
    q.grad, k.grad, v.grad = None, None, None

    ref_out.backward(dldout, retain_graph=True)
    dldq_ref, dldk_ref, dldv_ref = [_.grad.clone() for _ in [q,k,v]]
    q.grad, k.grad, v.grad = None, None, None

    torch.testing.assert_close(dldq_tri, dldq_ref, atol=atol, rtol=0.)
    torch.testing.assert_close(dldk_tri, dldk_ref, atol=atol, rtol=0.)
    torch.testing.assert_close(dldv_tri, dldv_ref, atol=atol, rtol=0.)
    print('passed bwd')



if __name__ == '__main__':  
    print(test_FA(1,1,128,32))