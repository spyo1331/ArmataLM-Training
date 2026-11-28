import torch
from torch import nn
import torch.nn.functional as F
from fla.layers import GatedDeltaNet2
from triton_kernels import triton_FA
from triton_kernels import triton_swiglu
from triton_kernels import triton_rmsnorm
import math

class GatedGQA(nn.Module):
    def __init__(self,emb_dim, n_heads, kv_heads, ctx_len, qkv_bias: bool) -> None:
        super().__init__()
        assert emb_dim%n_heads == 0, 'embedding dimension must be divisible by n_heads'
        self.n_heads = n_heads
        self.head_dim = emb_dim//n_heads
        self.emb_dim = emb_dim
        self.kv_heads = kv_heads
        self.scale = 1/math.sqrt(self.head_dim)
        self.group_size = self.n_heads//self.kv_heads



        self.Q_w = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.K_w = nn.Linear(emb_dim, kv_heads*self.head_dim, bias=qkv_bias)
        self.V_w = nn.Linear(emb_dim, kv_heads*self.head_dim, bias=qkv_bias)


        self.G_w = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.O_w = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)


        self.current_ptr = 0

        self.register_buffer('cache_k', None, persistent=False)
        self.register_buffer('cache_v', None, persistent=False)

    @staticmethod
    def rope_rotation(x, theta_base=10000.0):
        # x shape: [B, H, T, D]
        B, H, T, D = x.shape
        assert D % 2 == 0, "Embedding dimension must be even"


        positions = torch.arange(T, dtype=torch.float32, device=x.device)
        inv_freq = theta_base ** (-torch.arange(0, D, 2, dtype=torch.float32, device=x.device) / D)

        freqs = torch.outer(positions, inv_freq)


        emb = torch.cat((freqs, freqs), dim=-1)


        cos = emb.cos().to(x.dtype)[None, None, :, :]
        sin = emb.sin().to(x.dtype)[None, None, :, :]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot = torch.stack([-x_odd, x_even], dim=-1).reshape_as(x)

        return x * cos + x_rot * sin


    def forward(self, x: torch.Tensor, use_cache=False):
        B,T,E = x.shape

        q = self.Q_w(x)
        k = self.K_w(x)
        v = self.V_w(x)


        g = F.sigmoid(self.G_w(x))

        q = q.view(B,T,self.n_heads, self.head_dim).transpose(1,2)
        k = k.view(B,T,self.kv_heads, self.head_dim).transpose(1,2)
        v = v.view(B,T,self.kv_heads, self.head_dim).transpose(1,2)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k,v
            else:
                self.cache_k, self.cache_v = torch.concat([self.cache_k, k], dim=2), torch.concat([self.cache_v, v], dim=2)
            keys_new, values_new = self.cache_k, self.cache_v
        else:
            keys_new, values_new = k,v
            self.current_ptr=0

        keys_new,values_new = torch.repeat_interleave(keys_new, repeats=self.group_size, dim=1), torch.repeat_interleave(values_new, repeats=self.group_size, dim=1)
        keys_new, q = self.rope_rotation(keys_new), self.rope_rotation(q)


        if not use_cache:
            # training mode. efficient triton attention is used. KV caching technique have not implemented yet.
            attn = triton_FA(q,keys_new,values_new, self.scale)
            self.current_ptr = 0
        else:
            num_tokens_Q = q.shape[-2]
            num_tokens_K = keys_new.shape[-2]
            device = q.device
            q_positions = torch.arange(
                self.current_ptr,
                self.current_ptr + num_tokens_Q,
                device=device,
                dtype=torch.long,
            )
            self.current_ptr += num_tokens_Q
            k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
            mask = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

            attn = torch.nn.functional.scaled_dot_product_attention(q, keys_new, values_new, attn_mask=mask)


        attn = attn.contiguous().view(B,T,E)
        attn = attn * g
        attn = self.O_w(attn)

        return attn

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.current_ptr = 0



class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.w1 = nn.Linear(cfg['emb_dim'], int(cfg['intermed_dim']), bias=False, device=cfg['device'])
        self.w2 = nn.Linear(cfg['emb_dim'], int(cfg['intermed_dim']), bias=False, device=cfg['device'])
        self.w3 = nn.Linear(int(cfg['intermed_dim']), cfg['emb_dim'], bias=False, device=cfg['device'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return triton_swiglu(
            x,
            self.w1.weight.T,
            self.w2.weight.T,
            self.w3.weight.T,
            cache_uv=self.cfg['FFN_cache_uv']
        )



class ArmataV0TransformerBlockGQA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ffn = FFN(cfg=cfg)
        self.attn = GatedGQA(cfg['emb_dim'], cfg['n_heads'], cfg['kv_heads'], cfg['ctx_len'], cfg['qkv_bias'])
        self.scaling_w_for_rmsnorm1 = nn.Parameter(torch.rand(size=(cfg['emb_dim'],), device=cfg['device'], dtype=cfg['dtype']), requires_grad=True)
        self.scaling_w_for_rmsnorm2 = nn.Parameter(torch.rand(size=(cfg['emb_dim'],), device=cfg['device'], dtype=cfg['dtype']), requires_grad=True)




    def forward(self, x, use_cache: bool = False):

        shortcut = x
        x = triton_rmsnorm(x, self.scaling_w_for_rmsnorm1)
        x = self.attn.forward(x, use_cache=use_cache)
        x = x + shortcut

        shortcut = x
        x = triton_rmsnorm(x, self.scaling_w_for_rmsnorm2)
        x = self.ffn(x)
        x = x+shortcut

        return x


class ArmataV0TransformerBlockGDN2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ffn = FFN(cfg=cfg)
        self.attn = GatedDeltaNet2(
            cfg['emb_dim'],
            expand_v=1.0,
            head_dim=cfg['emb_dim'] // cfg['n_heads'],
            num_heads=cfg['n_heads'],
            num_v_heads=cfg['n_heads'],
            use_short_conv=True,
            conv_bias=False
        )
        self.scaling_w_for_rmsnorm1 = nn.Parameter(
            torch.rand(size=(cfg['emb_dim'],), device=cfg['device'], dtype=cfg['dtype']), requires_grad=True)
        self.scaling_w_for_rmsnorm2 = nn.Parameter(
            torch.rand(size=(cfg['emb_dim'],), device=cfg['device'], dtype=cfg['dtype']), requires_grad=True)

    def forward(self, x):
        shortcut = x
        x = triton_rmsnorm(x, self.scaling_w_for_rmsnorm1)

        attn_out = self.attn(x)
        if isinstance(attn_out, tuple):
            x = attn_out[0]
        else:
            x = attn_out

        x = x + shortcut

        shortcut = x
        x = triton_rmsnorm(x, self.scaling_w_for_rmsnorm2)
        x = self.ffn(x)
        x = x + shortcut

        return x






class ArmataLMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()


        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'], device=cfg['device'])

        self.layers = nn.ModuleList([
            ArmataV0TransformerBlockGQA(cfg) if i % 2 == 0 else ArmataV0TransformerBlockGDN2(cfg)
            for i in range(cfg['n_layers'])
        ])


        self.scaling_w_for_rmsnorm3 = nn.Parameter(
            torch.rand(size=(cfg['emb_dim'],), device=cfg['device'], dtype=cfg['dtype']), requires_grad=True)


        self.lm_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)


        self.apply(self._init_weights)



    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)




    def forward(self, idx, use_cache: bool = False):
        B, T = idx.shape


        x = self.tok_emb(idx)


        for layer in self.layers:
            if isinstance(layer, ArmataV0TransformerBlockGDN2):
                x = layer(x)
            else:
                x = layer(x, use_cache=use_cache)

        x = triton_rmsnorm(x, self.scaling_w_for_rmsnorm3)
        x = self.lm_head(x)

        return x




    def reset_cache(self):
        for layer in self.layers:
            if hasattr(layer.attn, 'reset_cache'):
                layer.attn.reset_cache()











