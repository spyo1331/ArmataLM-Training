import torch
from torch import nn

n_ctx = 18
emb_dim = 64
n_heads = 8


class GatedMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, n_ctx: int, n_heads: int, qkv_bias: bool = False) -> None:
        super().__init__()
        assert emb_dim%n_heads==0, "Emb dim must be divisible by num attention heads"

        self.head_dim = emb_dim//n_heads
        self.n_heads = n_heads

        self.qkv = nn.Linear(emb_dim, emb_dim*3, bias=qkv_bias)
        self.o = nn.Linear(emb_dim, emb_dim)

        self.gate = nn.Linear(emb_dim, emb_dim)

        self.pos_ptr = 0
        self.register_buffer('cache_k', None, persistent=False)
        self.register_buffer('cache_v', None, persistent=False)
        self.register_buffer('mask', torch.triu(torch.ones(n_ctx, n_ctx), diagonal=1).bool(), persistent=False)



    @staticmethod
    def rope_rotation(x, theta_base=10000.0):
        head_dim = x.shape[-1]
        assert head_dim % 2 == 0, "Embedding dimension must be even"


        positions = torch.arange(0, head_dim // 2, dtype=x.dtype, device=x.device)
        thetas = theta_base ** (-2 * positions / head_dim)

        cos = torch.cos(torch.repeat_interleave(thetas, 2))
        sin = torch.sin(torch.repeat_interleave(thetas, 2))

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot = torch.stack([-x_odd, x_even], dim=-1).reshape_as(x)

        return x * cos + x_rot * sin


    def forward(self, x: torch.Tensor, use_cache=False):
        batch_size, n_tokens, emb_dim = x.shape

        gate = torch.sigmoid(self.gate(x))
        qkv = self.qkv(x)


        qkv = qkv.view(batch_size, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2,0,1,3,4)
        q,k,v = qkv



        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k,v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)
            keys_new,values_new = self.cache_k, self.cache_v

        else:
            keys_new, values_new = k, v



        q = q.transpose(1, 2)
        keys_new = keys_new.transpose(1, 2)
        values_new = values_new.transpose(1, 2)
        keys_new, values_new = self.rope_rotation(keys_new), self.rope_rotation(values_new)



        num_tokens_q = q.shape[-2]
        num_tokens_k = keys_new.shape[-2]
        if use_cache:
            mask_bool = self.mask.bool()[
                self.ptr_pos:self.ptr_pos + num_tokens_q, :num_tokens_k
            ]
            self.ptr_pos += num_tokens_q

        else:
            mask_bool = self.mask.bool()[:num_tokens_q, :num_tokens_k]

        context_vec = torch.nn.functional.scaled_dot_product_attention(query=q, key=keys_new, value=values_new, attn_mask=mask_bool, scale=1/(self.head_dim**0.5))
        attn_out = context_vec.transpose(1, 2).contiguous().view(batch_size, n_tokens, emb_dim)
        out = self.o(attn_out)
        return out




    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_pos = 0



class GatedDeltaNet(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, qkv_bias:bool=False) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0, "Emb dim must be divisible by num attention heads"
        self.n_heads = n_heads
        self.head_dim = emb_dim//self.n_heads
        self.qk = nn.Linear(emb_dim, emb_dim*2, bias=qkv_bias)
        self.v = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)

        self.W_gate = nn.Linear(emb_dim, emb_dim)
        self.alpha = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.beta = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.norm = nn.RMSNorm(self.head_dim)

        self.qk_conv = nn.Conv1d(emb_dim*2, emb_dim*2, kernel_size=3, groups=emb_dim*2, padding=1)
        self.o = nn.Linear(emb_dim, emb_dim)



    @staticmethod
    def _l2_norm(x: torch.Tensor, eps: float = 1e-5):
        return nn.functional.normalize(x, p=2, dim=-1, eps=eps)



    def forward(self, x:torch.Tensor):
        batch_size, n_tokens, emb_dim = x.shape

        gate = self.W_gate(x)
        qk = self.qk(x)
        qk = qk.permute(0,2,1)
        qk_conv = self.qk_conv(qk)
        qk = qk_conv.permute(0,2,1)
        v = self.v(x)
        alpha = torch.sigmoid(self.alpha(x))
        alpha = alpha.view(batch_size, n_tokens, self.n_heads, self.head_dim)

        beta = torch.sigmoid(self.beta(x))
        beta = beta.view(batch_size, n_tokens, self.n_heads, self.head_dim)
        gate = gate.view(batch_size, n_tokens, self.n_heads, self.head_dim)

        qk = qk.view(batch_size, n_tokens, 2, self.n_heads, self.head_dim)
        qk = qk.permute(2,0,1,3,4)
        v = v.view(batch_size, n_tokens, self.n_heads, self.head_dim)
        q,k = qk


        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        beta = beta.transpose(1,2)


        q,k,v = nn.functional.silu(q), nn.functional.silu(k), nn.functional.silu(v)
        q = self._l2_norm(q)
        k = self._l2_norm(k)

        state = x.new_zeros(batch_size, self.n_heads, self.head_dim, self.head_dim)
        out = []
        for i in range(n_tokens):
            k_t = k[:,:,i]
            q_t = q[:,:, i]
            v_t = v[:,:, i]
            b_t = beta[:, :, i]
            a_t = alpha[:, i].unsqueeze(-1)

            state = state * a_t
            kv_mem = (state *  k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * b_t
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            y_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)

            out.append(y_t)

        context = torch.stack(out, dim=2).transpose(1,2).contiguous()
        context = context.view(batch_size, n_tokens, self.n_heads, self.head_dim)

        context = self.norm(context)
        context = context * nn.functional.silu(gate)

        context = context.view(batch_size, n_tokens, emb_dim)
        out = self.o(context)

        return out

    def reset_cache(self):
        pass



class FFN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg['emb_dim'], int(cfg['intermed_dim']), bias=False)
        self.w2 = nn.Linear(cfg['emb_dim'], int(cfg['intermed_dim']), bias=False)
        self.w3 = nn.Linear(int(cfg['intermed_dim']), cfg['emb_dim'], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = nn.functional.silu(self.w1(x))
        data = self.w2(x)
        return self.w3(gate * data)



class TransformerBlockGDN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.attn1 = GatedDeltaNet(cfg['emb_dim'], cfg['n_heads'], cfg['qkv_bias'])
        self.norm1 = nn.RMSNorm(cfg['emb_dim'])

        self.mlp = FFN(cfg)
        self.norm2 = nn.RMSNorm(cfg['emb_dim'])

    def forward(self, x: torch.Tensor, use_cache=False) -> torch.Tensor:
        residual_path = x
        x = self.norm1(x)
        x = self.attn1(x)
        x = x + residual_path

        residual_path = x
        x = self.norm1(x)
        x = self.mlp(x)
        x = x + residual_path

        return x

class TransformerBlockGA(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.attn1 = GatedMultiHeadAttention(cfg['emb_dim'], cfg['n_ctx'], cfg['n_heads'], cfg['qkv_bias'])
        self.norm1 = nn.RMSNorm(cfg['emb_dim'])

        self.mlp = FFN(cfg)
        self.norm2 = nn.RMSNorm(cfg['emb_dim'])

    def forward(self, x: torch.Tensor, use_cache=False) -> torch.Tensor:
        residual_path = x
        x = self.norm1(x)
        x = self.attn1.forward(x, use_cache=use_cache)
        x = x + residual_path

        residual_path = x
        x = self.norm1(x)
        x = self.mlp(x)
        x = x + residual_path

        return x



class Qwen3LikeModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.trf_blocks = nn.ModuleList([TransformerBlockGA(cfg) if (x+1)%4==3 else TransformerBlockGDN(cfg) for x in range(cfg['n_layers'])])
        self.final_norm = nn.RMSNorm(cfg['emb_dim'])
        self.final_proj = nn.Linear(cfg['emb_dim'], cfg['vocab_size'])


    def forward(self, x:torch.Tensor, use_cache=False) -> torch.Tensor:
        x = self.emb(x)
        for layer in self.trf_blocks:
            x = layer(x, use_cache=use_cache)

        x = self.final_norm(x)
        x = self.final_proj(x)
        return x


    def reset_cache(self):
        for i in self.trf_blocks:
            i.attn1.reset_cache()


