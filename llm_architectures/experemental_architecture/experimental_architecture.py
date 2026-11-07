import torch
from torch import nn
import torch.nn.functional as F

class GatedCCGQA(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        n_kv_heads: int,
        compress_coefficient: int,
        qkv_bias: bool = False
    ) -> None:

        super().__init__()

        assert emb_dim % n_heads == 0, 'emb_dim must be divisible by n_heads'
        assert n_heads % n_kv_heads == 0, 'n_heads must be divisible by n_kv_heads '

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.compressed_head_dim = (emb_dim // n_heads) // compress_coefficient
        self.compressed_emb_dim = emb_dim // compress_coefficient
        self.kv_compressed_emb_dim = self.n_kv_heads * self.compressed_head_dim


        self.q_down = nn.Linear(emb_dim, self.compressed_emb_dim, bias=qkv_bias)


        self.k_down = nn.Linear(emb_dim, self.kv_compressed_emb_dim, bias=qkv_bias)
        self.v_down_curr = nn.Linear(emb_dim, self.kv_compressed_emb_dim // 2, bias=qkv_bias)
        self.v_down_prev = nn.Linear(emb_dim, self.kv_compressed_emb_dim // 2, bias=qkv_bias)


        self.W_gate = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)


        self.q_seq_conv = nn.Conv1d(
            in_channels=self.compressed_emb_dim,
            out_channels=self.compressed_emb_dim,
            kernel_size=3,
            groups=1,
            padding=2
        )
        self.k_seq_conv = nn.Conv1d(
            in_channels=self.kv_compressed_emb_dim,
            out_channels=self.kv_compressed_emb_dim,
            kernel_size=3,
            groups=1,
            padding=2
        )


        self.q_seq_ch_conv = nn.Conv1d(
            in_channels=self.compressed_head_dim,
            out_channels=self.compressed_head_dim,
            kernel_size=3,
            groups=1,
            padding=2
        )
        self.k_seq_ch_conv = nn.Conv1d(
            in_channels=self.compressed_head_dim,
            out_channels=self.compressed_head_dim,
            kernel_size=3,
            groups=1,
            padding=2
        )


        self.context_vectors_up_projection = nn.Linear(self.compressed_emb_dim, emb_dim, bias=qkv_bias)
        self.o = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)


        self.register_buffer('x_prev', None, persistent=False)
        self.register_buffer('cache_k', None, persistent=False)
        self.register_buffer('cache_v', None, persistent=False)
        self.pos_ptr = 0

    @staticmethod
    def _l2_norm(x: torch.Tensor, eps: float = 1e-5):
        return F.normalize(x, p=2, dim=-1, eps=eps)

    @staticmethod
    def rope_rotation(x, theta_base=10000.0, pos_offset=0):
        head_dim = x.shape[-1]
        seq_len = x.shape[-2]
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        positions = torch.arange(0, head_dim // 2, dtype=x.dtype, device=x.device)
        thetas = theta_base ** (-2 * positions / head_dim)

        m = torch.arange(pos_offset, pos_offset + seq_len, dtype=x.dtype, device=x.device)
        freqs = torch.outer(m, thetas)

        cos = torch.cos(freqs).repeat_interleave(repeats=2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(repeats=2, dim=-1)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack([-x_odd, x_even], dim=-1).reshape_as(x)

        return x * cos + x_rot * sin

    def forward(self, x: torch.Tensor, use_cache: bool = False):
        b, t, e = x.shape


        q = self.q_down(x)                         # [b, t, compressed_emb_dim]
        k = self.k_down(x)                         # [b, t, kv_compressed_emb_dim]
        v_curr = self.v_down_curr(x)               # [b, t, kv_compressed_emb_dim // 2]
        gate = F.sigmoid(self.W_gate(x))


        q = q.permute(0, 2, 1)                     # [b, compressed_emb_dim, t]
        q = self.q_seq_conv(q)[..., :t]            # preserve length

        k = k.permute(0, 2, 1)                     # [b, kv_compressed_emb_dim, t]
        k = self.k_seq_conv(k)[..., :t]


        q = q.transpose(1, 2).view(b, t, self.n_heads, self.compressed_head_dim)
        k = k.transpose(1, 2).view(b, t, self.n_kv_heads, self.compressed_head_dim)


        q = q.permute(0, 2, 3, 1).reshape(b * self.n_heads, self.compressed_head_dim, t)
        q = self.q_seq_ch_conv(q)[..., :t]


        k = k.permute(0, 2, 3, 1).reshape(b * self.n_kv_heads, self.compressed_head_dim, t)
        k = self.k_seq_ch_conv(k)[..., :t]


        q = q.reshape(b, t, self.n_heads, self.compressed_head_dim)
        k = k.reshape(b, t, self.n_kv_heads, self.compressed_head_dim)


        if self.x_prev is None:
            prev_token = torch.zeros(b, 1, e, device=x.device, dtype=x.dtype)
        else:
            prev_token = self.x_prev.unsqueeze(1)

        v_prev_proj = self.v_down_prev(prev_token)

        shifted_v = torch.cat([
            torch.zeros(b, 1, v_prev_proj.shape[-1], device=x.device, dtype=x.dtype),
            v_prev_proj.repeat(1, t - 1, 1)
        ], dim=1)

        v_shifted_final = torch.cat([v_curr, shifted_v], dim=-1)


        scale = torch.sqrt(torch.tensor(self.compressed_head_dim, device=k.device, dtype=k.dtype))
        q = self._l2_norm(q / scale)
        k = self._l2_norm(k / scale)

        q = self.rope_rotation(q)
        k = self.rope_rotation(k)

        v = v_shifted_final.view(b, t, self.n_kv_heads, self.compressed_head_dim)


        if use_cache:
            if self.cache_k is None:
                self.cache_k = k
                self.cache_v = v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)
            k, v = self.cache_k, self.cache_v
        else:

            pass


        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        num_tokens_q = q.shape[-2]
        num_tokens_k = k.shape[-2]


        if use_cache:
            old_pos_ptr = self.pos_ptr
            q_start = old_pos_ptr
            k_start = old_pos_ptr + num_tokens_q - num_tokens_k
            q_pos = torch.arange(q_start, q_start + num_tokens_q, device=x.device)
            k_pos = torch.arange(k_start, k_start + num_tokens_k, device=x.device)
            mask_bool = (q_pos.unsqueeze(-1) < k_pos.unsqueeze(0))
            self.pos_ptr += num_tokens_q
        else:
            q_pos = torch.arange(num_tokens_q, device=x.device)
            k_pos = torch.arange(num_tokens_k, device=x.device)
            mask_bool = (q_pos.unsqueeze(-1) < k_pos.unsqueeze(0))
            self.pos_ptr = 0



        context = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            attn_mask=mask_bool,
            is_causal=False,  enable_gqa=True
        )


        context = context.transpose(1, 2).contiguous().view(b, t, self.compressed_emb_dim)
        up = self.context_vectors_up_projection(context)
        output = up * gate
        output = self.o(output)

        self.x_prev = x[:, -1, :]

        return output




    def reset_cache(self) -> None:
        self.cache_k, self.cache_v = None, None
        self.ptr_pos = 0




class GatedDeltaNet2(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, qkv_bias: bool = False) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0, "Emb dim must be divisible by num attention heads"
        self.n_heads = n_heads
        self.head_dim = emb_dim // self.n_heads


        self.qk = nn.Linear(emb_dim, emb_dim * 2, bias=qkv_bias)


        self.v = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)


        self.alpha = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)


        self.erase_gate = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)

        self.write_gate = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)


        self.W_gate = nn.Linear(emb_dim, emb_dim)


        self.qk_conv = nn.Conv1d(
            emb_dim * 2, emb_dim * 2,
            kernel_size=3, groups=emb_dim * 2, padding=1
        )


        self.norm = nn.RMSNorm(self.head_dim)
        self.o = nn.Linear(emb_dim, emb_dim)

    @staticmethod
    def _l2_norm(x: torch.Tensor, eps: float = 1e-5):
        return F.normalize(x, p=2, dim=-1, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, emb_dim = x.shape


        gate = self.W_gate(x)

        qk = self.qk(x)
        qk = qk.permute(0, 2, 1)
        qk = self.qk_conv(qk)
        qk = qk.permute(0, 2, 1)

        v = self.v(x)


        alpha = torch.sigmoid(self.alpha(x))
        alpha = alpha.view(batch_size, n_tokens, self.n_heads, self.head_dim)


        b = torch.sigmoid(self.erase_gate(x))
        b = b.view(batch_size, n_tokens, self.n_heads, self.head_dim)


        w = torch.sigmoid(self.write_gate(x))
        w = w.view(batch_size, n_tokens, self.n_heads, self.head_dim)

        gate = gate.view(batch_size, n_tokens, self.n_heads, self.head_dim)

        qk = qk.view(batch_size, n_tokens, 2, self.n_heads, self.head_dim)
        qk = qk.permute(2, 0, 1, 3, 4)
        v = v.view(batch_size, n_tokens, self.n_heads, self.head_dim)

        q, k = qk  # q: [B, T, H, D], k: [B, T, H, D]


        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)

        q = self._l2_norm(q)
        k = self._l2_norm(k)


        state = x.new_zeros(batch_size, self.n_heads, self.head_dim, self.head_dim)
        out = []

        for i in range(n_tokens):
            k_t = k[:, i, :, :]
            q_t = q[:, i, :, :]
            v_t = v[:, i, :, :]


            a_t = alpha[:, i, :, :]


            b_t = b[:, i, :, :]
            w_t = w[:, i, :, :]


            state = state * a_t.unsqueeze(-1)


            k_erased = b_t * k_t

            k_erased_T_S = torch.einsum('bhd,bhde->bhe', k_erased, state)

            state = state - k_t.unsqueeze(-1) * k_erased_T_S.unsqueeze(-2)


            v_write = w_t * v_t
            state = state + k_t.unsqueeze(-1) @ v_write.unsqueeze(-2)


            y_t = torch.einsum('bhde,bhe->bhd', state, q_t)
            out.append(y_t)


        context = torch.stack(out, dim=2)  # [B, H, T, D]
        context = context.transpose(1, 2).contiguous()  # [B, T, H, D]
        context = context.view(batch_size, n_tokens, self.n_heads, self.head_dim)

        context = self.norm(context)
        context = context * F.silu(gate)
        context = context.view(batch_size, n_tokens, emb_dim)

        return self.o(context)

    def reset_cache(self) -> None:
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




class TransformerBlockCCGQA(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.attn=GatedCCGQA(cfg['emb_dim'], cfg['n_heads'], cfg['n_kv_heads'], cfg['compress_coef'], cfg['qkv_bias'])
        self.ffn = FFN(cfg)
        self.norm1 = nn.RMSNorm(cfg['emb_dim'])


    def forward(self, x:torch.Tensor, use_cache=False) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x_attn = self.attn.forward(x, use_cache=use_cache)
        x_ffn = self.ffn(x)
        x = x_attn + x_ffn
        x = x + shortcut


        return x

class TransformerBlockGDN2(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.attn=GatedDeltaNet2(cfg['emb_dim'], cfg['n_heads'], cfg['qkv_bias'])
        self.ffn = FFN(cfg)
        self.norm1 = nn.RMSNorm(cfg['emb_dim'])
        self.norm2 = nn.RMSNorm(cfg['emb_dim'])




    def forward(self, x:torch.Tensor, use_cache=False) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn.forward(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + shortcut

        return x




class EXPModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.trf_blocks = nn.ModuleList(
                [TransformerBlockCCGQA(cfg) if (x + 1) % 4 == 3 else TransformerBlockGDN2(cfg) for x in
                 range(cfg['n_layers'])])
        self.final_norm = nn.RMSNorm(cfg['emb_dim'])
        self.final_proj = nn.Linear(cfg['emb_dim'], cfg['vocab_size'])

    def forward(self, x: torch.Tensor, use_cache=False) -> torch.Tensor:
        x = self.emb(x)
        for layer in self.trf_blocks:
            x = layer(x, use_cache=use_cache)

        x = self.final_norm(x)
        x = self.final_proj(x)
        return x

    def reset_cache(self):
        for i in self.trf_blocks:
            i.attn.reset_cache()


