import torch
from torch import nn

def compute_rope_params(head_dim, theta_base=10_000, context_length=33000, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))


    positions = torch.arange(context_length, dtype=dtype)


    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)


    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin, offset=0):

    B, H, T, D = x.shape


    if cos.shape[-1] == D // 2:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    elif cos.shape[-1] != D:
        raise ValueError(f"Размерность cos ({cos.shape[-1]}) не равна D ({D}) или D//2 ({D//2})")

    # Берём нужный срез по длине
    cos = cos[offset:offset + T, :]
    sin = sin[offset:offset + T, :]

    # Разворачиваем для умножения
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)


    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class SlidingWindowLatentAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, compress_ratio: int, sliding_window_size: int, qkv_bias: bool = False) -> None:
        super().__init__()
        assert emb_dim%n_heads == 0, 'emb dim must be divisible by num attention heads'
        self.n_heads = n_heads
        self.head_dim = emb_dim//n_heads
        self.window_size = sliding_window_size
        self.latent_head_dim = self.head_dim//compress_ratio
        self.q = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.d_kv = nn.Linear(emb_dim, emb_dim//compress_ratio, bias=qkv_bias)
        self.o = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.d = nn.Linear(emb_dim//compress_ratio, emb_dim//compress_ratio, bias=qkv_bias)

        self.latent_up_proj_k = nn.Linear(emb_dim//compress_ratio, emb_dim)
        self.latent_up_proj_v = nn.Linear(emb_dim//compress_ratio, emb_dim)


        self.register_buffer('cache_kv', None, persistent=False)
        self.pos_ptr = 0

    def forward(self, x: torch.Tensor, cos, sin, use_cache=False) -> torch.Tensor:
        batch_size, n_tokens, emb_dim = x.shape

        q = self.q(x)
        kv_down_projected = self.d_kv(x)

        if use_cache:
            # Объединяем старый кэш с новыми токенами
            if self.cache_kv is None:
                latent_full = kv_down_projected
            else:
                latent_full = torch.cat([self.cache_kv, kv_down_projected], dim=1)

            if self.window_size is not None:
                attn_keep = min(latent_full.size(1), self.window_size + n_tokens - 1)
                attn_keep = int(attn_keep)
                latent_for_attn = latent_full[:, -attn_keep:, :]
                cache_keep = min(latent_full.size(1), self.window_size)
                cache_keep = int(cache_keep)
                self.cache_kv = latent_full[:, -cache_keep:, :]
            else:
                latent_for_attn = latent_full
                self.cache_kv = latent_full

            # Сохраняем позицию до обновления
            old_pos_ptr = self.pos_ptr
            tok_q = n_tokens
            tok_k = latent_for_attn.size(1)


            if self.window_size is not None:
                q_pos = torch.arange(old_pos_ptr, old_pos_ptr + tok_q, device=x.device)
                k_pos = torch.arange(old_pos_ptr + tok_q - tok_k, old_pos_ptr + tok_q, device=x.device)

                mask = (q_pos.unsqueeze(-1) < k_pos.unsqueeze(0)) | ((q_pos.unsqueeze(-1) - k_pos.unsqueeze(0)) >= self.window_size)
                attn_mask = mask
                is_causal = False
            else:
                attn_mask = None
                is_causal = False

            # Обновляем указатель после построения маски
            self.pos_ptr += tok_q

        else:
            # Без кэша: обрабатываем все токены
            latent_for_attn = kv_down_projected
            self.pos_ptr = 0
            tok_k = n_tokens
            attn_mask = None
            is_causal = False

        # Восстанавливаем K и V из сжатых представлений
        up_k = self.latent_up_proj_k(latent_for_attn)
        up_v = self.latent_up_proj_v(latent_for_attn)


        q = q.view(batch_size, n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k = up_k.view(batch_size, tok_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = up_v.view(batch_size, tok_k, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin, offset=0)
        k = apply_rope(k, cos, sin, offset=0)

        # Вычисление внимания
        attn_out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=1.0 / (self.head_dim ** 0.5)
        )

        # Обратный reshape и проекция
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, n_tokens, emb_dim)
        out = self.o(attn_out)
        return out



class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.w1 = nn.Linear(cfg['emb_dim'], int(cfg['intermed_dim']), bias=False)
        self.w2 = nn.Linear(cfg['emb_dim'], int(cfg['intermed_dim']), bias=False)
        self.w3 = nn.Linear(int(cfg['intermed_dim']), cfg['emb_dim'], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = nn.functional.silu(self.w1(x))

        data = self.w2(x)

        return self.w3(gate * data)



class TransformerBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.attn=SlidingWindowLatentAttention(cfg['emb_dim'],
                                               cfg['n_heads'],
                                               cfg['compression_ratio'],
                                               cfg['sliding_window_size'],
                                               cfg['qkv_bias'])


        self.ffn = FFN(cfg)
        self.norm1 = nn.RMSNorm(cfg['emb_dim'])
        self.norm2 = nn.RMSNorm(cfg['emb_dim'])


    def forward(self, x:torch.Tensor, sin, cos, use_cache=False) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn.forward(x, sin, cos, use_cache=use_cache)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + shortcut

        return x



class DSLikeModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])

        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['n_layers'])])

        self.final_norm = nn.RMSNorm(cfg['emb_dim'])

        self.lm_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=cfg['qkv_bias'])

        cos, sin = compute_rope_params(
            head_dim=cfg['emb_dim']//cfg['n_heads'],
            theta_base=cfg["rope_base"],
            context_length=cfg["n_ctx"]
        )

        self.register_buffer('sin', sin, persistent=False)
        self.register_buffer('cos', cos, persistent=False)


    def forward(self, x: torch.Tensor, use_cache=False) -> torch.Tensor:
        token_embeddings = self.emb(x)
        x = token_embeddings
        for block in self.trf_blocks:
            x = block(x, self.sin, self.cos, use_cache=use_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


    def reset_cache(self) -> None:
        for block in self.trf_blocks:
            block.attn.cache_kv = None
            block.attn.pos_ptr = 0




