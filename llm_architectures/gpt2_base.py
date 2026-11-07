import torch
from torch import nn






class MultiHeadAttentionVanilla(nn.Module):
    """Multi Head Attention class that used in GPT 2.0 and "Attention is all you need" paper 2017 \n
        args:\n
        - num_heads - num attention heads;\n
        - emb_dim - embedding dimension;\n
        - n_ctx - sets the limit of the sequence length that can be passed through attention;\n
        - bias - bias in q,k,v,o matrices. Default is False
    """
    def __init__(self, num_heads: int, emb_dim: int, n_ctx: int, bias=False) -> None:
        super().__init__()
        assert emb_dim%num_heads==0, 'emb dim must be divisible by num attention heads'
        self.head_dim = emb_dim//num_heads
        self.n_heads = num_heads
        self.emb_dim = emb_dim
        self.q = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.k = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.v = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.o = nn.Linear(emb_dim, emb_dim, bias=bias)

        self.register_buffer('mask', torch.triu(torch.ones(n_ctx, n_ctx), diagonal=1).bool(), persistent=False)
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0



    def forward(self, x: torch.Tensor, use_cache=False):
        batch_size, num_tokens, emb_dim = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        keys_new = k.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        values_new = v.view(batch_size, num_tokens, self.n_heads, self.head_dim)
        q = q.view(batch_size, num_tokens, self.n_heads, self.head_dim)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = keys_new, values_new
            else:
                self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
                self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
            k, v = self.cache_k, self.cache_v
        else:
            k, v = keys_new, values_new

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)


        num_tokens_q = q.shape[-2]
        num_tokens_k = k.shape[-2]
        if use_cache:
            mask_bool = self.mask.bool()[
                self.ptr_current_pos:self.ptr_current_pos + num_tokens_q, :num_tokens_k
            ]
            self.ptr_current_pos += num_tokens_q

        else:
            mask_bool = self.mask.bool()[:num_tokens_q, :num_tokens_k]

        context_vec = torch.nn.functional.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask_bool, scale=1/(self.head_dim**0.5))
        attn_out = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, emb_dim)
        out = self.o(attn_out)
        return out





    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0






class TransformerBlock(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        self.pre_norm1 = nn.LayerNorm(cfg['emb_dim'])
        self.pre_norm2 = nn.LayerNorm(cfg['emb_dim'])
        self.attn = MultiHeadAttentionVanilla(cfg['num_heads'], cfg['emb_dim'], cfg['n_ctx'], cfg['qkv_bias'])
        self.mlp = nn.Sequential(
            nn.Linear(cfg['emb_dim'], cfg['intermed_dim']),
            nn.GELU(approximate='tanh'),
            nn.Linear(cfg['intermed_dim'], cfg['emb_dim'])
        )



    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.pre_norm1(x)
        x = self.attn(x, use_cache=use_cache)
        x = shortcut + x

        shortcut = x
        x = self.pre_norm2(x)
        x = self.mlp(x)
        x = shortcut + x

        return x




class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'], device=cfg['device'])
        self.pos_emb = nn.Embedding(cfg['n_ctx'], cfg['emb_dim'], device=cfg['device'])

        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['n_layers'])])

        self.final_norm = nn.LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'])

        self.current_pos = 0




    def forward(self, x: torch.Tensor, use_cache=False):
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)


        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=x.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=x.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)


        x = tok_embeds + pos_embeds


        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)


        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits




    def reset_cache(self):
        for blk in self.trf_blocks:
            blk.attn.reset_cache()
        self.current_pos = 0



