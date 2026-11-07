from llm_architectures.gpt2_base import GPT2Model
from llm_architectures.DS_v2_like import DSLikeModel
from llm_architectures.Qwen3_5_like import Qwen3LikeModel
from llm_architectures.experemental_architecture.experimental_architecture import EXPModel

max_context = 33000

standart_gpt2_cfg = {
    "emb_dim": 1024,
    "num_heads": 16,
    "n_layers": 16,
    "n_ctx": 1024,
    "intermed_dim": int(1024*2.67),
    "vocab_size": 50257,
    "qkv_bias": False,
    "device": 'cuda'
}
gpt_model = GPT2Model(cfg=standart_gpt2_cfg)



dslikeModel_config = {
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "sliding_window_size": None,
    "compression_ratio": 4,
    "n_ctx":max_context,
    "vocab_size": 50257,
    "intermed_dim": 768*2.67,
    "rope_base": 10000,
    "qkv_bias": False
}
Ds_like_model = DSLikeModel(cfg=dslikeModel_config)


qwen3_5_likemodel_cfg  = {
    "emb_dim": 768,
    "vocab_size": 50257,
    "n_ctx": max_context,
    "n_heads": 12,
    "n_layers": 12,
    "intermed_dim": 768*2.67,
    'qkv_bias': False
}

qwen3_5_likemodel = Qwen3LikeModel(qwen3_5_likemodel_cfg)


cfg = {
    "emb_dim": 1536,
    "n_heads": 24,
    "n_layers": 27,
    "vocab_size": 50257,
    "n_kv_heads": 4,
    "compress_coef": 2,
    "intermed_dim": 4608,
    "qkv_bias": False
}
exp_model = EXPModel(cfg=cfg)