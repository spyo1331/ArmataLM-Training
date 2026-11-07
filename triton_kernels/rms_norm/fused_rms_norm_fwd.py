import triton
import triton.language as tl
import torch





def fused_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    pass