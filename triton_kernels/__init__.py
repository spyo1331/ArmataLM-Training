from triton_kernels.sdpa import TritonFlashAttention
from triton_kernels.rms_norm import FusedRMSNormalization
from triton_kernels.swiglu import SwiGLUFFN

triton_FA = TritonFlashAttention.apply
triton_rmsnorm = FusedRMSNormalization.apply
triton_swiglu = SwiGLUFFN.apply


