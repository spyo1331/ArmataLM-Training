import torch
import functools

def make_tensors_contiguous(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        args   = tuple(a if not isinstance(a, torch.Tensor) or a.is_contiguous() else a.contiguous()
                       for a in args)
        kwargs = {k: (v if not isinstance(v, torch.Tensor) or v.is_contiguous() else v.contiguous())
                  for k, v in kwargs.items()}
        return f(*args, **kwargs)
    return wrapper