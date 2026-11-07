import tkthread
import torch
import triton
import triton.testing
from triton_kernels.swiglu import SwiGLUFFN
from swiglu_ffn_triton_forward import SwiGLU_ffn
import matplotlib
from torch import nn
import torch.nn.functional as F

DTYPE = torch.bfloat16
DEVICE = 'cuda'


class PyTorchSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.w_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.v_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.d_proj = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.w_proj(x)
        gate = self.v_proj(x)
        return self.d_proj(F.silu(up) * gate)


def make_module(K, N, dtype=None, requires_grad=False):
    dtype = dtype or DTYPE
    model = PyTorchSwiGLU(K, N).to(DEVICE).to(dtype)
    for p in model.parameters():
        p.requires_grad_(requires_grad)
    return model


def make_input(M, K, dtype=None, requires_grad=False):
    dtype = dtype or DTYPE
    return torch.randn(M, K, device=DEVICE, dtype=dtype, requires_grad=requires_grad)


def extract_weights(model, requires_grad=False):
    """
    nn.Linear хранит [N, K]. SwiGLUFFN ждёт [K, N].
    .contiguous() — для коалесцированного чтения в Triton.
    .detach().requires_grad_(True) — делает тензоры LEAF.
    """
    w = model.w_proj.weight.t().contiguous().detach()
    g = model.v_proj.weight.t().contiguous().detach()
    d = model.d_proj.weight.t().contiguous().detach()
    if requires_grad:
        w.requires_grad_(True)
        g.requires_grad_(True)
        d.requires_grad_(True)
    return w, g, d



def flops_fwd(M, K_in, N_exp, K_out):

    return 2 * M * K_in * N_exp + 2 * M * K_in * N_exp + 2 * M * N_exp * K_out


def flops_bwd(M, K_in, N_exp, K_out, recompute=False):

    base = (2 * M * K_out * N_exp    
            + 2 * N_exp * M * K_out    
            + 4 * M * N_exp * K_in    
            + 4 * K_in * M * N_exp)   
    if recompute:
        base += 4 * M * K_in * N_exp   
    return base


def flops_fwd_bwd(M, K_in, N_exp, K_out, recompute=False):
    return flops_fwd(M, K_in, N_exp, K_out) + flops_bwd(M, K_in, N_exp, K_out, recompute)


def bytes_fwd(M, K_in, N_exp, K_out, elem_size=2):
    reads  = (M * K_in + 2 * K_in * N_exp + N_exp * K_out + M * N_exp) * elem_size
    writes = (M * N_exp + M * K_out) * elem_size
    return reads + writes



def _forward_fn(provider, x, model):
    """Inference: torch — model(x) под no_grad; triton — SwiGLU_ffn(..., training=False)."""
    if provider == 'torch':
        return lambda: model(x)
    w, g, d = extract_weights(model)
    return lambda: SwiGLU_ffn(x, w, g, d, training=False)


def _fwd_bwd_fn(provider, x, model, go):
    if provider == 'torch':
        def step():
            y = model(x)
            return torch.autograd.grad(y, [x] + list(model.parameters()), go)
    else:
        w, g, d = extract_weights(model, requires_grad=True)   # leaf
        def step():
            y = SwiGLUFFN.apply(x, w, g, d)
            return torch.autograd.grad(y, [x, w, g, d], go)
    return step


def _bwd_only_fn(provider, x, model, go):
    if provider == 'torch':
        y = model(x)
        params = [x] + list(model.parameters())
        return lambda: torch.autograd.grad(y, params, go, retain_graph=True)
    else:
        w, g, d = extract_weights(model, requires_grad=True)   # leaf
        y = SwiGLUFFN.apply(x, w, g, d)
        return lambda: torch.autograd.grad(y, [x, w, g, d], go, retain_graph=True)



@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['M'], x_vals=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='swiglu_fwd_tflops_by_M',
    args={'K': 1920, 'N': 5120},
))
def bench_fwd_tflops_by_M(M, K, N, provider):
    model = make_module(K, N, requires_grad=False)
    x = make_input(M, K, requires_grad=False)
    fn = _forward_fn(provider, x, model)
    with torch.no_grad():
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return flops_fwd(M, K, N, K) / (ms * 1e-3) / 1e12


@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['K'], x_vals=[512, 1024, 2048, 3072, 4096, 6144, 8192],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='swiglu_fwd_tflops_by_K',
    args={'M': 8192, 'N': 5120},
))
def bench_fwd_tflops_by_K(M, K, N, provider):
    model = make_module(K, N, requires_grad=False)
    x = make_input(M, K, requires_grad=False)
    fn = _forward_fn(provider, x, model)
    with torch.no_grad():
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return flops_fwd(M, K, N, K) / (ms * 1e-3) / 1e12


@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['N'], x_vals=[1024, 2048, 3072, 4096, 5120, 6144, 8192],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='swiglu_fwd_tflops_by_N',
    args={'M': 8192, 'K': 1920},
))
def bench_fwd_tflops_by_N(M, K, N, provider):
    model = make_module(K, N, requires_grad=False)
    x = make_input(M, K, requires_grad=False)
    fn = _forward_fn(provider, x, model)
    with torch.no_grad():
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return flops_fwd(M, K, N, K) / (ms * 1e-3) / 1e12



@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['M'], x_vals=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='swiglu_fwd_bwd_tflops_by_M',
    args={'K': 1920, 'N': 5120},
))
def bench_fwd_bwd_tflops_by_M(M, K, N, provider):
    model = make_module(K, N, requires_grad=True)
    x = make_input(M, K, requires_grad=True)
    go = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    step = _fwd_bwd_fn(provider, x, model, go)
    ms = triton.testing.do_bench(step, warmup=25, rep=100)

    recompute = False
    return flops_fwd_bwd(M, K, N, K, recompute) / (ms * 1e-3) / 1e12


@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['K'], x_vals=[512, 1024, 2048, 3072, 4096, 6144, 8192],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='swiglu_fwd_bwd_tflops_by_K',
    args={'M': 8192, 'N': 5120},
))
def bench_fwd_bwd_tflops_by_K(M, K, N, provider):
    model = make_module(K, N, requires_grad=True)
    x = make_input(M, K, requires_grad=True)
    go = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    step = _fwd_bwd_fn(provider, x, model, go)
    ms = triton.testing.do_bench(step, warmup=25, rep=100)
    return flops_fwd_bwd(M, K, N, K, recompute=False) / (ms * 1e-3) / 1e12


@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['N'], x_vals=[1024, 2048, 3072, 4096, 5120, 6144, 8192],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='swiglu_fwd_bwd_tflops_by_N',
    args={'M': 8192, 'K': 1920},
))
def bench_fwd_bwd_tflops_by_N(M, K, N, provider):
    model = make_module(K, N, requires_grad=True)
    x = make_input(M, K, requires_grad=True)
    go = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    step = _fwd_bwd_fn(provider, x, model, go)
    ms = triton.testing.do_bench(step, warmup=25, rep=100)
    return flops_fwd_bwd(M, K, N, K, recompute=False) / (ms * 1e-3) / 1e12



@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['M'], x_vals=[256, 512, 1024, 2048, 4096, 8192],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='swiglu_bwd_tflops_by_M',
    args={'K': 1920, 'N': 5120},
))
def bench_bwd_tflops_by_M(M, K, N, provider):
    model = make_module(K, N, requires_grad=True)
    x = make_input(M, K, requires_grad=True)
    go = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    step = _bwd_only_fn(provider, x, model, go)
    ms = triton.testing.do_bench(step, warmup=25, rep=100)
    return flops_bwd(M, K, N, K, recompute=False) / (ms * 1e-3) / 1e12



@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['N'], x_vals=[256, 512, 1024, 2048, 4096, 8192],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='GB/s', plot_name='swiglu_fwd_bandwidth_by_N',
    args={'M': 8192, 'K': 1920},
))
def bench_fwd_bandwidth_by_N(M, K, N, provider):
    model = make_module(K, N, requires_grad=False)
    x = make_input(M, K, requires_grad=False)
    fn = _forward_fn(provider, x, model)
    with torch.no_grad():
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return bytes_fwd(M, K, N, K) / (ms * 1e-3) / 1e9



def measure_memory(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    for _ in range(iters):
        out = fn()
        del out
        torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    return {
        "base_MB":  base / 2**20,
        "peak_MB":  peak / 2**20,
        "delta_MB": (peak - base) / 2**20,
    }


def bench_memory(M=8192, K=1920, N=5120):
    print(f"\nMemory (M={M}, K={K}, N={N}, dtype={DTYPE})")
    print("-" * 75)


    model = make_module(K, N, requires_grad=False)
    x = make_input(M, K, requires_grad=False)
    w, g, d = extract_weights(model)

    with torch.no_grad():
        m_tf = measure_memory(lambda: model(x))
        m_tt = measure_memory(lambda: SwiGLU_ffn(x, w, g, d, training=False))

    print("Forward (inference, no cache):")
    print(f"  PyTorch : peak={m_tf['peak_MB']:8.1f} MB  delta={m_tf['delta_MB']:8.1f} MB")
    print(f"  Triton  : peak={m_tt['peak_MB']:8.1f} MB  delta={m_tt['delta_MB']:8.1f} MB")
    dlt = m_tf['delta_MB'] - m_tt['delta_MB']
    pct = 100 * dlt / m_tf['delta_MB'] if m_tf['delta_MB'] else 0
    print(f" экономия {dlt:+.1f} MB ({pct:+.1f}%)")


    model_t = make_module(K, N, requires_grad=True)
    w_t, g_t, d_t = extract_weights(model_t, requires_grad=True)

    m_tf = measure_memory(lambda: model_t(x))
    m_tt = measure_memory(lambda: SwiGLUFFN.apply(x, w_t, g_t, d_t))

    print("\nForward (training, cached u/v/h):")
    print(f"  PyTorch : peak={m_tf['peak_MB']:8.1f} MB  delta={m_tf['delta_MB']:8.1f} MB")
    print(f"  Triton  : peak={m_tt['peak_MB']:8.1f} MB  delta={m_tt['delta_MB']:8.1f} MB")
    dlt = m_tf['delta_MB'] - m_tt['delta_MB']
    pct = 100 * dlt / m_tf['delta_MB'] if m_tf['delta_MB'] else 0
    print(f" экономия {dlt:+.1f} MB ({pct:+.1f}%)")

    del model, model_t, w_t, g_t, d_t
    torch.cuda.empty_cache()


    model = make_module(K, N, requires_grad=True)
    x = make_input(M, K, requires_grad=True)
    go = torch.randn(M, K, device=DEVICE, dtype=DTYPE)

    def torch_fb():
        y = model(x)
        return torch.autograd.grad(y, [x] + list(model.parameters()), go)

    w, g, d = extract_weights(model, requires_grad=True)
    def triton_fb():
        y = SwiGLUFFN.apply(x, w, g, d)
        return torch.autograd.grad(y, [x, w, g, d], go)

    m_tf = measure_memory(torch_fb)
    m_tt = measure_memory(triton_fb)

    print("\nForward + Backward:")
    print(f"  PyTorch : peak={m_tf['peak_MB']:8.1f} MB  delta={m_tf['delta_MB']:8.1f} MB")
    print(f"  Triton  : peak={m_tt['peak_MB']:8.1f} MB  delta={m_tt['delta_MB']:8.1f} MB")
    dlt = m_tf['delta_MB'] - m_tt['delta_MB']
    pct = 100 * dlt / m_tf['delta_MB'] if m_tf['delta_MB'] else 0
    print(f" экономия {dlt:+.1f} MB ({pct:+.1f}%)")



def _rel_err(ref, got):
    return (ref - got).norm().item() / (ref.norm().item() + 1e-12)


_TOL_MAP = {torch.float32: 1e-3, torch.float16: 2e-2, torch.bfloat16: 3e-2}


def check_fwd_correctness(M=1024, K=512, N=2048, dtype=DTYPE, tol=None):
    tol = tol or _TOL_MAP[dtype]
    print(f"\nForward: M={M}, K={K}, N={N}, dtype={dtype}, tol={tol:.0e}")
    torch.manual_seed(42)
    model = make_module(K, N, dtype=dtype, requires_grad=False)
    x = make_input(M, K, dtype=dtype, requires_grad=False)

    with torch.no_grad():
        y_ref = model(x)
        y_tri = SwiGLU_ffn(x, *extract_weights(model), training=False)

    err = _rel_err(y_ref, y_tri)
    ok = err < tol
    print(f"  rel_err={err:.3e}  {'OK' if ok else 'FAIL'}")
    return ok


def check_bwd_correctness(M=1024, K=512, N=2048, dtype=DTYPE, tol=None):
    tol = tol or _TOL_MAP[dtype]
    print(f"\nBackward: M={M}, K={K}, N={N}, dtype={dtype}, tol={tol:.0e}")
    torch.manual_seed(42)
    model = make_module(K, N, dtype=dtype, requires_grad=True)
    x = make_input(M, K, dtype=dtype, requires_grad=True)
    go = torch.randn(M, K, device=DEVICE, dtype=dtype)


    y_ref = model(x)
    grads_ref = torch.autograd.grad(y_ref, [x] + list(model.parameters()), go)


    w, g, d = extract_weights(model, requires_grad=True)
    y_tri = SwiGLUFFN.apply(x, w, g, d)
    grads_tri = torch.autograd.grad(y_tri, [x, w, g, d], go)

    all_ok = True
    targets = [('dx', grads_ref[0], grads_tri[0], False),
               ('dw', grads_ref[1], grads_tri[1], True),
               ('dg', grads_ref[2], grads_tri[2], True),
               ('dd', grads_ref[3], grads_tri[3], True)]

    for name, ref, got, do_t in targets:
        if do_t:
            ref = ref.t()
        err = _rel_err(ref, got)
        ok = err < tol
        all_ok &= ok
        print(f"  {name}: rel_err={err:.3e}  {'OK' if ok else 'FAIL'}")
    return all_ok



if __name__ == "__main__":
    print("CORRECTNESS")
    check_fwd_correctness()
    check_bwd_correctness()


    print("MEMORY")
    bench_memory(M=8192, K=1920, N=5120)


    print("FORWARD TFLOPS")
    bench_fwd_tflops_by_M.run(show_plots=True, print_data=True)
    bench_fwd_tflops_by_K.run(show_plots=True, print_data=True)
    bench_fwd_tflops_by_N.run(show_plots=True, print_data=True)


    print("FWD+BWD TFLOPS  (cache_uv=True)")
    bench_fwd_bwd_tflops_by_M.run(show_plots=True, print_data=True)
    bench_fwd_bwd_tflops_by_K.run(show_plots=True, print_data=True)
    bench_fwd_bwd_tflops_by_N.run(show_plots=True, print_data=True)


    print("BWD-ONLY TFLOPS  (cache_uv=True)")
    bench_bwd_tflops_by_M.run(show_plots=True, print_data=True)


    print("FORWARD BANDWIDTH")
    bench_fwd_bandwidth_by_N.run(show_plots=True, print_data=True)
