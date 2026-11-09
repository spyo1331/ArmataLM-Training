import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
from triton_kernels.rms_norm import FusedRMSNormalization

DEVICE = 'cuda'
DTYPE = torch.bfloat16

fused_rms_norm = FusedRMSNormalization.apply


def rms_norm_torch(x, g, eps=1e-6):
    """Эталонная реализация на чистом PyTorch."""
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (x / rms) * g


def rms_norm_bwd_torch(go, x, g, rms, eps=1e-6):
    """Эталонный бэкворд на чистом PyTorch."""
    B, M, N = x.shape

    y = x / rms.unsqueeze(-1)
    mean_term = (y * g * go).mean(dim=-1, keepdim=True)
    dx = (g * go - y * mean_term) / rms.unsqueeze(-1)


    dg = (go * y).sum(dim=(0, 1))

    return dx, dg



def _rel_err(ref, got):
    return (ref - got).norm().item() / (ref.norm().item() + 1e-12)


def test_forward_correctness(B=4, M=128, N=1024, eps=1e-6, tol=1e-5):
    """Тест корректности форварда."""
    print(f"\n{'=' * 70}")
    print(f"FORWARD CORRECTNESS: B={B}, M={M}, N={N}, dtype={DTYPE}")
    print(f"{'=' * 70}")

    torch.manual_seed(42)
    x = torch.randn(B, M, N, device=DEVICE, dtype=DTYPE)
    g = torch.randn(N, device=DEVICE, dtype=DTYPE)

    # PyTorch эталон
    y_ref = rms_norm_torch(x, g, eps)

    # Triton
    y_tri = fused_rms_norm(x, g, eps)

    err = _rel_err(y_ref, y_tri)
    max_diff = (y_ref - y_tri).abs().max().item()
    ok = err < tol

    print(f"  rel_err = {err:.3e}  tol={tol:.0e}  {'OK' if ok else 'FAIL'}")
    print(f"  max_diff = {max_diff:.3e}")
    return ok


def test_backward_correctness(B=4, M=128, N=1024, eps=1e-6, tol=1e-4):
    """Тест корректности бэкворда через autograd."""
    print(f"\n{'=' * 70}")
    print(f"BACKWARD CORRECTNESS: B={B}, M={M}, N={N}, dtype={DTYPE}")
    print(f"{'=' * 70}")

    torch.manual_seed(42)

    # PyTorch эталон
    x_ref = torch.randn(B, M, N, device=DEVICE, dtype=DTYPE, requires_grad=True)
    g_ref = torch.randn(N, device=DEVICE, dtype=DTYPE, requires_grad=True)
    go = torch.randn(B, M, N, device=DEVICE, dtype=DTYPE)

    y_ref = rms_norm_torch(x_ref, g_ref, eps)
    y_ref.backward(go)

    # Triton
    x_tri = x_ref.detach().clone().requires_grad_(True)
    g_tri = g_ref.detach().clone().requires_grad_(True)

    y_tri = fused_rms_norm(x_tri, g_tri, eps)
    y_tri.backward(go)


    err_y = _rel_err(y_ref, y_tri)
    err_dx = _rel_err(x_ref.grad, x_tri.grad)
    err_dg = _rel_err(g_ref.grad, g_tri.grad)

    print(f"y:  rel_err = {err_y:.3e}")
    print(f"dx: rel_err = {err_dx:.3e}  {'OK' if err_dx < tol else 'FAIL'}")
    print(f"dg: rel_err = {err_dg:.3e}  {'OK' if err_dg < tol else 'FAIL'}")

    return err_dx < tol and err_dg < tol


def test_gradcheck(B=2, M=4, N=32, eps=1e-6):
    """Строгий тест градиентов через torch.autograd.gradcheck (требует float64)."""
    print(f"\n{'=' * 70}")
    print(f"GRADCHECK: B={B}, M={M}, N={N}, dtype=float64")
    print(f"{'=' * 70}")

    torch.manual_seed(42)
    x = torch.randn(B, M, N, device=DEVICE, dtype=torch.float64, requires_grad=True)
    g = torch.randn(N, device=DEVICE, dtype=torch.float64, requires_grad=True)

    def fn(x_, g_):
        return FusedRMSNormalization.apply(x_, g_, eps)

    try:
        ok = torch.autograd.gradcheck(fn, (x, g), eps=1e-6, atol=1e-4, rtol=1e-3)
        print(f"  gradcheck: {'PASSED' if ok else 'FAILED'}")
        return ok
    except Exception as e:
        print(f"  gradcheck: FAILED with {e}")
        return False


def test_2d_input(M=128, N=1024, eps=1e-6):
    """Тест для 2D входа (без батча)."""
    print(f"\n{'=' * 70}")
    print(f"2D INPUT TEST: M={M}, N={N}")
    print(f"{'=' * 70}")

    torch.manual_seed(42)
    x = torch.randn(M, N, device=DEVICE, dtype=DTYPE, requires_grad=True)
    g = torch.randn(N, device=DEVICE, dtype=DTYPE, requires_grad=True)
    go = torch.randn(M, N, device=DEVICE, dtype=DTYPE)

    # PyTorch
    x_ref = x.detach().clone().requires_grad_(True)
    g_ref = g.detach().clone().requires_grad_(True)
    y_ref = rms_norm_torch(x_ref.unsqueeze(0), g_ref, eps).squeeze(0)
    y_ref.backward(go)

    # Triton
    y_tri = fused_rms_norm(x, g, eps)
    y_tri.backward(go)

    err_y = _rel_err(y_ref, y_tri)
    err_dx = _rel_err(x_ref.grad, x.grad)
    err_dg = _rel_err(g_ref.grad, g.grad)

    print(f"  y:  rel_err = {err_y:.3e}")
    print(f"  dx: rel_err = {err_dx:.3e}")
    print(f"  dg: rel_err = {err_dg:.3e}")

    return err_dx < 1e-4 and err_dg < 1e-4



def flops_fwd(B, M, N):
    """
    FLOPS для форварда:
    - Возведение в квадрат: N
    - Сумма: N (редукция)
    - Деление и умножение: 3N (x/rms, *g)
    """
    return 5 * B * M * N


def flops_bwd(B, M, N):
    """
    FLOPS для бэкворда:
    - Градиент по входу: ~6N на строку
    - Градиент по весу: ~3N на строку (умножение + редукция)
    """
    return 9 * B * M * N


def flops_fwd_bwd(B, M, N):
    return flops_fwd(B, M, N) + flops_bwd(B, M, N)



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        x_log=False,
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='TFLOPS',
        plot_name='rmsnorm_fwd_tflops_by_M',
        args={'B': 4, 'N': 2048},
    )
)
def bench_fwd_tflops_by_M(B, M, N, provider):
    x = torch.randn(B, M, N, device=DEVICE, dtype=DTYPE)
    g = torch.randn(N, device=DEVICE, dtype=DTYPE)

    if provider == 'torch':
        fn = lambda: rms_norm_torch(x, g)
    else:
        fn = lambda: fused_rms_norm(x, g)

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return flops_fwd(B, M, N) / (ms * 1e-3) / 1e12


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512, 1024, 2048, 4096, 8192],
        x_log=False,
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='TFLOPS',
        plot_name='rmsnorm_fwd_tflops_by_N',
        args={'B': 4, 'M': 2048},
    )
)
def bench_fwd_tflops_by_N(B, M, N, provider):
    x = torch.randn(B, M, N, device=DEVICE, dtype=DTYPE)
    g = torch.randn(N, device=DEVICE, dtype=DTYPE)

    if provider == 'torch':
        fn = lambda: rms_norm_torch(x, g)
    else:
        fn = lambda: fused_rms_norm(x, g)

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return flops_fwd(B, M, N) / (ms * 1e-3) / 1e12


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        x_log=False,
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='TFLOPS',
        plot_name='rmsnorm_fwd_bwd_tflops_by_M',
        args={'B': 4, 'N': 2048},
    )
)
def bench_fwd_bwd_tflops_by_M(B, M, N, provider):
    x = torch.randn(B, M, N, device=DEVICE, dtype=DTYPE, requires_grad=True)
    g = torch.randn(N, device=DEVICE, dtype=DTYPE, requires_grad=True)
    go = torch.randn(B, M, N, device=DEVICE, dtype=DTYPE)

    if provider == 'torch':
        def fn():
            y = rms_norm_torch(x, g)
            return torch.autograd.grad(y, [x, g], go)
    else:
        def fn():
            y = fused_rms_norm(x, g)
            return torch.autograd.grad(y, [x, g], go)

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return flops_fwd_bwd(B, M, N) / (ms * 1e-3) / 1e12


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        x_log=False,
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='rmsnorm_fwd_bandwidth_by_M',
        args={'B': 4, 'N': 2048},
    )
)
def bench_fwd_bandwidth_by_M(B, M, N, provider):
    x = torch.randn(B, M, N, device=DEVICE, dtype=DTYPE)
    g = torch.randn(N, device=DEVICE, dtype=DTYPE)

    if provider == 'torch':
        fn = lambda: rms_norm_torch(x, g)
    else:
        fn = lambda: fused_rms_norm(x, g)

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)


    elem_size = 4 if DTYPE==torch.float32 else 2
    bytes_moved = (B * M * N + N + B * M * N + B * M) * elem_size
    return bytes_moved / (ms * 1e-3) / 1e9



def measure_memory(fn, warmup=10, iters=50):
    """Измеряет пиковое потребление памяти."""

    for _ in range(warmup):
        out = fn()
        if isinstance(out, tuple):
            del out
        else:
            del out

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
        "base_MB": base / 2 ** 20,
        "peak_MB": peak / 2 ** 20,
        "delta_MB": (peak - base) / 2 ** 20,
    }


def bench_memory(B=4, M=4096, N=2048, eps=1e-6):
    """Сравнение потребления памяти между PyTorch и Triton."""
    print(f"\n{'=' * 70}")
    print(f"MEMORY BENCHMARK: B={B}, M={M}, N={N}, dtype={DTYPE}")
    print(f"{'=' * 70}")

    x = torch.randn(B, M, N, device=DEVICE, dtype=DTYPE, requires_grad=True)
    g = torch.randn(N, device=DEVICE, dtype=DTYPE, requires_grad=True)
    go = torch.randn(B, M, N, device=DEVICE, dtype=DTYPE)


    print("\nForward only:")

    m_pt = measure_memory(lambda: rms_norm_torch(x, g, eps))
    m_tri = measure_memory(lambda: fused_rms_norm(x, g, eps))

    print(f"  PyTorch : peak={m_pt['peak_MB']:8.1f} MB  delta={m_pt['delta_MB']:8.1f} MB")
    print(f"  Triton  : peak={m_tri['peak_MB']:8.1f} MB  delta={m_tri['delta_MB']:8.1f} MB")

    delta = m_pt['delta_MB'] - m_tri['delta_MB']
    pct = 100 * delta / m_pt['delta_MB'] if m_pt['delta_MB'] else 0
    print(f"  → экономия {delta:+.1f} MB ({pct:+.1f}%)")


    print("\nForward + Backward:")

    def torch_fwd_bwd():
        y = rms_norm_torch(x, g, eps)
        return torch.autograd.grad(y, [x, g], go)

    def triton_fwd_bwd():
        y = fused_rms_norm(x, g, eps)
        return torch.autograd.grad(y, [x, g], go)

    m_pt = measure_memory(torch_fwd_bwd)
    m_tri = measure_memory(triton_fwd_bwd)

    print(f"  PyTorch : peak={m_pt['peak_MB']:8.1f} MB  delta={m_pt['delta_MB']:8.1f} MB")
    print(f"  Triton  : peak={m_tri['peak_MB']:8.1f} MB  delta={m_tri['delta_MB']:8.1f} MB")

    delta = m_pt['delta_MB'] - m_tri['delta_MB']
    pct = 100 * delta / m_pt['delta_MB'] if m_pt['delta_MB'] else 0
    print(f"  → экономия {delta:+.1f} MB ({pct:+.1f}%)")



if __name__ == "__main__":
    print("=" * 70)
    print("CORRECTNESS TESTS")
    print("=" * 70)

    test_forward_correctness()
    test_backward_correctness()
    test_gradcheck()
    test_2d_input()

    print("\n" + "=" * 70)
    print("MEMORY BENCHMARK")
    print("=" * 70)
    bench_memory(B=4, M=4096, N=2048)

    print("\n" + "=" * 70)
    print("FORWARD TFLOPS")
    print("=" * 70)
    bench_fwd_tflops_by_M.run(show_plots=True, print_data=True)
    bench_fwd_tflops_by_N.run(show_plots=True, print_data=True)

    print("\n" + "=" * 70)
    print("FWD+BWD TFLOPS")
    print("=" * 70)
    bench_fwd_bwd_tflops_by_M.run(show_plots=True, print_data=True)

    print("\n" + "=" * 70)
    print("FORWARD BANDWIDTH")
    print("=" * 70)
    bench_fwd_bandwidth_by_M.run(show_plots=True, print_data=True)