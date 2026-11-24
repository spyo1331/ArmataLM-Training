import torch
import triton
import math
from triton_kernels.sdpa import trit_attn
DEVICE = 'cuda'
DTYPE = torch.float16





def test_FA(B,H,N,Dh, device=DEVICE, atol=1e-2):
    q = torch.randn((B,H,N,Dh), dtype=torch.float16, device=device, requires_grad=True)
    k = torch.randn((B,H,N,Dh), dtype=torch.float16, device=device, requires_grad=True)
    v = torch.randn((B,H,N,Dh), dtype=torch.float16, device=device, requires_grad=True)
    scale = 1/math.sqrt(Dh)
    tri_out = trit_attn(q,k,v,scale)
    ref_out = torch.nn.functional.scaled_dot_product_attention(q,k,v, is_causal=True, scale=scale)
    torch.testing.assert_close(tri_out, ref_out, atol=atol, rtol=0.0)
    print('passed fwd')

    dldout = 0.1 * torch.randn_like(q)

    tri_out.backward(dldout, retain_graph=True)
    dldq_tri, dldk_tri, dldv_tri = [_.grad.clone() for _ in [q,k,v]]
    q.grad, k.grad, v.grad = None, None, None

    ref_out.backward(dldout, retain_graph=True)
    dldq_ref, dldk_ref, dldv_ref = [_.grad.clone() for _ in [q,k,v]]
    q.grad, k.grad, v.grad = None, None, None

    torch.testing.assert_close(dldq_tri, dldq_ref, atol=atol, rtol=0.)
    torch.testing.assert_close(dldk_tri, dldk_ref, atol=atol, rtol=0.)
    torch.testing.assert_close(dldv_tri, dldv_ref, atol=atol, rtol=0.)
    print('passed bwd')



def flops_attn_fwd(B, H, N, Dh):

    return 0.5 * B * H * 4 * N * N * Dh


def flops_attn_bwd(B, H, N, Dh):

    return 0.5 * B * H * 10 * N * N * Dh


def flops_attn_fwd_bwd(B, H, N, Dh):
    return flops_attn_fwd(B, H, N, Dh) + flops_attn_bwd(B, H, N, Dh)


def bytes_attn_fwd(B, H, N, Dh):

    es = torch.empty(0, dtype=DTYPE).element_size()
    return 4 * B * H * N * Dh * es



def _attn_fwd_fn(provider, q, k, v, scale):
    if provider == 'torch':
        return lambda: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale)
    return lambda: trit_attn(q, k, v, scale)


def _attn_fwd_bwd_step(provider, q, k, v, go, scale):
    if provider == 'torch':
        def step():
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=scale)
            out.backward(go)
            q.grad = k.grad = v.grad = None
    else:
        def step():
            out = trit_attn(q, k, v, scale)
            out.backward(go)
            q.grad = k.grad = v.grad = None
    return step


def _attn_bwd_only_fn(provider, q, k, v, go, scale):
    if provider == 'torch':
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale)
    else:
        out = trit_attn(q, k, v, scale)
    return lambda: out.backward(go, retain_graph=True)


def make_attn_inputs(B, H, N, Dh, requires_grad=False):
    q = torch.randn((B, H, N, Dh), dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)
    k = torch.randn((B, H, N, Dh), dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)
    v = torch.randn((B, H, N, Dh), dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)
    return q, k, v



@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['N'], x_vals=[512, 1024, 2048, 4096, 8192, 16384],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch SDPA', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='attn_fwd_tflops_by_N',
    args={'B': 4, 'H': 16, 'Dh': 64},
))
def bench_attn_fwd_tflops_by_N(B, H, N, Dh, provider):
    q, k, v = make_attn_inputs(B, H, N, Dh)
    scale = 1 / math.sqrt(Dh)
    ms = triton.testing.do_bench(_attn_fwd_fn(provider, q, k, v, scale), warmup=25, rep=100)
    return flops_attn_fwd(B, H, N, Dh) / (ms * 1e-3) / 1e12


@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['Dh'], x_vals=[16, 32, 64, 128],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch SDPA', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='attn_fwd_tflops_by_Dh',
    args={'B': 4, 'H': 16, 'N': 8192},
))
def bench_attn_fwd_tflops_by_Dh(B, H, N, Dh, provider):
    q, k, v = make_attn_inputs(B, H, N, Dh)
    scale = 1 / math.sqrt(Dh)
    ms = triton.testing.do_bench(_attn_fwd_fn(provider, q, k, v, scale), warmup=25, rep=100)
    return flops_attn_fwd(B, H, N, Dh) / (ms * 1e-3) / 1e12



@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['N'], x_vals=[512, 1024, 2048, 4096, 8192, 16384],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch SDPA', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='attn_fwd_bwd_tflops_by_N',
    args={'B': 4, 'H': 16, 'Dh': 64},
))
def bench_attn_fwd_bwd_tflops_by_N(B, H, N, Dh, provider):
    q, k, v = make_attn_inputs(B, H, N, Dh, requires_grad=True)
    go = torch.randn((B, H, N, Dh), dtype=DTYPE, device=DEVICE)
    scale = 1 / math.sqrt(Dh)
    step = _attn_fwd_bwd_step(provider, q, k, v, go, scale)
    ms = triton.testing.do_bench(step, warmup=25, rep=100)
    return flops_attn_fwd_bwd(B, H, N, Dh) / (ms * 1e-3) / 1e12


@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['Dh'], x_vals=[16, 32, 64, 128],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch SDPA', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='attn_fwd_bwd_tflops_by_Dh',
    args={'B': 4, 'H': 16, 'N': 8192},
))
def bench_attn_fwd_bwd_tflops_by_Dh(B, H, N, Dh, provider):
    q, k, v = make_attn_inputs(B, H, N, Dh, requires_grad=True)
    go = torch.randn((B, H, N, Dh), dtype=DTYPE, device=DEVICE)
    scale = 1 / math.sqrt(Dh)
    step = _attn_fwd_bwd_step(provider, q, k, v, go, scale)
    ms = triton.testing.do_bench(step, warmup=25, rep=100)
    return flops_attn_fwd_bwd(B, H, N, Dh) / (ms * 1e-3) / 1e12



@triton.testing.perf_report(triton.testing.Benchmark(
    x_names=['N'], x_vals=[512, 1024, 2048, 4096, 8192, 16384],
    x_log=False, line_arg='provider',
    line_vals=['torch', 'triton'], line_names=['PyTorch SDPA', 'Triton'],
    styles=[('blue', '-'), ('red', '-')],
    ylabel='TFLOPS', plot_name='attn_bwd_tflops_by_N',
    args={'B': 4, 'H': 16, 'Dh': 64},
))
def bench_attn_bwd_tflops_by_N(B, H, N, Dh, provider):
    q, k, v = make_attn_inputs(B, H, N, Dh, requires_grad=True)
    go = torch.randn((B, H, N, Dh), dtype=DTYPE, device=DEVICE)
    scale = 1 / math.sqrt(Dh)
    fn = _attn_bwd_only_fn(provider, q, k, v, go, scale)
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return flops_attn_bwd(B, H, N, Dh) / (ms * 1e-3) / 1e12



def measure_memory(fn, warmup=5, iters=20):
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


def bench_memory_attn(B=4, H=16):
    es = torch.empty(0, dtype=DTYPE).element_size()
    print(f"\nAttention memory (B={B}, H={H}, dtype={DTYPE}, elem={es} B)")
    print("-" * 100)
    print(f"{'N':>6} {'Dh':>4} | {'fwd torch':>12} {'fwd triton':>12} | "
          f"{'f+b torch':>12} {'f+b triton':>12} | {'экономия fwd':>14}")
    for N in [512, 1024, 2048, 4096, 8192, 16384]:
        for Dh in [16, 32, 64, 128]:
            scale = 1 / math.sqrt(Dh)

            # Forward
            q, k, v = make_attn_inputs(B, H, N, Dh)
            m_fwd_t = measure_memory(lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=scale))
            m_fwd_r = measure_memory(lambda: trit_attn(q, k, v, scale))
            del q, k, v
            torch.cuda.empty_cache()

            # Forward + Backward
            q, k, v = make_attn_inputs(B, H, N, Dh, requires_grad=True)
            go = torch.randn((B, H, N, Dh), dtype=DTYPE, device=DEVICE)

            def torch_fb():
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=scale)
                out.backward(go)
                q.grad = k.grad = v.grad = None

            def triton_fb():
                out = trit_attn(q, k, v, scale)
                out.backward(go)
                q.grad = k.grad = v.grad = None

            m_fb_t = measure_memory(torch_fb)
            m_fb_r = measure_memory(triton_fb)
            del q, k, v, go
            torch.cuda.empty_cache()

            save_fwd = m_fwd_t['delta_MB'] - m_fwd_r['delta_MB']
            print(f"{N:>6} {Dh:>4} | {m_fwd_t['delta_MB']:>10.1f}MB {m_fwd_r['delta_MB']:>10.1f}MB | "
                  f"{m_fb_t['delta_MB']:>10.1f}MB {m_fb_r['delta_MB']:>10.1f}MB | {save_fwd:>12.1f}MB")

if __name__ == '__main__':
    print(test_FA(1,1,64,32))
    print(test_FA(1, 1, 128, 64))
    print(test_FA(1, 1, 1024, 64))

    bench_attn_fwd_tflops_by_N.run(show_plots=True, print_data=True)
    bench_attn_fwd_tflops_by_Dh.run(show_plots=True, print_data=True)


    bench_attn_fwd_bwd_tflops_by_N.run(show_plots=True, print_data=True)
    bench_attn_fwd_bwd_tflops_by_Dh.run(show_plots=True, print_data=True)

    bench_attn_bwd_tflops_by_N.run(show_plots=True, print_data=True)

    bench_memory_attn()