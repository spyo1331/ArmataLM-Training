import gigatoken
import wandb
import torch
from llm_architectures import gpt_model, Ds_like_model, max_context, qwen3_5_likemodel, armataLM
import gigatoken as gt
import loguru as lg
import time
from typing import Optional
from pydantic import BaseModel
import gc
import argparse

tokenizer = gt.Tokenizer('gpt2').as_hf()
def generate_tokens_greedy_untrained(input_text: torch.Tensor, max_new_tokens: int, device: str = 'cuda',
                                     dtype=torch.float16, use_cache: bool = True, model: torch.nn.Module = armataLM, tokenizer: gigatoken.Tokenizer = tokenizer) -> None:

    model.eval()
    model.reset_cache()
    model = model.to(device).to(dtype)


    idx = torch.tensor(tokenizer.encode(input_text), device=device).unsqueeze(0)


    current_input = idx

    with torch.no_grad():
        for _ in range(max_new_tokens):

            out = model.forward(current_input, use_cache=use_cache)


            last_token_out = out[:, -1, :]
            greedy_next_token = torch.argmax(last_token_out, dim=-1, keepdim=True)


            idx = torch.cat((idx, greedy_next_token), dim=-1)


            current_input = greedy_next_token

    return None


def run_speedtest(n_tokens_to_generate: int, model: torch.nn.Module, input_text: str, n_iter_warmups: int = 2, n_tokens_per_warmup_iter: int = 20):
    lg.logger.info(f'Starting speedtest. Generating {n_tokens_to_generate} new tokens.')


    for _ in range(n_iter_warmups):
        _ = generate_tokens_greedy_untrained(input_text=input_text, max_new_tokens=n_tokens_per_warmup_iter, model=model)

    start = time.time()
    _ = generate_tokens_greedy_untrained(input_text=input_text, max_new_tokens=n_tokens_to_generate, model=model)
    end = time.time() - start
    lg.logger.info(f"Tokens: {n_tokens_to_generate} - Time: {end} - Tokens/s: {n_tokens_to_generate / float(end)}")
    lg.logger.success("Tests ended.")

def measure_memory_usage() -> None:
    """
    Замеряет использование GPU-памяти до и после выполнения функции.

    Возвращает словарь:
        - 'allocated_before': память, выделенная до вызова (в МБ)
        - 'allocated_after': память после вызова (в МБ)
        - 'peak_allocated': пиковая память за время выполнения (в МБ)
        - 'delta': прирост памяти = allocated_after - allocated_before (в МБ)
        - 'peak_delta': пиковый прирост = peak_allocated - allocated_before (в МБ)
    """
    if not torch.cuda.is_available():
        print("CUDA недоступна, измерение памяти невозможно.")
        return None

    # Очищаем кэш и сбрасываем пиковые счётчики
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Замеряем память до
    allocated_before = torch.cuda.memory_allocated() / (1024 ** 2)  # в МБ


    # Замеряем память после
    allocated_after = torch.cuda.memory_allocated() / (1024 ** 2)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)

    lg.logger.info({
        'allocated_before': allocated_before,
        'allocated_after': allocated_after,
        'peak_allocated': peak_allocated,
        'delta': allocated_after - allocated_before,
        'peak_delta': peak_allocated - allocated_before,
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program for speed testing (Tokens/s) any implementation.')
    parser.add_argument('-n', help='Speedtest will generate n new tokens.', default=256, type=int)
    parser.add_argument('-warmup_iters', help='argument for test to warmup and get more accurate results', default=2, type=int)
    parser.add_argument('-n_tokens_per_warmup_iter', help='generated tokens per warmup iteration', default=20, type=int)
    parser.add_argument('-input_text', help='prefill text', default='Hi, my name is...', type=str)
    parser.add_argument('-model', help='choose model for a speed test\n-ArmataLM\n-GPT2-small\n-Deepseek-likeV2-Dense\n-qwen3_5-like-Dense\n Are available right now', default='ArmataLM', type=str)

    args = parser.parse_args()
    if args.model == 'ArmataLM':
        model = armataLM
    elif args.model == 'Deepseek-likeV2-Dense':
        model = Ds_like_model
    elif args.model == 'qwen3_5-like-Dense':
        model = qwen3_5_likemodel
    elif args.model == 'GPT2-small':
        model = gpt_model
    else:
        model = armataLM

    lg.logger.info(f'Model chosen: {model.__class__.__name__}')
    lg.logger.info(f'Parameters: {sum((x.numel() for x in model.parameters()))}')
    run_speedtest(n_tokens_to_generate=args.n, model=model, input_text=args.input_text, n_iter_warmups=args.warmup_iters, n_tokens_per_warmup_iter=args.n_tokens_per_warmup_iter)
    measure_memory_usage()


