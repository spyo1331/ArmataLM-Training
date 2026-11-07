import wandb
import torch
from llm_architectures import gpt_model, Ds_like_model, max_context, qwen3_5_likemodel, exp_model
import gigatoken as gt
import loguru as lg
import time
from typing import Optional
from pydantic import BaseModel
tokenizer = gt.Tokenizer('gpt2').as_hf()
import gc



class TestingConfig(BaseModel):
    architecture: str
    use_cache: bool





def generate_text_with_cache(model, idx, max_new_tokens,
                                context_size=None, use_cache=True, device='cuda'):

    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    idx = torch.tensor(idx).unsqueeze(0).to(device)


    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):

                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)

                idx = torch.cat([idx, next_idx], dim=1)

                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx



def run_speedtest(model,n_tokens_testcases: list[int], config_dict: TestingConfig, run_name: Optional[str]) -> None:
    lg.logger.info('Starting speed time test.')
    lg.logger.info(f"Model parameters: {sum(x.numel() for x in model.parameters())}")
    model.to('cuda').to(torch.bfloat16)
    for i in n_tokens_testcases:
        model.reset_cache()
        start = time.time()
        generate_text_with_cache(model=model, idx=tokenizer.encode('Привет я максим и я'), context_size=config_dict['max_n_ctx'], max_new_tokens=i, use_cache=config_dict['use_cache'], device=config_dict['device'])
        end = time.time() - start

        lg.logger.info(f"Tokens: {i} - Time: {end} - Tokens/s: {i/float(end)}")


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

test_cases = [256,1024, 2048, 3800, 16384, 32768]

if __name__ == "__main__":
    speed_ds_like = run_speedtest(exp_model,test_cases, {'use_cache': True, 'architeture': "DS2023_like", "device": 'cuda', "max_n_ctx": max_context}, run_name='cached_run')
    print(measure_memory_usage())
