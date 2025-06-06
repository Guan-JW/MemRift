import os, json, mmap, queue, threading, struct, argparse, gc, humanize, sys, types
import numpy as np, torch, zstandard as zstd, psutil, time
import pandas as pd
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/Mistral-7B-v0.1")
parser.add_argument("--outdir", default="./zstd_comped_weights")
parser.add_argument("--check_diff", action="store_true",
                    help="同时加载原始模型并比对差值（耗显存）")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
print(f"{args.model=}, {args.outdir=}")

# Set GPU!
device = torch.device("cuda:0") 
torch.cuda.set_device(device) 

# --------------------------------------------------------------
#                1. Get Base model and LoRA Adapter 
# --------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": "cpu"})
peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.0,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules= ["gate_proj", "up_proj", "down_proj"]
    )
model = get_peft_model(model, peft_config, 
                autocast_adapter_dtype=True)   # set this to keep the adapters in bfloat16
def keep_embedding_and_lm_head_on_gpu(model, gpu_device):
    """
    - 找到名字里含 'embed_tokens' 的 nn.Embedding   → 常驻 GPU
    - 找到 nn.Linear 且 out_features == vocab_size  → 当作 lm_head
    """
    modules_to_pin = []

    for name, module in model.named_modules():
        # --- 1) token embedding ---
        if isinstance(module, torch.nn.Embedding) and 'embed_tokens' in name:
            modules_to_pin.append(module)

        # --- 2) lm_head ---
        elif isinstance(module, torch.nn.Linear):
            # heuristically: same dim as tokenizer vocab
            if module.out_features == model.config.vocab_size:
                modules_to_pin.append(module)
        
        if "rotary_emb" in name:          # Mistral/LLama
            for n, buf in module._buffers.items():
                module._buffers[n] = buf.to(device)

    if not modules_to_pin:
        raise RuntimeError("didn't find embedding / lm_head automatically")

    # 真正搬运权重（Parameter.data 或 Tensor）
    for mod in modules_to_pin:
        for attr in ("weight", "bias"):
            p = getattr(mod, attr, None)
            if torch.is_tensor(p) and p.device != gpu_device:
                if isinstance(p, torch.nn.Parameter):
                    p.data = p.data.to(gpu_device, non_blocking=True)
                else:
                    setattr(mod, attr, p.to(gpu_device, non_blocking=True))
        mod.to(gpu_device)   # 把内部 buffer 也搬上来（e.g. Embedding.padding_idx）

    names = [n for n, m in model.named_modules() if m in modules_to_pin]
    print("Pinned on GPU:", names)

keep_embedding_and_lm_head_on_gpu(model, device)

# --------------- 2. 逐层 offload / prefetch -----------------
def attach_offload_hooks(model, gpu_device):
    """
    forward_pre_hook:  把本层权重搬到 GPU
    forward_hook    :  forward 结束 → 把权重丢回 CPU
                       (训练时应改到 full_backward_hook)
    """
    for mod in model.modules():
        # 只处理有权重的模块
        if not any(hasattr(mod, a) for a in ("weight", "bias")): 
            continue

        # ---------- 2-1 pre-hook：H2D copy ----------
        def _to_gpu(m, *_):
            for attr in ("weight", "bias"):
                p = getattr(m, attr, None)
                if p is None or p.device == gpu_device:
                    continue
                if isinstance(p, torch.nn.Parameter):
                    p.data = p.data.to(gpu_device, non_blocking=True)
                else:  # 普通 tensor
                    setattr(m, attr, p.to(gpu_device, non_blocking=True))
            torch.cuda.synchronize()
            print(f"[FW] {m.__class__.__name__:30s} "
              f"=> {torch.cuda.memory_allocated()/1024**2:8.1f} MB")

        # ---------- 2-2 post-hook：GPU → 释放 ----------
        # 推理：forward 结束就丢   训练：用 full_backward_hook
        def _release(m, *_):
            for attr in ("weight", "bias"):
                p = getattr(m, attr, None)
                if p is None or p.device != gpu_device:
                    continue
                if isinstance(p, torch.nn.Parameter):
                    # ① 只搬 data，不换对象
                    p.data = p.data.cpu()              # or .pin_memory()
                else:
                    # m 上原本不是 Parameter（极少见的 buffer 情况）
                    setattr(m, attr, p.cpu())
                torch.cuda.empty_cache()    # 立即归还缓存池

        mod.register_forward_pre_hook(_to_gpu,  prepend=True)
        mod.register_forward_hook(_release,     prepend=False)

attach_offload_hooks(model, device)

# --------------- 3. 准备一条样本 ----------------

dataset = load_dataset("/opt/models/dataset/openassistant-guanaco")
df = pd.DataFrame(dataset['train'])
#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id =  tokenizer.unk_token_id
tokenizer.padding_side = 'left'


# sample one input
sample_text = dataset['train'][0]['text']      # 也可以随机选一条
encoding = tokenizer(
    sample_text,
    max_length=512,
    padding='max_length',          # 填充到 512
    truncation=True,
    return_tensors='pt'
)
# 把 tensors 挪到模型所在设备
input_ids = encoding['input_ids'].to(model.device)
attention_mask = encoding['attention_mask'].to(model.device)

# 标签通常直接等于 input_ids（自回归语言模型）
inputs = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'labels': input_ids.clone()
}

# --------------- 4. 跑一次推理并记录峰值 ---------------

EVAL=True
if EVAL:
    model.eval()
else:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

def measure(n=5):
    for i in range(n):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peak0 = torch.cuda.max_memory_allocated(device)
        
        process = psutil.Process(os.getpid())
        cpu_mem_peak0 = process.memory_info().rss / 1024 / 1024

        t0 = time.time()
        if EVAL:
            with torch.no_grad():
                loss = model(**inputs).loss
        else:
            loss = model(**inputs).loss
        print(f"{loss=}")
        t = (time.time() - t0) / n

        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        print(f"Round {i}: {t*1000:.1f} ms , peak ={peak/1024/1024:.1f} MB")

        process = psutil.Process(os.getpid())
        cpu_mem_peak = process.memory_info().rss / 1024 / 1024  # 单位: MB
        print(f"Peak CPU memory usage: {cpu_mem_peak:.2f} MB")
        break


measure()