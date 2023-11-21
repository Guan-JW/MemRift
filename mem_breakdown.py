import os, json, mmap, queue, threading, struct, argparse, gc, humanize, sys, types
import numpy as np, torch, zstandard as zstd, psutil, time
import pandas as pd
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from collections import defaultdict
import bitsandbytes as bnb
import contextlib

def bytes_of(*tensors):
    "返回多张量总字节数"
    return sum(t.element_size() * t.numel() for t in tensors)

def pretty(num_bytes):
    "GiB 保留两位小数"
    return f"{num_bytes/1024**3:.2f} GB"

@contextlib.contextmanager
def fresh_peak():
    "上下文内测一次峰值；退出时返回字节数"
    torch.cuda.reset_peak_memory_stats()
    yield
    peak = torch.cuda.max_memory_allocated()
    fresh_peak.peak = peak

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/Mistral-7B-v0.1")
parser.add_argument("--outdir", default="./zstd_comped_weights")
parser.add_argument("--check_diff", action="store_true",
                    help="同时加载原始模型并比对差值（耗显存）")
parser.add_argument(
        "--max_length", type=int, default=512, help="Input length"
    )
parser.add_argument(
        "--batch_size", type=int, default=1, help="Input batch size"
    )
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
print(f"{args.model=}, {args.outdir=}")

# Set GPU!
device = torch.device("cuda:0") 
torch.cuda.set_device(device) 

# --------------------------------------------------------------
#                1. Get Base model and LoRA Adapter 
# --------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": 0})
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

dataset = load_dataset("timdettmers/openassistant-guanaco")
df = pd.DataFrame(dataset['train'])
#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
tokenizer.padding_side = 'left'


# sample one input
sample_text = [dataset['train'][i]['text'] for i in range(args.batch_size)]
encoding = tokenizer(
    sample_text,
    max_length=args.max_length,
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

gradient_checkpointing = False
if gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False  # for mistral or LLaMA, 在大多数 HuggingFace 的 decoder-only 模型（如 Mistral、LLaMA）中，use_cache=True 会导致模型跳过中间状态保存，从而禁用 gradient checkpointing。

# ------------------------------------------------------------------
# 1. 先建一份 “data_ptr → 参数名” 索引，方便反查是谁
# ------------------------------------------------------------------
def build_ptr2name(model):
    ptr2name = {}
    for n, p in model.named_parameters():
        ptr2name[p.data_ptr()] = n
    return ptr2name

# ------------------------------------------------------------------
# 2. 定义 pack / unpack hook
#    - pack_hook 会在 ctx.save_for_backward(tensor) 时被调用
#    - unpack_hook 会在 backward 取回 tensor 时被调用
# ------------------------------------------------------------------
def make_hooks(ptr2name, log=defaultdict(int)):
    def pack_hook(t):
        name = ptr2name.get(t.data_ptr(), "<non-param>")
        log[name] += 1
        return t                 # 不做搬运，直接返回原 tensor
    def unpack_hook(t):
        return t                 # 必须对称返回
    return pack_hook, unpack_hook, log


model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# ---------------- baseline：权重 + optim state ----------------
param_bytes = bytes_of(*[p.data for p in model.parameters()])
opt_tensors = [
    # 把 state 里所有张量都抓出来，包含 exp_avg、exp_avg_sq、step…
    v
    for state in optimizer.state.values()
    for v in state.values()
    if torch.is_tensor(v)
]
opt_bytes = bytes_of(*opt_tensors)
# opt_bytes   = bytes_of(*optimizer.state[p]['exp_avg'] for p in optimizer.state if 'exp_avg' in optimizer.state[p])

baseline = param_bytes + opt_bytes
print("Baseline  (P+Opt):", pretty(baseline))

ptr2name = build_ptr2name(model)
pack, unpack, counter = make_hooks(ptr2name)
def measure(n=1):
    for i in range(n):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peak0 = torch.cuda.max_memory_allocated(device)
        
        process = psutil.Process(os.getpid())
        cpu_mem_peak0 = process.memory_info().rss / 1024 / 1024

        t0 = time.time()
        with fresh_peak():                    # ⬅️ 记录前向期间峰值
            loss = model(**inputs).loss
        act_peak = fresh_peak.peak
        act_bytes = act_peak - baseline       # 可能含少量 kernel 缓冲，但偏差 <5%
        print("Activations @Fwd:", pretty(act_bytes))
        print(f"{loss=}")

        # ---------------- backward：测梯度 ----------------
        with fresh_peak():           
            loss.backward()
        bwd_peak = fresh_peak.peak
        grad_bytes = bytes_of(*[p.grad for p in model.parameters() if p.grad is not None])
        print("Gradients size:", pretty(grad_bytes))

        # 反传峰值校验（含 P+Opt+Act+Grad）
        assert abs(bwd_peak - (baseline + act_bytes + grad_bytes)) / bwd_peak < 0.05

        optimizer.step()
        # optimizer.zero_grad()
        opt_tensors = [
            # 把 state 里所有张量都抓出来，包含 exp_avg、exp_avg_sq、step…
            v
            for state in optimizer.state.values()
            for v in state.values()
            if torch.is_tensor(v)
        ]
        opt_bytes = bytes_of(*opt_tensors)

        t = time.time() - t0
        peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        print(f"Round {i}: {t*1000:.1f} ms , peak ={peak/1024/1024:.1f} MB")

        process = psutil.Process(os.getpid())
        cpu_mem_peak = process.memory_info().rss / 1024 / 1024  # 单位: MB
        print(f"Peak CPU memory usage: {cpu_mem_peak:.2f} MB")
        
        # ---------------- 结果百分比 ----------------
        total_iter_peak = max(act_peak, bwd_peak)      # 通常是 bwd_peak
        parts = {
            "Parameters" : param_bytes,
            "Opt-state"  : opt_bytes,
            "Activations": act_bytes,
            "Gradients"  : grad_bytes,
        }
        print("\n=== Memory breakdown ===")
        for k,v in parts.items():
            print(f"{k:<12}: {pretty(v):>9}  ({v/total_iter_peak:5.1%})")
        print(f"Iteration peak: {pretty(total_iter_peak)}")


measure()
