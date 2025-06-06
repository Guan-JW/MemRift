import os, json, mmap, queue, threading, struct, argparse, gc, humanize, sys, types
import numpy as np, torch, zstandard as zstd, psutil, time
import pandas as pd
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from collections import defaultdict
import bitsandbytes as bnb

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

def attach_live_mem_hooks(model):
    def _fw_hook(mod, *_):
        torch.cuda.synchronize()
        print(f"[FW] {mod.__class__.__name__:30s} "
              f"=> {torch.cuda.memory_allocated()/1024**2:8.1f} MB")
    def _bw_hook(mod, *_):
        torch.cuda.synchronize()
        print(f"[BW] {mod.__class__.__name__:30s} "
              f"=> {torch.cuda.memory_allocated()/1024**2:8.1f} MB")
    for m in model.modules():
        m.register_forward_hook(_fw_hook,      prepend=False)
        m.register_full_backward_hook(_bw_hook,prepend=False)

attach_live_mem_hooks(model)     


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

def run_one_step():
    loss = model(**inputs).loss
    torch.cuda.synchronize()

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


EVAL=False
if EVAL:
    model.eval()
else:
    model.train()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    optimizer = bnb.optim.Adam8bit(                    # or AdamW8bit
        model.parameters(),
        lr=2e-4,
        # betas=(0.9, 0.999),
        # eps=1e-8,
        # weight_decay=0.01,
        # # --- 8-bit specific knobs --------------------
        # min_8bit_size=4096,        # tensors <4 k params stay in FP32 (safer)
        # percentile_clipping=0,     # 0 = adaptive; 99/99.9 = hard clip
        # block_wise=True            # keeps statistics per 2048-elem chunk
    )

ptr2name = build_ptr2name(model)
pack, unpack, counter = make_hooks(ptr2name)
def measure(n=10):
    for i in range(n):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peak0 = torch.cuda.max_memory_allocated(device)
        
        process = psutil.Process(os.getpid())
        cpu_mem_peak0 = process.memory_info().rss / 1024 / 1024

        t0 = time.time()
        if EVAL:
            with torch.no_grad():
                with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                    loss = model(**inputs).loss
        else:
            loss = model(**inputs).loss
        print(f"{loss=}")

        if not EVAL:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        t = time.time() - t0
        peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

        print(f"Round {i}: {t*1000:.1f} ms , peak ={peak/1024/1024:.1f} MB")

        process = psutil.Process(os.getpid())
        cpu_mem_peak = process.memory_info().rss / 1024 / 1024  # 单位: MB
        print(f"Peak CPU memory usage: {cpu_mem_peak:.2f} MB")
        
        if EVAL:
            break


measure()

# ------------------------------------------------------------------
# 5. 统计结果
# ------------------------------------------------------------------
print("Saved into ctx (tensor → times saved):")
for name, n in sorted(counter.items(), key=lambda x: -x[1]):
    print(f"{name:60s} : {n}")