import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, mmap, queue, threading, struct, gc, humanize, sys, types
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
args = parser.parse_args()

# Set GPU!
device = torch.device("cuda:0") 
torch.cuda.set_device(device) 


# --------------------------------------------------------------
#                1. Get Base model and LoRA Adapter 
# --------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map={"": 0})
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
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
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

def get_model_weight_ptrs(model: torch.nn.Module):
    ptrs = set()
    for p in model.parameters():
        try:
            if (args.weight and p.requires_grad) or (not args.weight):
                ptrs.add(p.untyped_storage().data_ptr())
        except Exception:
            pass  # some fake tensors like QLoRA quantizers may not support this
    return ptrs

model_weight_ptrs = get_model_weight_ptrs(model)

def is_lora_weight(tensor: torch.Tensor):
    try:
        return tensor.is_leaf and tensor.untyped_storage().data_ptr() in model_weight_ptrs
    except Exception:
        return False

# ------------------------------------------------------------------
# 1. 先建一份 “data_ptr → 参数名” 索引，方便反查是谁
# ------------------------------------------------------------------
def build_ptr2name(model):
    ptr2name = {}
    for n, p in model.named_parameters():
        ptr2name[p.data_ptr()] = n
    return ptr2name

def make_hooks(ptr2name):
    def pack_hook(t):
        global seq_id, pack_log

        if is_lora_weight(t):   # 模型权重 tensor
            return t
        
        pack_log.append(seq_id)
        seq_id += 1

        return t, seq_id - 1                 # 不做搬运，直接返回原 tensor

    def unpack_hook(t):
        if isinstance(t, torch.Tensor):
            return t
        
        global unpack_log
        ts, seq_id = t
        
        unpack_log.append(seq_id)
        
        return ts                 # 必须对称返回
    return pack_hook, unpack_hook


pack_log = []     # 记录 pack 顺序
unpack_log = []   # 记录 unpack 顺序
seq_id = 0

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

ptr2name = build_ptr2name(model)
pack_hook, unpack_hook = make_hooks(ptr2name)

if 1:
    for i in range(1):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peak0 = torch.cuda.max_memory_allocated(device)
        
        process = psutil.Process(os.getpid())
        cpu_mem_peak0 = process.memory_info().rss / 1024 / 1024

        t0 = time.time()
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            loss = model(**inputs).loss
        print(f"{loss=}")

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
        
        pack_ids = pack_log
        unpack_ids = unpack_log

        print(f"Total packed: {len(pack_ids)}, unpacked: {len(unpack_ids)}")

        # 方法一：是否严格逆序
        is_strict_reverse = (unpack_ids == pack_ids[::-1])
        print(f"Is unpack reverse of pack? {is_strict_reverse}")

        # 方法二：统计最大“位置偏差”
        max_diff = 0
        for i, uid in enumerate(unpack_ids):
            if uid in pack_ids:
                orig_idx = pack_ids[::-1].index(uid)
                diff = abs(orig_idx - i)
                max_diff = max(max_diff, diff)

        print(f"Max position diff between pack/unpack: {max_diff}")
