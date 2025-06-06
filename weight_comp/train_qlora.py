import os, json, mmap, queue, threading, struct, argparse, gc, humanize, sys, types, contextlib
import numpy as np, torch, zstandard as zstd, psutil, time
import pandas as pd
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/Mistral-7B-v0.1")
parser.add_argument("--outdir", default="./zstd_comped_weights")
parser.add_argument("--autocast_context", action="store_true", help="Set torch.amp.autocast")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
print(f"{args.model=}, {args.outdir=}")

# Set GPU!
device = torch.device("cuda:0") 
torch.cuda.set_device(device) 

# --------------------------------------------------------------
#                1. Get Base model and LoRA Adapter 
# --------------------------------------------------------------
compute_dtype = getattr(torch, "bfloat16")
bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
    )
model = AutoModelForCausalLM.from_pretrained(
                args.model, 
                torch_dtype=torch.bfloat16, 
                quantization_config = bnb_config,
                device_map={"": 0})
model = prepare_model_for_kbit_training(model, 
            use_gradient_checkpointing=False)
peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.0,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules= ["gate_proj", "up_proj", "down_proj"]
    )
model = get_peft_model(model, peft_config, 
                autocast_adapter_dtype=False)   # set this to keep the adapters in bfloat16
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

# attach_live_mem_hooks(model)  

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

EVAL=True
if EVAL:
    model.eval()
else:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

def run_one_step():
    loss = model(**inputs).loss
    torch.cuda.synchronize()
    
bf16_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if args.autocast_context else contextlib.nullcontext()

def measure(n=5):
    for i in range(n):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peak0 = torch.cuda.max_memory_allocated(device)

        t0 = time.time()
        if EVAL:
            with torch.no_grad():
                # with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(**inputs).loss
        else:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
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