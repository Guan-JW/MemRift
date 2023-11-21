import os, json, mmap, queue, threading, struct, argparse, gc, humanize, sys, types, contextlib
import numpy as np, torch, zstandard as zstd, psutil, time
import pandas as pd
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from collections import defaultdict
import bitsandbytes as bnb
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/Mistral-7B-v0.1")
parser.add_argument("--finetune_type", choices=["full", "lora", "qlora"], default="lora", help="Type of finetuning")
parser.add_argument("--autocast_context", action="store_true", help="Set torch.amp.autocast")
parser.add_argument("--outdir", default="./zstd_comped_weights")
parser.add_argument("--check_diff", action="store_true",
                    help="同时加载原始模型并比对差值（耗显存）")
parser.add_argument(
        "--max_length", type=int, default=512, help="Input length"
    )
parser.add_argument(
        "--batch_size", type=int, default=1, help="Input batch size"
    )
parser.add_argument(
        "--round", type=int, default=5, help="# training cycles"
    )
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
print(f"{args=}")

# Set GPU!
device = torch.device("cuda:0") 
torch.cuda.set_device(device) 

# --------------------------------------------------------------
#                1. Get Base model and LoRA Adapter 
# --------------------------------------------------------------
MODEL_TYPE = torch.bfloat16
if args.finetune_type == "qlora":
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=MODEL_TYPE,
            bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
            args.model, 
            torch_dtype=MODEL_TYPE, # must use this to specify bfloat16 data type !!!
            quantization_config = bnb_config,
            device_map={"": 0}
    )
    model = prepare_model_for_kbit_training(model, 
            use_gradient_checkpointing=False)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, 
                    torch_dtype=MODEL_TYPE, device_map={"": 0})


if args.finetune_type in ["lora", "qlora"]:
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
        torch.cuda.reset_peak_memory_stats()
        print(f"[FW] {mod.__class__.__name__:30s} "
              f"=> {torch.cuda.max_memory_allocated()/1024**2:8.1f} MB")
    def _bw_hook(mod, *_):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        print(f"[BW] {mod.__class__.__name__:30s} "
              f"=> {torch.cuda.max_memory_allocated()/1024**2:8.1f} MB")
    for m in model.modules():
        m.register_forward_hook(_fw_hook,      prepend=False)
        # m.register_full_backward_hook(_bw_hook,prepend=False)

# attach_live_mem_hooks(model)     


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
# tokenizer.pad_token = tokenizer.unk_token
# tokenizer.pad_token_id =  tokenizer.unk_token_id
tokenizer.padding_side = 'left'


# sample one input
# def collate(batch):
#     enc = tokenizer(batch,              # ← 直接喂一批原始文本
#                     max_length=args.max_length,
#                     padding='max_length',
#                     truncation=True,
#                     return_tensors='pt')
#     enc = {k: v.to(model.device) for k, v in enc.items()}
#     enc["labels"] = enc["input_ids"].clone()
#     return enc
# dl = DataLoader(dataset["train"]["text"],
#                 batch_size=args.batch_size,
#                 shuffle=True,
#                 collate_fn=collate,
#                 drop_last=True)

# sample_text = dataset['train'][0]['text']      # 也可以随机选一条
sample_text = [dataset['train'][i]['text'] for i in range(args.batch_size)]
encoding = tokenizer(
    sample_text,
    # max_length=512,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    # optimizer = bnb.optim.Adam8bit(                    # or AdamW8bit
    #     model.parameters(),
    #     lr=2e-4,
    #     # betas=(0.9, 0.999),
    #     # eps=1e-8,
    #     # weight_decay=0.01,
    #     # # --- 8-bit specific knobs --------------------
    #     # min_8bit_size=4096,        # tensors <4 k params stay in FP32 (safer)
    #     # percentile_clipping=0,     # 0 = adaptive; 99/99.9 = hard clip
    #     # block_wise=True            # keeps statistics per 2048-elem chunk
    # )

ptr2name = build_ptr2name(model)
pack, unpack, counter = make_hooks(ptr2name)
bf16_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if args.autocast_context else contextlib.nullcontext()

def measure(n=args.round):
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
            with bf16_ctx:
                loss = model(**inputs).loss
        print(f"{loss=}")
        peak1 = torch.cuda.max_memory_allocated(device)
        print(f"Round {i}: peak1={(peak1)/1024/1024:.1f} MB")
        cpu_mem_peak = process.memory_info().rss / 1024 / 1024  # 单位: MB
        print(f"Peak CPU memory usage: {cpu_mem_peak:.2f} MB")

        if not EVAL:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.reset_peak_memory_stats(device)
        t = time.time() - t0
        peak = torch.cuda.max_memory_allocated(device)

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