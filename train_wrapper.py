import os, psutil, contextlib, weakref
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, mmap, queue, threading, struct, argparse, gc, humanize, sys, types, re, pprint, json, mmap
import numpy as np, torch, time, imagecodecs, concurrent.futures as fut
from collections import deque, defaultdict
import pandas as pd
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from torch.utils._pytree import tree_map 
from torch.utils.cpp_extension import load
import bitsandbytes as bnb

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import MistralModel, MistralForCausalLM, MistralConfig, LlamaModel, LlamaForCausalLM, LlamaConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from accelerate import init_empty_weights
import float_split_stride_pin as fs_sp

import zstandard as zstd
import lz4.frame as lz4f, lz4.block as lz4b
import io
from dataclasses import dataclass
import wandb, subprocess, re
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import torch.cuda.nvtx as nvtx
from typing import Union, List, Optional

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/TinyLlama-1.1B-Chat-v1.0")
parser.add_argument("--outdir", default="./weight_comp/prepare_weight/zstd_comped_weights_level21")
parser.add_argument("--finetune_type", choices=["full", "lora", "qlora"], default="lora", help="Type of finetuning")
parser.add_argument("--autocast_context", action="store_true", help="Set torch.amp.autocast")
parser.add_argument("--check_diff", action="store_true",
                    help="同时加载原始模型并比对差值（耗显存）")
parser.add_argument("--hook", action="store_true", help="Run with compression hooks")
parser.add_argument("--debug", action="store_true", help="Run with debug")
parser.add_argument("--print_ratio", action="store_true", help="Print activation's compression ratio")
parser.add_argument("--print_time", action="store_true", help="Print activation's compression ratio")
parser.add_argument("--weight", default=False, action="store_true", help="Switch on weight compression")
parser.add_argument("--activation", default=False, action="store_true", help="Switch on activation compression")
parser.add_argument(
        "--level", type=int, default=1, help="Zstd compression level (<22)"
    )
parser.add_argument(
        "--round", type=int, default=5, help="# training cycles"
    )
parser.add_argument(
        "--max_length", type=int, default=2048, help="Input length"
    )
parser.add_argument(
        "--batch_size", type=int, default=1, help="Input batch size"
    )
args = parser.parse_args()


nvmlInit(); gpu = nvmlDeviceGetHandleByIndex(0)
proj = ""
if args.hook:
    if not args.weight:
        proj = "wrapper_memrift_jetson"
    else:
        proj = "memrift_act_weight_jetson"
else:
    if args.finetune_type == "lora":
        proj = "vanilla_jetson"
    elif args.finetune_type == "qlora":
        if args.autocast_context:
            proj = "qlora_amp_jetson"
        else:
            proj = "qlora_jetson"
if args.max_length != 2048:
    proj = f"{proj}_{args.max_length}"
if args.batch_size > 1:
    proj = f"{proj}_bs{args.batch_size}"

if "Mistral-7B" in args.model:
    proj = f"Mistral-7B_{proj}"
elif "Llama-3.1-8B" in args.model:
    proj = f"Llama-3.1-8B_{proj}"

wandb.init(project=proj, config=vars(args))  # 把 CLI 参数也存进去
wandb.define_metric("timestamp")          # 用绝对时间
wandb.define_metric("*", step_metric="timestamp")

T0 = time.time()
# ──────────────────────────────────────────
# 2.  采样线程：tegrastats + torch
PAT = re.compile(
    r'RAM\s+(\d+)/(\d+)MB'           # RAM used / total
    r'.*?CPU\s+\[([^\]]+)\]'         # 整个  [...]  片段
    r'.*?GR3D_FREQ\s+(\d+)%',        # GPU util %
    re.I)


def tegra_loop():
    cmd = ['/usr/bin/tegrastats', '--interval', '500']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            text=True, bufsize=1)

    for line in proc.stdout:
        m = PAT.search(line)
        # print(f"{m=}")
        if not m:
            continue                          # 依旧安全兜底

        ram_u, ram_t, cpu_blk, gpu_u = m.groups()

        # ① CPU 方括号里提取所有 “99%@freq”
        cpu_vals = [int(x) for x in re.findall(r'(\d+)%@', cpu_blk)]
        cpu_avg  = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0

        wandb.log({
            "timestamp": int((time.time()-T0)*1000),   # ms step
            "ram_used_MB": int(ram_u),
            "ram_total_MB": int(ram_t),
            "cpu_util":  cpu_avg,                      # 平均 %
            "gpu_util":  int(gpu_u),
            "gpu_alloc_MB": torch.cuda.memory_allocated() // 2**20,
            "gpu_reserved_MB": torch.cuda.memory_reserved() // 2**20 or -1,
            "cpu_proc_MB": psutil.Process().memory_info().rss // 2**20,
        }, commit=True)
    proc.stdout.close()


threading.Thread(target=tegra_loop, daemon=True).start()

os.makedirs(args.outdir, exist_ok=True)
print(f"\n\n{args.model=}, {args.outdir=}, {args.weight=}, {args.activation=}, {args.max_length=}")
rid = 0
orig_len = new_len = 0

# Set GPU!
device = torch.device("cuda:0") 
torch.cuda.set_device(device) 

# --------------------------------------------------------------
#                1. Get Base model and LoRA Adapter 
# --------------------------------------------------------------

wait_comp_done = wait_comp_start = comp_done_time = decomp_done_time = 0
jxl_decode_done = 0
pack_time = unpack_time = 0
pp1_time = pp2_time = pp3_time = pp4_time = 0
wait_time = decode_time = copy_time = merge_time = 0
prep_time1 = prep_time2 = prep_time3 = prep_time4 = prep_time5 = 0
post_time1 = post_time2 = post_time3 = post_time4 = post_time5 = 0
rid = 0

MODEL_TYPE = torch.bfloat16
gradient_checkpointing = False
_PTR2CP = weakref.WeakValueDictionary()

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
            use_gradient_checkpointing=gradient_checkpointing)
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


gc.collect()
process = psutil.Process(os.getpid())
cpu_mem_peak0 = process.memory_info().rss / 1024 / 1024  # 单位: MB
print(f"Peak CPU memory usage: {cpu_mem_peak0:.2f} MB")
print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated(device) / 1024 / 1024:.2f} MB")


_tls = threading.local()
decomp_lock = threading.Lock()
decomp_time = comp_time = 0

# -------------------------------------------------------------------------
#                           Activation compression
# -------------------------------------------------------------------------

@dataclass
class TensorLayout:
    shape: tuple
    stride: tuple
    offset: int
    dtype: torch.dtype

    @staticmethod
    def from_tensor(t: torch.Tensor):
        return TensorLayout(
            shape=tuple(t.shape),
            stride=tuple(t.stride()),
            offset=t.storage_offset(),
            dtype=t.dtype
        )

    def __repr__(self):
        return (f"TensorLayout(shape={self.shape}, stride={self.stride}, "
                f"offset={self.offset}, dtype={self.dtype})")

class Token:
    def __init__(self, cpu_exp_buf, sm_bits, evt, rid):
        # self.future = future            # 便于重复获取压缩数据
        self.sm_bits = sm_bits          # 用于恢复数据
        self.cpu_exp_buf = cpu_exp_buf  # 在数据拷贝和压缩期间存放 cpu（pin_memory） 上的 exp_bits，防治拷贝出错
        self.refcount = 1           # 计数器。todo: no need for an explicit refcount?
        self.DtoC_copy_evt = evt

        self.scheduled = False          # ➜ 已提交 decode 任务？
        self.cpu_exp = None # todo : for decompressed data   
        self.st_ref = None  # todo

        self.ready_evt     = threading.Event()   # CPU+H2D 全完成 ⇒ set()

    def inc_refcount(self):
        self.refcount += 1
    def dec_refcount(self):
        self.refcount -= 1
        if self.refcount == 0:
            return True # to delete element from tensor_cache
        return False

    def check_invalid_refcount(self):
        return self.refcount <= 0

    def free_future(self):
        self.future = None

    def free_cpu_exp_buf(self):
        self.cpu_exp_buf = None
        self.DtoC_copy_evt = None
        
    def release_payload(self):
        self.cpu_exp = None
        self.sm_bits = None
        self.rst = None
        self.CtoD_copy_evt = None
    
def get_model_weight_ptrs(model: torch.nn.Module):
    ptrs = set()
    for name, p in model.named_parameters():
        # if not p.requires_grad:
        #     print(f"[FROZEN] {name} shape={p.shape} requires_grad={p.requires_grad} is_leaf={p.is_leaf}, {p.data_ptr()=}")
        # elif "lora_" in name:
        #     print(f"[LoRA]   {name} shape={p.shape} requires_grad={p.requires_grad} is_leaf={p.is_leaf}")
        # else:
        #     print(f"[TRAIN]  {name} shape={p.shape} requires_grad={p.requires_grad} is_leaf={p.is_leaf}")

        try:
            if (args.weight and p.requires_grad) or (not args.weight):
                ptrs.add(p.untyped_storage().data_ptr())
        except Exception:
            pass  # some fake tensors like QLoRA quantizers may not support this
    return ptrs

model_weight_ptrs = get_model_weight_ptrs(model)

def is_lora_weight(tensor: torch.Tensor):
    try:
        return tensor.untyped_storage().data_ptr() in model_weight_ptrs
    except Exception:
        return False


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


@dataclass
class PlaceHolderToken:
    t_ref: torch.Tensor          # 指向原 GPU 张量
    dtype: torch.dtype
    shape: torch.Size
    stride: torch.Size
    offset: int
    # ↓ 运行时填充：
    # cpu_buf: torch.Tensor | None = None
    future: fut.Future | None = None

class AsyncCompressor:
    """Minimal async compressor skeleton.

    Replace `compress_async` / `decompress_sync` with your JPEG‑XL + zstd
    pipeline (or the fused CUDA kernels you already built).  The goal is:

    • *Forward*: kick off an asynchronous host‑side compression of the GPU
      activation on a separate CUDA stream *or* in a CPU thread and keep an
      opaque *token* with all metadata.

    • *Backward*: block until the compression job finishes, then decode the
      token back into a torch.Tensor on the original device.  The tensor is
      fed directly into autograd; no extra copies are needed.
    """

    def __init__(self, stream: torch.cuda.Stream | None = None, pool_workers=4):
        # Dedicated stream for D→H copy + encode so we don't block the main
        # compute stream.  Feel free to expose this in your API.
        print("init AsyncCompressor")
        self.stream = stream or torch.cuda.Stream()
        self.pool = fut.ThreadPoolExecutor(pool_workers)
        self.cctx = zstd.ZstdCompressor(level=args.level, threads=-1, write_checksum=False)
        self.dctx = zstd.ZstdDecompressor()
        self.counter = {}
        self.ptr_t = {}

    # ---------------------------------------------------------------------
    #  Interfaces you need to flesh out
    # ---------------------------------------------------------------------
    def kickoff(self, tok: PlaceHolderToken, t: torch.Tensor):


        self.stream.wait_stream(torch.cuda.current_stream())
        # stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.stream):
            cpu_exp, sm_bits = fs_sp.split(t, self.stream.cuda_stream)
            evt = self.stream.record_event()   # 拷贝结束事件
        evt.synchronize()
        # tok.cpu_exp = cpu_exp
        tok.sm_bits = sm_bits
        # tok.t_ref = t.clone()
        del t

        arr = cpu_exp.numpy()
        comped_bytes = self.cctx.compress(arr)
        tok.comped_cpu_exp = comped_bytes
        tok.numel = arr.size

        # def _encode_and_free(exp: torch.Tensor, sm: torch.Tensor):
        #     # comp = jpegxl_zstd_encode(exp, sm)   # 这里内部最好直接用 numpy buffer，别再复制
        #     # del exp, sm                          # 🔑 立刻释放线程私有引用
        #     return exp, sm

        # 立即腾显存
        # t.data = torch.empty(0, device=t.device, dtype=t.dtype)
        # torch.cuda.empty_cache()

        # tok.cpu_exp = torch.zeros(tok.shape, dtype=torch.uint8, device='cpu', pin_memory=True)
        # if tok.dtype == torch.bfloat16:
        #     tok.sm_bits = torch.zeros(tok.shape, dtype=torch.uint8, device=device)
        # else:
        #     tok.sm_bits = torch.zeros(tok.cpu_exp.numel() * 3, dtype=torch.uint8, device=device)
        
        # tok.future = self.pool.submit(_encode_and_free, cpu_exp, sm_bits)
        # del cpu_exp, sm_bits
        # key = (t.data_ptr(), t.nbytes)
        # key = t.untyped_storage().data_ptr()
        # self.counter[key] = self.counter.get(key, 0) + 1
        
        # equal = 0
        # for tensor in self.counter.keys():
        #     if torch.equal(tensor, t):
        #         equal += 1
        #         self.counter[tensor] += 1
        #         print(f"{tok.t_ref.shape=}, {tok.t_ref.data_ptr()=}, {tensor.data_ptr()=}, {self.counter[tensor]=}")
        #         break
        # if equal == 0:
        #     self.counter[t] = 1

        return 

    def decompress_sync(self, 
            tok: PlaceHolderToken,
            device: torch.device | int):
        # out = tok.t_ref

        # # stream = self.stream
        # # cpu_exp, sm_bits = tok.future.result()  # 阻塞，等待压缩完成
        # cpu_exp = tok.cpu_exp
        comped_bytes = tok.comped_cpu_exp
        cpu_exp = torch.empty(tok.numel, dtype=torch.uint8, pin_memory=True)
        with self.dctx.stream_reader(memoryview(comped_bytes)) as reader:
            view = memoryview(cpu_exp.numpy())   # numpy() 不复制，只拿 data_ptr
            nread = reader.readinto(view)
            assert nread == tok.numel, "decompress size mismatch"

        sm_bits = tok.sm_bits
        stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            rst = fs_sp.merge(cpu_exp, sm_bits, tok.shape, tok.stride, tok.offset, tok.dtype, stream.cuda_stream)
        evt = stream.record_event()
        evt.synchronize()
        
        # out.data = rst
        # del tok.cpu_exp, tok.sm_bits
        
        # out.data = tok.cpu_buf.to(device, non_blocking=False).view(tok.shape).to(tok.dtype)
        return rst
        # return tok.t_ref

    def reset(self):
        self.counter.clear()

counter = {}
class DecoderLayerWrapper(torch.nn.Module):
    def __init__(self, layer: nn.Module, compressor: AsyncCompressor):
        super().__init__()
        self.layer = layer
        self.comp = compressor          # 你的 AsyncCompressor

    # ---------- forward ----------
    def forward(self, *inp, **kw):
        tokens: list[PlaceHolderToken] = []
        seen = {}
        ts = []

        # -------- 1) 前向期间：仅保存占位 token --------
        def _pack(t):
            if t.is_leaf:
                return t
            if is_lora_weight(t):
                return t
            if not t.dtype in (torch.float32, torch.bfloat16):
                return t
            if not t.requires_grad:
                return t

            key = (t.data_ptr(), t.nbytes)
            if key in seen:
                return seen[key]()

            tok = PlaceHolderToken(None, t.dtype, t.shape, t.stride(), t.storage_offset())
            seen[key] = weakref.ref(tok)
            tokens.append(weakref.ref(tok))
            ts.append(t)
            
            return tok                      # 给 autograd 的就是它

        def _unpack(tok):
            if isinstance(tok, PlaceHolderToken):
                # if tok.t_ref.numel() > 0:
                #     return tok.t_ref
                return self.comp.decompress_sync(tok, torch.cuda.current_device())
                # return tok.t_ref
            return tok                    # 反向时会被替换成 token，再解压
        
        # def _pack(t):
        #     return t
        # def _unpack(t):
        #     return t

        with torch.autograd.graph.saved_tensors_hooks(_pack, _unpack):
            out = self.layer(*inp, **kw)      # 真正计算

        # -------- 2) layer 结束：开始异步 D→H + 压缩 --------
        # print(f"{len(tokens)=}")
        for i, tok_ptr in enumerate(tokens):
            tok = tok_ptr()
            # t = tok.t_ref
            t = ts[i]
            if t is None:                   # 理论上不会
                continue

            self.comp.kickoff(tok, t)

        del ts

        return out

# Inject before peft
def inject_async_compression(model: MistralModel | MistralForCausalLM | LlamaModel | LlamaForCausalLM,
                             compressor: AsyncCompressor):
    """Recursively wrap every decoder layer inside `model` with
    `DecoderLayerWrapper`.  Works for `MistralForCausalLM` and bare `MistralModel`."""

    if hasattr(model, "model"):
        container = model.model
    if hasattr(container, "model"):
        container = container.model
    else:  # bare MistralModel
        container = model
    # print(f"type(model) = {type(model)}")       # LlamaForCausalLM
    # print(f"type(model.model) = {type(model.model)}")  # LlamaModel
    # print(f"type(container) = {type(container)}")       # 期望也是 LlamaModel

    for i, layer in enumerate(container.layers):
        container.layers[i] = DecoderLayerWrapper(layer, compressor)
    return model

if args.hook and args.activation:
    compressor = AsyncCompressor()
    inject_async_compression(model, compressor)

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

EVAL=False
if EVAL:
    model.eval()
else:
    model.train()
    if not args.weight:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    else:
        trainable = (
            p for p in model.parameters()
            if isinstance(p, torch.nn.Parameter) and p.requires_grad
        )
        optimizer = torch.optim.AdamW(trainable, lr=2e-4)

if gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False  # for mistral or LLaMA, 在大多数 HuggingFace 的 decoder-only 模型（如 Mistral、LLaMA）中，use_cache=True 会导致模型跳过中间状态保存，从而禁用 gradient checkpointing。

bf16_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if args.autocast_context and args.finetune_type=="qlora" else contextlib.nullcontext()

def measure(n=args.round):
    if args.weight:
        prefetch_first_layer(layer2cps, layer_names)
    
    global decomp_time, comp_time, pack_time, comp_done_time, decomp_done_time, merge_time, prep_time1, prep_time2, prep_time3, prep_time4
    global unpack_time, post_time1, post_time2, post_time3
    global pp1_time, pp2_time, pp3_time, pp4_time, wait_comp_start

    for i in range(n):
        # wandb.log({"timestamp": time.time()-T0,
        #         "phase": "train_start", "round": i})

        print(f"\n\n*************** {i=} ****************")
        global rid
        rid = i
        if args.print_ratio:
            global orig_len, new_len
            orig_len = new_len = 0

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        peak0 = torch.cuda.max_memory_allocated(device)

        hook_ctx = contextlib.nullcontext()

        with hook_ctx:
            t0 = time.time()
            if EVAL:
                with torch.no_grad():
                    loss = model(**inputs).loss
            else:
                with bf16_ctx:
                    loss = model(**inputs).loss

            print(f"{loss=}")
            tf = time.time()

            if not EVAL:
                # prefetch_last_layer(layer2cps, layer_names)
                loss.backward()
                # print(f"After backward")
                optimizer.step()
                # print(f"After step")
                optimizer.zero_grad()
                # print(f"After zero grad")
            tb = time.time()
            t = (tb - t0)
            torch.cuda.synchronize()
            t1 = (time.time() - t0)

            peak = torch.cuda.max_memory_allocated(device)
            
            # 打「训练结束」锚点和 loss
            # wandb.log({"timestamp": time.time()-T0,
            #         "phase": "train_end", "round": i,
            #         "loss": loss.item()})

            print(f"Round {i}: {t*1000:.1f} ms , {t1*1000:.1f} ms , peak Δ={peak/1024/1024:.1f} MB")

            cpu_mem_peak = process.memory_info().rss / 1024 / 1024  # 单位: MB
            print(f"Peak CPU memory usage: {(cpu_mem_peak-cpu_mem_peak0):.2f} MB")
            print(f"- Fwd time: {(tf - t0)*1000:.1f} ms")
            print(f"- Bwd time: {(tb - tf)*1000:.1f} ms")

            print(f"Compression time: {comp_time*1000:.1f} ms")
            print(f"\tWait comp start time: {wait_comp_start*1000:.1f} ms")
            print(f"Decompression time: {decomp_time*1000:.1f} ms")
            print(f"\tWait comp done time: {comp_done_time*1000:.1f} ms")
            print(f"\tWait decomp done time: {decomp_done_time*1000:.1f} ms")
            print(f"\tWait merge done time: {merge_time*1000:.1f} ms")

            print(f"Pack time: {pack_time*1000:.1f} ms")
            print(f"\tPrep time 1: {prep_time1*1000:.1f} ms")
            print(f"\t\t - pp1: {pp1_time*1000:.1f} ms")
            print(f"\t\t - pp2: {pp2_time*1000:.1f} ms")
            print(f"\t\t - pp3: {pp3_time*1000:.1f} ms")
            print(f"\t\t - pp4: {pp4_time*1000:.1f} ms")
            print(f"\tPrep time 2: {prep_time2*1000:.1f} ms")
            print(f"\tPrep time 3: {prep_time3*1000:.1f} ms")
            print(f"\tPrep time 4: {prep_time4*1000:.1f} ms")

            print(f"Unpack time: {unpack_time*1000:.1f} ms")
            print(f"\tPost time 1: {post_time1*1000:.1f} ms")
            print(f"\tPost time 2: {post_time2*1000:.1f} ms")
            print(f"\tPost time 3: {post_time3*1000:.1f} ms")
            if args.print_ratio and orig_len > 0 and new_len > 0:
                print(f"{orig_len=}, {new_len=}, ratio={orig_len/new_len}")
            decomp_time = pack_time = comp_done_time = decomp_done_time = merge_time = prep_time1 = prep_time2 = prep_time3 = prep_time4 = 0
            unpack_time = post_time1 = post_time2 = post_time3 = 0
            pp1_time = pp2_time = pp3_time = pp4_time = 0
            orig_len = new_len = wait_comp_start = 0


            # for key in compressor.counter:
            #     print(f"{key=}, {compressor.counter[key]=}")

            # break

        if EVAL:
            break

measure()