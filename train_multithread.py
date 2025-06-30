import os, psutil, contextlib, weakref
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.8,max_split_size_mb:128"

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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from datasets import load_dataset
from accelerate import init_empty_weights
import float_split_stride_pin as fs_sp

import zstandard as zstd
import lz4.frame as lz4f, lz4.block as lz4b
import io, functools
from dataclasses import dataclass, field
import wandb, subprocess, re
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import torch.cuda.nvtx as nvtx
from typing import Union, List, Optional

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/TinyLlama-1.1B-Chat-v1.0")
parser.add_argument("--outdir", default="./weight_comp/prepare_weight/TinyLlama-1.1B-zstd-compressed-weights/level18")
parser.add_argument("--finetune_type", choices=["full", "lora", "qlora"], default="lora", help="Type of finetuning")
parser.add_argument("--autocast_context", action="store_true", help="Set torch.amp.autocast")
parser.add_argument("--check_diff", action="store_true",
                    help="同时加载原始模型并比对差值（耗显存）")
parser.add_argument("--hook", action="store_true", help="Run with compression hooks")
parser.add_argument("--debug", action="store_true", help="Run with debug")
parser.add_argument("--print_ratio", action="store_true", help="Print activation's compression ratio")
parser.add_argument("--print_time", action="store_true", help="Print activation's compression ratio")
parser.add_argument("--weight", default=False, action="store_true", help="Switch on weight compression")
parser.add_argument("--weight_async", default=False, action="store_true", help="Switch on asynchronous weight decompression")
parser.add_argument("--activation", default=False, action="store_true", help="Switch on activation compression")
parser.add_argument("--layerwise", default=False, action="store_true", help="Switch on layerwise activation compression")
parser.add_argument("--act_async", default=False, action="store_true", help="Switch on asynchronous activation (de)compression")
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


PHASE_LOAD   = "weight_load_end"   # 权重加载完
PHASE_EPOCH  = "round_end"         # 每轮训练完

nvmlInit(); gpu = nvmlDeviceGetHandleByIndex(0)
proj = ""
# if args.hook:
#     if not args.weight:
#         proj = "wrapper_memrift_jetson"
#     else:
#         proj = "memrift_act_weight_jetson"
# else:
#     if args.finetune_type == "lora":
#         proj = "vanilla_jetson"
#     elif args.finetune_type == "qlora":
#         if args.autocast_context:
#             proj = "qlora_amp_jetson"
#         else:
#             proj = "qlora_jetson"
proj = "memrift_act_weight_jetson"
if args.max_length != 2048:
    proj = f"{proj}_{args.max_length}"
if args.batch_size > 1:
    proj = f"{proj}_bs{args.batch_size}"

if "Mistral-7B" in args.model:
    proj = f"Mistral-7B_{proj}"
elif "Llama-3.1-8B" in args.model:
    proj = f"Llama-3.1-8B_{proj}"
elif "gemma-2-2b-it" in args.model:
    proj = f"gemma-2-2b-it_{proj}"
elif "Llama-3.2-3B-Instruct" in args.model:
    proj = f"Llama-3.2-3B-Instruct_{proj}"

wandb.init(project=proj, config=vars(args), group="group1")  # 把 CLI 参数也存进去
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

# Set GPU!
device = torch.device("cuda:0") 
torch.cuda.set_device(device) 

# --------------------------------------------------------------
#                1. Get Base model and LoRA Adapter 
# --------------------------------------------------------------

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

if args.hook and args.weight:
    # 0. 仅用 config 起“空架”，避免 from_pretrained 拉满权重进显存
    cfg    = AutoConfig.from_pretrained(args.model)
    with init_empty_weights():
        # if "gemma" in args.model:
        #     model  = AutoModelForCausalLM.from_config(cfg, torch_dtype=MODEL_TYPE, attn_implementation='eager')
        # else:
        model  = AutoModelForCausalLM.from_config(cfg, torch_dtype=MODEL_TYPE)
else:
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
        # if "gemma" in args.model:
        #     model = AutoModelForCausalLM.from_pretrained(args.model, 
        #                     torch_dtype=MODEL_TYPE, device_map={"": 0}, attn_implementation='eager')
        # else:
        model = AutoModelForCausalLM.from_pretrained(args.model, 
                        torch_dtype=MODEL_TYPE, device_map={"": 0})

_tls = threading.local()

def get_cctx(algo="zstd", level=-1):
    try:
        return _tls.cctx
    except AttributeError:
        if algo == "zstd":
            _tls.cctx = zstd.ZstdCompressor(level=level, write_checksum=False)
        elif algo == "lz4":
            _tls.cctx = lz4f.LZ4FrameCompressor(compression_level=level)
        return _tls.cctx

def get_dctx(algo="zstd"):
    try:
        return _tls.dctx
    except AttributeError:
        if algo == "zstd":
            _tls.dctx = zstd.ZstdDecompressor()
        elif algo == "lz4":
            _tls.dctx = lz4f.LZ4FrameDecompressor()
        return _tls.dctx

class CompressedParam(torch.nn.Parameter):
    def __new__(cls, orig_shape, sm_gpu, exp_mv, parent, attr, layer_id, typ):
        dummy = torch.empty(     # 0-element，占不了显存
                0, dtype=typ, device=sm_gpu.device
            )
            # 这里只能传 (data, requires_grad) 两个位置参数
        return super().__new__(cls, dummy, requires_grad=False)

    def __init__(self, orig_shape, sm_gpu, exp_mv, parent, attr, layer_id, typ):
        super().__init__()

        self.orig_shape   = tuple(orig_shape)
        self.sm_gpu    = sm_gpu            # uint8 pinned
        self.exp_mv    = exp_mv               # compressed exponent mv
        self._bf16     = None

        self._ready_event = threading.Event()
        self.CtoD_evt   = None
        self._hooked = False

        if args.debug:
            self.parent, self.attr = parent, attr
            self.layer_id = layer_id

    @classmethod
    def from_existing(cls,
                        other: "CompressedParam",
                        *,
                        sm_gpu: torch.Tensor | None = None,
                        exp_mv: memoryview | bytes | None = None,
                        parent=None,
                        attr: str | None = None,
                        layer_id: int | None = None):
        obj = cls(
                orig_shape = other.orig_shape,
                sm_gpu     = sm_gpu if sm_gpu is not None else other.sm_gpu,
                exp_mv     = exp_mv if exp_mv is not None else other.exp_mv,
                parent     = parent   if parent   is not None else getattr(other, "parent", None),
                attr       = attr     if attr     is not None else getattr(other, "attr",   None),
                layer_id   = layer_id if layer_id is not None else getattr(other, "layer_id", -1),
                typ        = other.dtype,                # 直接继承 dtype
            )
            # ➊ 记录继承来源（弱引用）
        obj.source = weakref.ref(other) 
        return obj

    # ------------ on-demand materialize ------------------------
    def materialize(self, sync=True):
        def c_contiguous_strides(shape):
            strides = [1] * len(shape)
            running = 1
            for i in range(len(shape) - 2, -1, -1):   # 从倒数第二维向前推
                running *= shape[i + 1]
                strides[i] = running
            return tuple(strides)

        """
        解压 exponent → 重建 bf16 → H2D copy
        调用方负责与计算流同步（可 wait_stream）。
        """
        try:
            if self._bf16 is not None:
                return self._bf16   # double-checked

            # CPU 解压
            numel = int(np.prod(self.orig_shape))

            # 1) 解压 exponent → pinned uint8
            self._exp_host = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
            if not hasattr(_tls, "dctx"):        # 该线程第一次用，创建解压器
                _tls.dctx = zstd.ZstdDecompressor()
            with _tls.dctx.stream_reader(memoryview(self.exp_mv)) as reader:
                view = memoryview(self._exp_host.numpy())   # numpy() 不复制，只拿 data_ptr
                nread = reader.readinto(view)
                assert nread == numel, "decompress size mismatch"

            # self._exp_host = decompress_into_pinned(self.exp_mv, numel)   # uint8 pinned, 记录为 class 成员，防止kernel执行时被释放

            # 2) 异步 H2D copy

            # with torch.cuda.stream(stream):
            stride = c_contiguous_strides(self.orig_shape)
            stream = torch.cuda.current_stream()
            with torch.cuda.stream(stream):
                self._bf16 = fs_sp.merge(self._exp_host, self.sm_gpu, self.orig_shape, stride, 0, MODEL_TYPE, stream.cuda_stream)
            ev = stream.record_event()

            self.CtoD_evt = ev
            self._ready_event.set()
            if args.debug:
                print(f"Setted event! layer={self.layer_id}: {self.parent}-{self.attr}")

            if sync:                       # ------- 同步路径 (漏预取) -------
                self.CtoD_evt.synchronize()
                    
                # return self._bf16
        except Exception as e:
            import traceback, sys
            traceback.print_exc(file=sys.stderr)
            raise                # 重新抛给上层，让你能看到

    def set_data(self):
            # print(f"[Syncing child]")
        if self._bf16 is None:
            self.data = torch.empty(0, dtype=torch.bfloat16, device=self.sm_gpu.device)
        else:
            self.data = self._bf16

    def set_param(self):
        # 现在才把最终对象挂回父模块
        if self._hooked:
            return
                
        assert self._bf16 is not None
        
        if hasattr(self, "_exp_host"):
            del self._exp_host
                
        self._bf16.record_stream(torch.cuda.current_stream())
        self.data = self._bf16
        _PTR2CP[self._bf16.data_ptr()] = self

        # 3) 只挂一次 backward hook
        self._hooked = True

    def release(self, delref=True):
        "回收 GPU 张量，恢复到压缩状态"
        if self._bf16 is None:
            return 

        if delref:  # called after backward, 防止之后重用空间的tensor 又识别到 cp
            del _PTR2CP[self._bf16.data_ptr()]

        del self._bf16
        self._bf16 = None
            
        self.data = torch.empty(0, dtype=torch.bfloat16, device=self.sm_gpu.device)
        self._ready_event.clear()
        self._hooked = False
        self.CtoD_evt = None
            
        # self.sync_childs()

    def release_comp(self):
        self.exp_mv = self.exp_host = self.sm_gpu = None


@dataclass
class PlaceHolderToken:
    dtype: torch.dtype
    shape: torch.Size
    stride: torch.Size
    offset: int
    # ↓ 运行时填充：
    # future: fut.Future | None = None
    # fut_id: int
    decomped_data: torch.Tensor | None = None
    ready_evt: threading.Event = field(default_factory=threading.Event)
    # CtoD_copy_evt: torch.cuda.Event | None = None

    def _clear_after_recover(self):
        # if hasattr(self, "cpu_exp"):
        #     del self.cpu_exp
        # if hasattr(self, "sm_bits"):
        #     del self.sm_bits
        if hasattr(self, "fut_id"):
            del self.fut_id
        # self.future = None
        self.ready_evt.clear()
        del self.CtoD_copy_evt
        self.CtoD_copy_evt = None

decomp_time = comp_time = 0
wait_comp_done = wait_comp_start = 0
orig_len = new_len = 0
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

    def __init__(self):
        # Dedicated stream for D→H copy + encode so we don't block the main
        # compute stream.  Feel free to expose this in your API.
        print("init AsyncCompressor")

        if not args.act_async and not args.weight_async:
            self.cctx = zstd.ZstdCompressor(level=args.level, threads=-1, write_checksum=False)
            self.dctx = zstd.ZstdDecompressor()
        else:
            self.pool_workers = os.cpu_count() // 2
            # --- NEW: Add Semaphores ---
            # Limit concurrent compressions to control peak CPU/GPU buffer memory
            if args.act_async:
                concurrency_limit = 3
                self.comp_semaphore = threading.Semaphore(value=self.pool_workers)
                self.decomp_semaphore = threading.Semaphore(value=concurrency_limit)
            if args.weight_async:
                concurrency_limit = 4
                # self.materialize_semaphore = threading.Semaphore(value=concurrency_limit)
                self.decomp_semaphore = threading.Semaphore(value=concurrency_limit)

        self._build()

    def _build(self):
        if args.act_async:
            self.compress_pool= fut.ThreadPoolExecutor(self.pool_workers)
            self.decode_pool  = fut.ThreadPoolExecutor(2)
        if args.weight_async:
            # self.materialize_pool = fut.ThreadPoolExecutor(2)
            self.decode_pool  = fut.ThreadPoolExecutor(4)
            # self.mat_stream = torch.cuda.Stream()
        self.d2h_stream   = torch.cuda.Stream()
        self.h2d_stream   = torch.cuda.Stream()
    
    def _reset(self):
        if args.act_async:
            for pool in (self.compress_pool, self.decode_pool):
                pool.shutdown(wait=True)
            del self.compress_pool
            del self.decode_pool
        if args.weight_async:
            # self.materialize_pool.shutdown(wait=True)
            # del self.materialize_pool
            if hasattr(self, "decode_pool"):
                self.decode_pool.shutdown(wait=True)
                del self.decode_pool
        
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        # torch._C._host_emptyCache()

        self._build()          # 彻底重建所有资源
        torch.cuda.reset_peak_memory_stats()
        # gc.collect()

    # ---------------------------------------------------------------------
    #  Interfaces you need to flesh out
    # ---------------------------------------------------------------------
    def kickoff_sync(self, tok: PlaceHolderToken, t: torch.Tensor):
        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        # stream = torch.cuda.current_stream()
        with torch.cuda.stream(self.d2h_stream):
            cpu_exp, sm_bits = fs_sp.split(t, self.d2h_stream.cuda_stream)
            evt = self.d2h_stream.record_event()   # 拷贝结束事件
        evt.synchronize()
        # tok.cpu_exp = cpu_exp
        tok.sm_bits = sm_bits
        # del t

        arr = cpu_exp.numpy()
        comped_bytes = self.cctx.compress(arr)
        tok.comped_cpu_exp = comped_bytes
        tok.numel = arr.size
    
    def kickoff_async(self, tok: PlaceHolderToken, t: torch.Tensor):


        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.d2h_stream):
            cpu_exp, sm_bits = fs_sp.split(t, self.d2h_stream.cuda_stream)
            evt = self.d2h_stream.record_event()   # 拷贝结束事件
            t.record_stream(self.d2h_stream)
        # evt.synchronize()
        tok.sm_bits = sm_bits
        # tok.t = t

        def _encode(cpu_exp, evt, algo="zstd"):
            self.comp_semaphore.acquire() # Wait for a free slot
            try:
                evt.synchronize()

                arr = cpu_exp.numpy()
                cctx = get_cctx(algo, args.level)
                comped_bytes = cctx.compress(arr)
                numel = arr.size

                if args.print_ratio:
                    global orig_len, new_len
                    orig_len += arr.nbytes
                    new_len += len(comped_bytes)
                
                return comped_bytes, numel
            except Exception as e:
                import traceback, sys
                traceback.print_exc(file=sys.stderr)
                raise                # 重新抛给上层，让你能看到
            finally:
                self.comp_semaphore.release() # Release the slot for others
                del cpu_exp
                del arr
                del evt
        
        # _encode(tok, cpu_exp, evt)
        fut = self.compress_pool.submit(_encode, cpu_exp, evt)

        return fut

    def decompress_sync(self, 
            tok: PlaceHolderToken):

        comped_bytes = tok.comped_cpu_exp
        numel = tok.numel
        cpu_exp = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
        with self.dctx.stream_reader(memoryview(comped_bytes)) as reader:
            view = memoryview(cpu_exp.numpy())   # numpy() 不复制，只拿 data_ptr
            nread = reader.readinto(view)
            assert nread == numel, "decompress size mismatch"

        sm_bits = tok.sm_bits
        stream = self.h2d_stream
        with torch.cuda.stream(stream):
            rst = fs_sp.merge(cpu_exp, sm_bits, tok.shape, tok.stride, tok.offset, tok.dtype, stream.cuda_stream)
        evt = stream.record_event()
        evt.synchronize()
        
        tok.decomped_data = rst

    def decompress_async(self, 
            tok: PlaceHolderToken,
            fut: fut.Future):

        def _decode(tok, fut, algo="zstd"):
            self.decomp_semaphore.acquire() # Wait for a free slot
            comped_bytes, numel = None, None
            try:
                comped_bytes, numel = fut.result()

                cpu_exp = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
                dctx = get_dctx(algo)
                with dctx.stream_reader(memoryview(comped_bytes)) as reader:
                    view = memoryview(cpu_exp.numpy())   # numpy() 不复制，只拿 data_ptr
                    nread = reader.readinto(view)
                    assert nread == numel, "decompress size mismatch"

                with torch.cuda.stream(self.h2d_stream):
                    rst = fs_sp.merge(cpu_exp, tok.sm_bits, tok.shape, tok.stride, tok.offset, tok.dtype, self.h2d_stream.cuda_stream)
                    tok.sm_bits.record_stream(self.h2d_stream)
                evt = self.h2d_stream.record_event()
                # print(f"waiting for merging")
                evt.synchronize()
                tok.CtoD_copy_evt = evt
                # tok.cpu_exp = cpu_exp
                tok.ready_evt.set()
                tok.decomped_data = rst

                # del tok.future
                # tok.future = None
                # del fut
            except Exception as e:
                import traceback, sys
                traceback.print_exc(file=sys.stderr)
                raise                # 重新抛给上层，让你能看到
            finally:
                self.decomp_semaphore.release() # Release the slot
                del fut
                if comped_bytes is not None:
                    del comped_bytes
                del tok.sm_bits
                del cpu_exp
                # del evt

        # _decode(tok)
        self.decode_pool.submit(_decode, tok, fut)
    
    def materialize_async(self, 
                            cp: CompressedParam,
                            sync: bool =True):
        if cp._bf16 is not None:
            return cp._bf16   # double-checked
                            
        def c_contiguous_strides(shape):
            strides = [1] * len(shape)
            running = 1
            for i in range(len(shape) - 2, -1, -1):   # 从倒数第二维向前推
                running *= shape[i + 1]
                strides[i] = running
            return tuple(strides)
        
        def _materialize(cp, sync, algo="zstd"):
            # self.materialize_semaphore.acquire()
            self.decomp_semaphore.acquire()
            try:
                # CPU 解压
                numel = int(np.prod(cp.orig_shape))

                # 1) 解压 exponent → pinned uint8
                dctx = get_dctx(algo)
                cp._exp_host = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
                with dctx.stream_reader(memoryview(cp.exp_mv)) as reader:
                    view = memoryview(cp._exp_host.numpy())   # numpy() 不复制，只拿 data_ptr
                    nread = reader.readinto(view)
                    assert nread == numel, "decompress size mismatch"
                # cp._exp_host = decompress_into_pinned(cp.exp_mv, numel)   # uint8 pinned, 记录为 class 成员，防止kernel执行时被释放

                # 2) 异步 H2D copy

                # with torch.cuda.stream(stream):
                stride = c_contiguous_strides(cp.orig_shape)
                # with torch.cuda.stream(self.mat_stream):
                #     bf16 = fs_sp.merge(cp._exp_host, cp.sm_gpu, cp.orig_shape, stride, 0, MODEL_TYPE, self.mat_stream.cuda_stream)
                # ev = self.mat_stream.record_event()
                with torch.cuda.stream(self.h2d_stream):
                    cp._bf16 = fs_sp.merge(cp._exp_host, cp.sm_gpu, cp.orig_shape, stride, 0, MODEL_TYPE, self.h2d_stream.cuda_stream)
                ev = self.h2d_stream.record_event()

                cp.CtoD_evt = ev
                cp._ready_event.set()

                if sync:                       # ------- 同步路径 (漏预取) -------
                    cp.CtoD_evt.synchronize()
                    
            except Exception as e:
                import traceback, sys
                traceback.print_exc(file=sys.stderr)
                raise                # 重新抛给上层，让你能看到
            finally:
                # self.materialize_semaphore.release()
                self.decomp_semaphore.release()
                if sync:
                    del cp._exp_host
                    del ev

        # _materialize(cp, sync)
        # self.materialize_pool.submit(_materialize, cp, sync)
        self.decode_pool.submit(_materialize, cp, sync)
    
def _unpack(tok, compressor):
    if args.weight:
        if isinstance(tok, CompressedParam):
            data = tok.source().data
            assert data.numel() > 0
            del tok
            return data
        elif isinstance(tok, tuple):
            cp, shape, stride = tok
            data = cp.source().data
            assert data.numel() > 0
            del cp
            rec_t = data.as_strided(shape, stride, 0)
            return rec_t

    # ----------------------------------------------------
    #               2. Check activations
    # ----------------------------------------------------
    if not args.activation:
        return tok

    if isinstance(tok, PlaceHolderToken):
                
        if args.act_async:
            if tok.decomped_data is not None and tok.CtoD_copy_evt is None:
            # if tok.future is None and tok.CtoD_copy_evt is None:
            #     assert tok.decomped_data is not None
                return tok.decomped_data
            tok.ready_evt.wait()
            # tok.CtoD_copy_evt.synchronize()
            tok.decomped_data.record_stream(torch.cuda.current_stream())    #!!
            tok._clear_after_recover()
        else:
            if tok.decomped_data is None:
                # self.comp.decompress_sync(tok)
                compressor.decompress_sync(tok)
                tok.decomped_data.record_stream(torch.cuda.current_stream())    #!!
        return tok.decomped_data

    return tok                    # 反向时会被替换成 token，再解压

class DecoderLayerWrapper(torch.nn.Module):
    def __init__(self, layer: nn.Module, compressor: AsyncCompressor, empty: bool):
        super().__init__()

        self.layer = layer
        self.comp = compressor          # 你的 AsyncCompressor

        self.tokens: list[PlaceHolderToken] = []
        self.futures = []
        self.do_empty = empty

        # --- NEW: Layer-specific LoRA pointer tracking ---
        self.lora_weight_ptrs = set()
        for name, p in self.layer.named_parameters():
            if (args.weight and p.requires_grad) or (not args.weight):
                self.lora_weight_ptrs.add(p.untyped_storage().data_ptr())

    def _is_lora_weight(self, tensor: torch.Tensor):
        return tensor.untyped_storage().data_ptr() in self.lora_weight_ptrs

    # ---------- forward ----------
    def forward(self, *inp, **kw):
        self.tokens.clear()
        self.futures.clear()
        seen = {}
        ts = []

        # -------- 1) 前向期间：仅保存占位 token --------
        def _pack(t):
            if args.weight:
                if isinstance(t, CompressedParam):
                    new_cp = CompressedParam.from_existing(t)
                    assert new_cp.data.numel() == 0
                    assert t.data.numel() > 0

                    return new_cp
                
                if not t.requires_grad:
                    cp = _PTR2CP.get(t.data_ptr(), None)
                    if cp is not None:
                        new_cp = CompressedParam.from_existing(cp)
                        assert new_cp.data.numel() == 0
                        assert cp.data.numel() > 0
                        return (new_cp, t.shape, t.stride())    # 就目前的参数来看，没有对参数做中间 view 的，都是整个转置，故不存 offset
            
            if not args.activation:
                return t
            if not t.dtype in (torch.float32, torch.bfloat16):
                return t
            if t.is_leaf:
                return t
            if not t.requires_grad:
                return t
            if self._is_lora_weight(t):
                return t

            MEM_THRESHOLD_BYTES = 1 * 1024 * 1024  # 1 MB
            if t.nbytes < MEM_THRESHOLD_BYTES:
                return t # For small tensors, do nothing and keep them on GPU.
            
            key = (t.data_ptr(), t.nbytes)

            if args.layerwise:
                # Impl.2 -- delay compression to the finish of layer computation
                if key in seen:
                    return seen[key]()
                tok = PlaceHolderToken(t.dtype, t.shape, t.stride(), t.storage_offset())
                seen[key] = weakref.ref(tok)
                self.tokens.append(weakref.ref(tok))
                ts.append(t)
            else:
                # Impl.1 -- start compression inside of the layer computation
                if key in seen:
                    tok_ref, t_ref = seen[key]
                    if t_ref() is not None:
                        if args.debug:
                            return tok_ref(), t
                        return tok_ref()
                tok = PlaceHolderToken(t.dtype, t.shape, t.stride(), t.storage_offset())
                seen[key] = (weakref.ref(tok), weakref.ref(t))
                if args.act_async:
                    fut = self.comp.kickoff_async(tok, t)
                    self.futures.append(fut)
                    tok.fut_id = len(self.futures) - 1
                else:
                    self.comp.kickoff_sync(tok, t)
                self.tokens.append(weakref.ref(tok))
                ts.append(weakref.ref(t))

            if args.debug:
                return tok, t                      # 给 autograd 的就是它
            return tok

        unpack_fn = functools.partial(_unpack, compressor=self.comp)
        with torch.autograd.graph.saved_tensors_hooks(_pack, unpack_fn):
            out = self.layer(*inp, **kw)      # 真正计算

        # -------- 2) layer 结束：开始异步 D→H + 压缩 -------
        if args.layerwise:
            # print(f"{len(tokens)=}")
            # Impl.1
            for i, tok_ptr in enumerate(self.tokens):
                tok = tok_ptr()
                t = ts[i]
                if t is None:                   # 理论上不会
                    continue

                if args.act_async:
                    self.comp.kickoff_async(tok, t)
                else:
                    self.comp.kickoff_sync(tok, t)
                del t
            
        # else:
        #     # Impl.2
        #     # print(f"{len(ts)=}")
        #     for t_ptr in ts:
        #         t = t_ptr()
        #         if t is None:
        #             continue
        #         del t

        del ts
        del seen
        if self.do_empty:
            torch.cuda.empty_cache()
        # torch._C._host_emptyCache()

        return out

class HookRuntime:
    def __init__(self, 
                    model: MistralModel | MistralForCausalLM | LlamaModel | LlamaForCausalLM, 
                    compressor: AsyncCompressor):
        self.model = model
        self.compressor = compressor
        
        if args.weight:
            self.layer_names = [f"base_model.model.model.layers.{i}"
                    for i in range(self.model.config.num_hidden_layers)]

            if "Llama-3.1-8B" in args.model:
                self.layer2cps, max_job_size = self.inject_from_files_old(args.outdir)
            else:
                self.layer2cps, max_job_size = self.inject_from_files(args.outdir)
            torch.cuda.empty_cache()
            self._report_meta()

            self._EMPTY_STEP = 5
            self.bwd_counter = 0

        if args.finetune_type in ["lora", "qlora"]:
            peft_config = LoraConfig(
                        lora_alpha=16,
                        lora_dropout=0.0,
                        r=16,
                        bias="none",
                        task_type="CAUSAL_LM",
                        target_modules= ["gate_proj", "up_proj", "down_proj"]
                )
            self.model = get_peft_model(self.model, peft_config, 
                                autocast_adapter_dtype=True)   # set this to keep the adapters in bfloat16
    
    def _reset(self):
        self.compressor._reset()
        self.bwd_counter = 0
            
    def _report_meta(self, limit=20):
        missing = [n for n, p in self.model.named_parameters() if p.is_meta]
        print(f"[meta] still {len(missing)} tensors on meta device")
        for n in missing[:limit]:
            print("  •", n)
            if n == "lm_head.weight":
                with torch.no_grad():                         # 避免进 autograd
                    self.model.lm_head.weight = self.model.model.embed_tokens.weight

    def inject_async_act_comp(self):

        if hasattr(self.model, "model"):
            container = self.model.model
        else:  # bare MistralModel
            container = self.model
        if hasattr(container, "model"):
            container = container.model
        # print(f"type(model) = {type(model)}")       # LlamaForCausalLM
        # print(f"type(model.model) = {type(model.model)}")  # LlamaModel
        # print(f"type(container) = {type(container)}")       # 期望也是 LlamaModel

        for i, layer in enumerate(container.layers):
            do_empty = False
            if i % 2 == 1:
                do_empty = True
            container.layers[i] = DecoderLayerWrapper(layer, compressor, do_empty)
            if args.activation and args.act_async:
                # if i > 0:
                #     def _bwd_pre_hook(_, __, prev_layer=container.layers[i-1], compressor=compressor):
                #         for tok_ptr in prev_layer.tokens:
                #             tok = tok_ptr()
                #             if tok is None:
                #                 continue
                #             compressor.decompress_async(tok)
                #     container.layers[i].register_full_backward_pre_hook(_bwd_pre_hook)
                if i >= 0:
                    def _bwd_pre_hook(_, __, layer=container.layers[i], compressor=compressor):
                        for tok_ptr in layer.tokens[::-1]:
                            tok = tok_ptr()
                            if tok is None:
                                continue
                            fut = layer.futures[tok.fut_id]
                            compressor.decompress_async(tok, fut)
                            layer.futures[tok.fut_id] = None
                    container.layers[i].register_full_backward_pre_hook(_bwd_pre_hook)
                def _bwd_hook(_, __, ___, layer=container.layers[i]):
                    for tok_ptr in layer.tokens:
                        tok = tok_ptr()
                        if tok is not None:
                            del tok
                    layer.tokens.clear()
                    layer.futures.clear()
                    # torch.cuda.empty_cache()
                    # torch._C._host_emptyCache()
                    # gc.collect()
                container.layers[i].register_full_backward_hook(_bwd_hook)

    def inject_from_files(self, comp_dir):
        layer2cps = {}  # {"base_model.model.model.layers.3": [cp1, cp2, ...]}
        max_job_size = 0

        idx = json.load(open(os.path.join(comp_dir, "index.json")))

        for it in idx:
            module, _, attr = it["name"].rpartition(".")
            mod = dict(self.model.named_modules())[module]
            file_path = os.path.join(comp_dir, it["file"])

            if it["scheme"] == "split_zstd":
                with open(file_path, "rb") as f:
                    numel, = struct.unpack("<Q", f.read(8))
                    sm     = np.frombuffer(f.read(numel * (1 if it["dtype"]=="bfloat16" else 3)),
                                        dtype=np.uint8)
                    exp    = f.read()

                sm_gpu = torch.as_tensor(sm, dtype=torch.uint8, device=device)
                dtype  = torch.bfloat16 if it["dtype"]=="bfloat16" else torch.float32
                del sm, numel
                    
                parts = module.split(".")
                layer_name = None
                layer_id = -1
                for i in range(len(parts) - 1):
                    if parts[i] == "layers":
                        layer_name = ".".join(parts[: i + 2])   # 保留 … layers.<idx>
                        layer_id = parts[i+1]
                        break    
                cp = CompressedParam(it["shape"], sm_gpu, exp, parent=mod, attr=attr, layer_id=layer_id, typ=dtype)

                if layer_name is None:
                    print(f"Materializing {module}")
                    cp.materialize(sync=True)
                    cp._ready_event.wait()
                    cp.CtoD_evt.synchronize()
                    cp.set_param()
                    cp.release_comp()
                    print(f"{cp.data.shape=}")
                else:
                    layer_name = f"base_model.model.{layer_name}"
                    # print(f"{layer_name=}")
                    layer2cps.setdefault(layer_name, []).append(cp)
                    max_job_size = max(len(layer2cps[layer_name]), max_job_size)
                    
                if attr in mod._parameters:
                    # del mod._parameters[attr]   # 删除原来的 Parameter
                    mod._parameters[attr] = cp           # property 安全
                else:
                    assert attr in mod._parameters
                    # setattr(mod, attr, cp)

            elif it["scheme"] == "raw_torch":
                print(f"raw_torch")
                raw = torch.load(file_path, map_location=device)
                if attr in mod._parameters:
                    del mod._parameters[attr]
                    mod._parameters[attr] = torch.nn.Parameter(raw, requires_grad=False)
        idx.clear()    
        return layer2cps, max_job_size

    def inject_from_files_old(self, comp_dir):
        def rm_prefixes(s):
            prefix = "base_model.model."
            if s.startswith(prefix):
                s = s[len(prefix):]
            pfix = ".base_layer"
            if s.endswith(pfix):
                s = s[:-len(pfix)]
            return s
        layer2cps = {}  # {"base_model.model.model.layers.3": [cp1, cp2, ...]}
        max_job_size = 0

        idx = json.load(open(os.path.join(comp_dir, "index.json")))

        for it in idx:
            # --- 打开文件并一次性读到内存 ---------------------------------
            with open(os.path.join(comp_dir, it["file"]), "rb") as f:
                buf = f.read()                       # ≤ 9 GB, 但一次只留两小段
            numel, = struct.unpack("<I", buf[:4])
            sign_off = 4
            exp_off  = sign_off + numel             # 1 byte / elem

            # --- 1) sign → gpu ------------------------------------------------
            sign_np   = np.frombuffer(buf, dtype=np.uint8,
                                    count=numel, offset=sign_off)
                
            sm_gpu = torch.as_tensor(sign_np, dtype=torch.uint8, device=device)
            del sign_np


            # --- 2) exponent 压缩段 → 独立 bytes（1.5 GB 左右） ---------------
            exp_bytes = bytes(buf[exp_off:])        # 只占压缩后大小
            # 让大 buffer 提前释放
            del buf

            # --- 3) 建 CompressedParam 并挂回模块 ------------------------------
            module, _, attr = it["name"].rpartition(".")
            module = rm_prefixes(module)
            mod = dict(self.model.named_modules())[module]
                
            # --- 记录到 layer2cps ---
            # 找到最近的 DecoderLayer 名称
            parts = module.split(".")
            layer_name = None
            layer_id = -1
            for i in range(len(parts) - 1):
                if parts[i] == "layers":
                    layer_name = ".".join(parts[: i + 2])   # 保留 … layers.<idx>
                    layer_id = parts[i+1]
                    break    
            cp = CompressedParam(it["shape"], sm_gpu, exp_bytes, parent=mod, attr=attr, layer_id=layer_id, typ=torch.bfloat16)
            # print(f"{layer_name=}")

            if layer_name is None:
                print(f"Materializing {module}")
                cp.materialize(sync=True)
                cp._ready_event.wait()
                cp.CtoD_evt.synchronize()
                cp.set_param()
                cp.release_comp()
                print(f"{cp.data.shape=}")
            else:
                layer_name = f"base_model.model.{layer_name}"
                layer2cps.setdefault(layer_name, []).append(cp)
                max_job_size = max(len(layer2cps[layer_name]), max_job_size)
                    
            if attr in mod._parameters:
                # del mod._parameters[attr]   # 删除原来的 Parameter
                mod._parameters[attr] = cp           # property 安全
            else:
                setattr(mod, attr, cp)

        idx.clear()    

        return layer2cps, max_job_size

    def install_fwd_prefetch_hooks(self):
        name2layer = {n: m for n, m in self.model.named_modules()}
        for cur, nxt in zip(self.layer_names[:-1], self.layer_names[1:]):
            nxt_pars = self.layer2cps.get(nxt, [])
            cur_pars = self.layer2cps.get(cur, [])

            def _hook_post(_, __, ___, cur_pars=cur_pars):
                for p in cur_pars:
                    p.release()  # 释放当前层的 CompressedParam
                    # p.release_comp()
                # torch.cuda.empty_cache()
            def _hook_pre(_, __, nxt_pars=nxt_pars, cur_pars=cur_pars, cur=cur):
                # print(f"Fwd hook pre {cur}")
                if args.weight_async:
                    for cp in nxt_pars:
                        self.compressor.materialize_async(cp, sync=False)
                for cp in cur_pars:
                    if not args.weight_async:
                        cp.materialize(sync=True)
                    cp._ready_event.wait()
                    cp.CtoD_evt.synchronize()
                    assert cp._bf16 is not None
                    cp.set_param()
            
            name2layer[cur].register_forward_pre_hook(_hook_pre)    # overlap current layer's computation with next layer's decompression
            name2layer[cur].register_forward_hook(_hook_post)   # release after calculation
        
        # -------- 最后一层单独处理 --------
        last = self.layer_names[-1]
        last_pars = self.layer2cps.get(last, [])

        def _last_pre(_, __, last_pars=last_pars):
            for cp in last_pars:
                if not args.weight_async:
                    cp.materialize(sync=True)
                cp._ready_event.wait()          # 等解压 + H2D
                cp.CtoD_evt.synchronize()
                cp.set_param()                  # 替换成 Tensor
        def _last_post(_, __, ___, last_pars=last_pars):
            for p in last_pars:
                p.release()  # 释放当前层的 CompressedParam
            torch.cuda.empty_cache()
        name2layer[last].register_forward_pre_hook(_last_pre)
        # name2layer[last].register_forward_hook(_last_post)

    def install_bwd_prefetch_hooks(self):   
        name2layer = {n: m for n, m in self.model.named_modules()}
        layer_names = self.layer_names[::-1]

        for cur, nxt in zip(layer_names[:-1], layer_names[1:]):
            nxt_pars = self.layer2cps.get(nxt, [])
            cur_pars = self.layer2cps.get(cur, [])

            def _hook_post(_, __, ___, cur_pars=cur_pars):
                for p in cur_pars:
                    p.release()  # 释放当前层的 CompressedParam
                # torch.cuda.empty_cache()
                self.bwd_counter += 1
                if self.bwd_counter % self._EMPTY_STEP == 0:
                    torch.cuda.empty_cache()
            def _hook_pre(_, __, nxt_pars=nxt_pars, cur_pars=cur_pars, cur=cur):
                # print(f"Bwd hook pre {cur}")
                if args.weight_async:
                    for cp in nxt_pars:
                        self.compressor.materialize_async(cp, sync=False)
                for cp in cur_pars:
                    if not args.weight_async:
                        cp.materialize(sync=True)
                    cp._ready_event.wait()
                    cp.CtoD_evt.synchronize()
                    cp.set_param()
            
            name2layer[cur].register_full_backward_hook(_hook_post)   # release after calculation
            name2layer[cur].register_full_backward_pre_hook(_hook_pre)    # overlap current layer's computation with next layer's decompression
        
        # -------- 最后一层单独处理 --------
        last = layer_names[-1]
        last_pars = self.layer2cps.get(last, [])

        def _last_pre(_, __, last_pars=last_pars):
            for cp in last_pars:
                if not args.weight_async:
                    cp.materialize(sync=True)
                cp._ready_event.wait()          # 等解压 + H2D
                cp.CtoD_evt.synchronize()
                cp.set_param()                  # 替换成 Tensor
                # cp.sync_childs()
        def _last_post(_, __, ___, last_pars=last_pars):
            for p in last_pars:
                p.release()  # 释放当前层的 CompressedParam
            torch.cuda.empty_cache()
        name2layer[last].register_full_backward_pre_hook(_last_pre) # 必须添加这个，为了覆盖 _PTR2CP 中对应的指针，防止访问到非法 tensor
        # name2layer[last].register_full_backward_hook(_hook_post)   # release after calculation

    def prefetch_layer(self, layer_id=-1):
        # This must be called when args.weight_async is set
        for cp in self.layer2cps.get(self.layer_names[layer_id], []):
            self.compressor.materialize_async(cp, sync=False)

if args.hook:
    compressor = AsyncCompressor()
    runtime = HookRuntime(model, compressor)
else:
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
      
if args.weight:

    def sizeof_tensor(t: torch.Tensor) -> int:
        """真实占用字节：元素数 × dtype 大小"""
        return t.numel() * t.element_size()

    def compressed_param_footprint(model, include_gpu=False):
        """
        统计:
        • sm_gpu          (GPU, uint8)
        • exp_mv / exp_bytes (CPU, 压缩后)
        • _bf16              (GPU bf16, 训练或已 materialize 时才有)
        • Python 对象开销    (sys.getsizeof)
        Returns: dict  { 'sign': bytes, 'exp': bytes, 'bf16': bytes, 'python': bytes }
        """
        tot = dict(sign=0, exp=0, bf16=0, python=0)

        visited_ids = set()          # 避免多次引用重复计算
        for mod in model.modules():
            for attr in ("weight", "bias"):
                obj = getattr(mod, attr, None)
                if not isinstance(obj, CompressedParam):
                    continue

                if id(obj) in visited_ids:
                    continue
                visited_ids.add(id(obj))

                if obj.exp_mv is None or obj.sm_gpu is None:
                    continue

                # ---- 1) sm_gpu (gpu) ----
                tot['sign'] += sizeof_tensor(obj.sm_gpu)

                # ---- 2) exponent bytes / memoryview --
                exp_len = len(obj.exp_mv) if hasattr(obj, "exp_mv") else len(obj.exp_bytes)
                tot['exp'] += exp_len

                # ---- 3) _bf16 GPU tensor --------------
                if include_gpu and obj._bf16 is not None:
                    tot['bf16'] += sizeof_tensor(obj._bf16)

                # ---- 4) Python object overhead -------
                tot['python'] += sys.getsizeof(obj)

        return tot

    def pretty_print_footprint(fp):
        import humanize
        for k, v in fp.items():
            print(f"{k:6s}: {humanize.naturalsize(v, binary=True)}")
        total = sum(fp.values())
        print(f"{'-'*15}\nTOTAL : {humanize.naturalsize(total, binary=True)}")

    fp = compressed_param_footprint(model, include_gpu=False)
    pretty_print_footprint(fp)

    runtime.install_fwd_prefetch_hooks()
    runtime.install_bwd_prefetch_hooks()
          
gc.collect()
process = psutil.Process(os.getpid())
cpu_mem_peak0 = process.memory_info().rss / 1024 / 1024  # 单位: MB
print(f"Peak CPU memory usage: {cpu_mem_peak0:.2f} MB")
print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated(device) / 1024 / 1024:.2f} MB")


# -------------------------------------------------------------------------
#                           Activation compression
# -------------------------------------------------------------------------

        
if args.hook:
    runtime.inject_async_act_comp()


T_load = time.time() - T0
wandb.log({
    "timestamp": int((time.time()-T0)*1000),
    "phase": PHASE_LOAD,
    "weight_load_sec": T_load,
})

dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
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
texts = dataset['text']
longest_texts = sorted(texts, key=len, reverse=True)[:args.batch_size]
# sample_text = [dataset['train'][i]['text'] for i in range(args.batch_size)]
encoding = tokenizer(
    longest_texts,
    max_length=args.max_length,
    padding='max_length',
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
    if "gemma" in args.model:
        model.config.return_dict = False

if gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False  # for mistral or LLaMA, 在大多数 HuggingFace 的 decoder-only 模型（如 Mistral、LLaMA）中，use_cache=True 会导致模型跳过中间状态保存，从而禁用 gradient checkpointing。

bf16_ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if args.autocast_context and args.finetune_type=="qlora" else contextlib.nullcontext()

def measure(n=args.round):
    
    global T_load
    global decomp_time, comp_time, pack_time, comp_done_time, decomp_done_time, merge_time, prep_time1, prep_time2, prep_time3, prep_time4
    global unpack_time, post_time1, post_time2, post_time3
    global pp1_time, pp2_time, pp3_time, pp4_time, wait_comp_start

    round_times = np.zeros(n)
    t_start = time.time()
    if args.weight_async:
        runtime.prefetch_layer(0)
    for i in range(n):

        print(f"\n\n*************** {i=} ****************")
        global rid
        rid = i
        if args.print_ratio:
            global orig_len, new_len
            orig_len = new_len = 0

        torch.cuda.empty_cache()
        torch._C._host_emptyCache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        peak0 = torch.cuda.max_memory_allocated(device)

        # t0 = time.time()
        if EVAL:
            with torch.no_grad():
                loss = model(**inputs).loss
        else:
            with bf16_ctx:
                if "gemma" in args.model:
                    out = model(**inputs, return_dict=False)
                    loss = out[0] if isinstance(out, tuple) else out.loss
                else:   
                    loss = model(**inputs).loss

        print(f"{loss=}")
        tf = time.time()

        if not EVAL:
            # if args.weight_async:
            #     runtime.prefetch_layer(-1)
            # if args.hook and args.act_async:
            #     model.model.model.layers[-1].decomp_tokens()
            loss.backward()
            # print(f"After backward")
            optimizer.step()
            # print(f"After step")
            optimizer.zero_grad()
            if args.hook:
                runtime._reset()
            # print(f"After zero grad")
            
        # t = (tb - t0)
        torch.cuda.synchronize()
        tb = time.time()
        # t1 = (time.time() - t0)

        # peak = torch.cuda.max_memory_allocated(device)

        # print(f"Round {i}: {t*1000:.1f} ms , {t1*1000:.1f} ms , peak Δ={peak/1024/1024:.1f} MB")

        # cpu_mem_peak = process.memory_info().rss / 1024 / 1024  # 单位: MB
        # print(f"Peak CPU memory usage: {(cpu_mem_peak-cpu_mem_peak0):.2f} MB")
        # print(f"- Fwd time: {(tf - t0)*1000:.1f} ms")
        # print(f"- Bwd time: {(tb - tf)*1000:.1f} ms")

        # print(f"Compression time: {comp_time*1000:.1f} ms")
        # print(f"\tWait comp start time: {wait_comp_start*1000:.1f} ms")
        # print(f"Decompression time: {decomp_time*1000:.1f} ms")
        # print(f"\tWait comp done time: {comp_done_time*1000:.1f} ms")
        # print(f"\tWait decomp done time: {decomp_done_time*1000:.1f} ms")
        # print(f"\tWait merge done time: {merge_time*1000:.1f} ms")

        # print(f"Pack time: {pack_time*1000:.1f} ms")
        # print(f"\tPrep time 1: {prep_time1*1000:.1f} ms")
        # print(f"\t\t - pp1: {pp1_time*1000:.1f} ms")
        # print(f"\t\t - pp2: {pp2_time*1000:.1f} ms")
        # print(f"\t\t - pp3: {pp3_time*1000:.1f} ms")
        # print(f"\t\t - pp4: {pp4_time*1000:.1f} ms")
        # print(f"\tPrep time 2: {prep_time2*1000:.1f} ms")
        # print(f"\tPrep time 3: {prep_time3*1000:.1f} ms")
        # print(f"\tPrep time 4: {prep_time4*1000:.1f} ms")

        # print(f"Unpack time: {unpack_time*1000:.1f} ms")
        # print(f"\tPost time 1: {post_time1*1000:.1f} ms")
        # print(f"\tPost time 2: {post_time2*1000:.1f} ms")
        # print(f"\tPost time 3: {post_time3*1000:.1f} ms")
        # if args.print_ratio and orig_len > 0 and new_len > 0:
        #     print(f"{orig_len=}, {new_len=}, ratio={orig_len/new_len}")
        # decomp_time = pack_time = comp_done_time = decomp_done_time = merge_time = prep_time1 = prep_time2 = prep_time3 = prep_time4 = 0
        # unpack_time = post_time1 = post_time2 = post_time3 = 0
        # pp1_time = pp2_time = pp3_time = pp4_time = 0
        # orig_len = new_len = wait_comp_start = 0

        # break

        if EVAL:
            break

        dur   = time.time() - t_start               # 纯训练耗时
        dur_f = tf - t_start               # forward
        dur_b = tb - tf               # backward

        wandb.log({
            "timestamp": int((time.time()-T0)*1000),
            "phase": PHASE_EPOCH,
            "round": i,
            "round_sec": dur,
            "round_fwd_sec": dur_f,
            "round_bwd_sec": dur_b,
            "loss": loss.item(),
        })
        t_start = time.time()

        round_times[i] = dur

    avg_t  = np.mean(round_times)
    std_t  = np.std(round_times)
    wandb.summary["round_time_mean_sec"] = avg_t
    wandb.summary["round_time_std_sec"]  = std_t
    wandb.summary["weight_load_sec"]  = T_load
    
measure()