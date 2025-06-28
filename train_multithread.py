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
parser.add_argument("--activation", default=False, action="store_true", help="Switch on activation compression")
parser.add_argument("--layerwise", default=False, action="store_true", help="Switch on layerwise activation compression")
parser.add_argument("--asynchronous", default=False, action="store_true", help="Switch on asynchronous activation (de)compression")
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
orig_len = new_len = 0

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

decomp_lock = threading.Lock()
decomp_time = comp_time = 0
if args.weight:
    index = []
    # 模块级别，全局只有这一份；属性却会被线程隔离
    def decompress_into_pinned(bytes_like, numel):
        """把 exponent 压缩数据直接解压到 pinned Tensor"""
        global decomp_time, decomp_lock
        t0 = time.time()
        # 1) 申请 pinned buffer
        buf = torch.empty(numel, dtype=torch.uint8, pin_memory=True)

        tls = _tls                          # 取到当前线程的 thread-local 对象
        if not hasattr(tls, "dctx"):        # 该线程第一次用，创建解压器
            tls.dctx = zstd.ZstdDecompressor()
        with tls.dctx.stream_reader(memoryview(bytes_like)) as reader:
            view = memoryview(buf.numpy())   # numpy() 不复制，只拿 data_ptr
            nread = reader.readinto(view)
            assert nread == numel, "decompress size mismatch"

        if args.print_time:
            with decomp_lock:
                decomp_time += (time.time() - t0)
        return buf

    def c_contiguous_strides(shape):
        strides = [1] * len(shape)
        running = 1
        for i in range(len(shape) - 2, -1, -1):   # 从倒数第二维向前推
            running *= shape[i + 1]
            strides[i] = running
        return tuple(strides)

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
            self._exp_host = None
            self._bf16     = None
            self.childs    = []

            self._ready_event = threading.Event()
            self.CtoD_evt   = None
            self._hooked = False
            self.stm   = torch.cuda.Stream()

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
            # other.childs.append(weakref.ref(obj))
            return obj

        # ------------ on-demand materialize ------------------------
        def materialize(self, sync=True):
            """
            解压 exponent → 重建 bf16 → H2D copy
            调用方负责与计算流同步（可 wait_stream）。
            """
            try:

                if self._bf16 is not None:
                    return self._bf16   # double-checked

                if args.debug:
                    print(f"Materializing layer={self.layer_id}: {self.parent}-{self.attr}, {self.orig_shape=}")

                # CPU 解压
                numel = int(np.prod(self.orig_shape))

                # 1) 解压 exponent → pinned uint8
                self._exp_host = decompress_into_pinned(self.exp_mv, numel)   # uint8 pinned, 记录为 class 成员，防止kernel执行时被释放

                # 2) 异步 H2D copy

                # with torch.cuda.stream(stream):
                stride = c_contiguous_strides(self.orig_shape)
                with torch.cuda.stream(self.stm):
                    bf16 = fs_sp.merge(self._exp_host, self.sm_gpu, self.orig_shape, stride, 0, MODEL_TYPE, self.stm.cuda_stream)
                ev = self.stm.record_event()
                self._bf16 = bf16.view(self.orig_shape)

                # ev = torch.cuda.Event()
                # ev.record()          # ensure all copies/kernels before event
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

        def sync_childs(self):
            new_childs = []
            for child_ptr in self.childs:
                child = child_ptr()
                if child is None:
                    continue
                child._bf16 = self._bf16
                child.set_data()
                new_childs.append(weakref.ref(child))
            self.childs = new_childs
                # if self._bf16 is not None:
                #     print(f"{self._bf16.shape=}, {child._bf16.shape=}, {child.data.numel()=}")
                # with torch.no_grad():                    # 避免误进 autograd
                #     child.data.set_(self.data)           # ← 关键，一步到位

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

            if self._exp_host is not None:
                del self._exp_host
                self._exp_host = None
                
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

    def compressed_param_memory(model, verbose=False):
        """
        统计当前所有 CompressedParam 的 CPU / GPU 占用（字节数）
        返回:
            {
                "total_cpu": int,   # 字节
                "total_gpu": int,   # 字节
                "details": [ (qualified_name, cpu_bytes, gpu_bytes), ... ]
            }
        """
        total_cpu = total_gpu = 0
        details = []

        for mod_name, mod in model.named_modules():
            # 这里只统计常见 attr；如果你还有其他名字自行补充
            for attr in ("weight", "bias"):
                obj = getattr(mod, attr, None)
                if not isinstance(obj, CompressedParam):
                    continue

                cpu = gpu = 0

                # -------- CPU ----------
                cpu += obj.sm_gpu.numel() * obj.sm_gpu.element_size()  # uint8 → 1B
                cpu += len(obj.exp_mv)                                       # 压缩后 bytes
                if getattr(obj, "exp_host", None) is not None:
                    cpu += obj.exp_host.numel() * obj.exp_host.element_size()

                # -------- GPU ----------
                if getattr(obj, "_bf16", None) is not None:
                    gpu += obj._bf16.numel() * obj._bf16.element_size()
                if getattr(obj, "sign_gpu", None) is not None:
                    gpu += obj.sign_gpu.numel() * obj.sign_gpu.element_size()
                if getattr(obj, "exp_gpu", None) is not None:
                    gpu += obj.exp_gpu.numel() * obj.exp_gpu.element_size()

                total_cpu += cpu
                total_gpu += gpu
                details.append((f"{mod_name}.{attr}", cpu, gpu))

        if verbose:
            print("-" * 60)
            print(f"{'Param':50s} | {'CPU':>10s} | {'GPU':>10s}")
            print("-" * 60)
            for name, cpu, gpu in details:
                print(f"{name:50s} | {humanize.naturalsize(cpu, binary=True):>10s} | "
                    f"{humanize.naturalsize(gpu, binary=True):>10s}")
            print("-" * 60)
            print(f"Total CPU: {humanize.naturalsize(total_cpu, binary=True)}  |  "
                f"Total GPU: {humanize.naturalsize(total_gpu, binary=True)}")
            print("-" * 60)

        return {
            "total_cpu": total_cpu,
            "total_gpu": total_gpu,
            "details": details,
        }

    def inject_from_files(model, comp_dir):
        layer2cps = {}  # {"base_model.model.model.layers.3": [cp1, cp2, ...]}
        max_job_size = 0

        idx = json.load(open(os.path.join(comp_dir, "index.json")))

        for it in idx:
            module, _, attr = it["name"].rpartition(".")
            mod = dict(model.named_modules())[module]
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

    if "Llama-3.1-8B" in args.model:
        def rm_prefixes(s):
            prefix = "base_model.model."
            if s.startswith(prefix):
                s = s[len(prefix):]
            pfix = ".base_layer"
            if s.endswith(pfix):
                s = s[:-len(pfix)]
            return s
        def inject_from_files(model, comp_dir):
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
                # sign_host = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
                # sign_host.copy_(torch.as_tensor(sign_np, dtype=torch.uint8))
                # # 释放 view；原 buf 很快会被 GC，因为我们马上切 exponent
                
                sm_gpu = torch.as_tensor(sign_np, dtype=torch.uint8, device=device)
                del sign_np


                # --- 2) exponent 压缩段 → 独立 bytes（1.5 GB 左右） ---------------
                exp_bytes = bytes(buf[exp_off:])        # 只占压缩后大小
                # 让大 buffer 提前释放
                del buf

                # --- 3) 建 CompressedParam 并挂回模块 ------------------------------
                module, _, attr = it["name"].rpartition(".")
                module = rm_prefixes(module)
                mod = dict(model.named_modules())[module]
                
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

    # ---------- 工具函数 ----------
    def bytes2mb(n): return n / (1024 ** 2)

    def snapshot_mem(tag=""):
        """返回 (gpu_alloc_MB, gpu_reserved_MB, cpu_rss_MB)"""
        torch.cuda.synchronize()
        gc.collect(); torch.cuda.empty_cache()
        alloc   = torch.cuda.memory_allocated()
        reserv  = torch.cuda.memory_reserved()
        rss     = psutil.Process(os.getpid()).memory_info().rss
        print(f"[{tag}] GPU: allocated={bytes2mb(alloc):.1f} MB, "
            f"reserved={bytes2mb(reserv):.1f} MB | "
            f"CPU RSS={bytes2mb(rss):.1f} MB")
        return alloc, reserv, rss

    # ------------------------------------------------------------
    # helper 1 ── 返回模型所有权重的 “逻辑字节数”
    #            （不含梯度、优化器状态，也不含激活）
    # ------------------------------------------------------------
    def model_size_bytes(model):
        total = 0
        for name, obj in model.named_parameters():
            if isinstance(obj, torch.nn.Parameter):
                total += obj.numel() * obj.element_size()
            elif isinstance(obj, CompressedParam):
                # print(f"{name=}")
                # sm_gpu: pinned uint8 Tensor
                total += obj.sm_gpu.numel()          # 1B × n
                # exp_bytes：压缩后 exponent（bytes 对象 or memoryview）
                total += len(obj.exp_mv if hasattr(obj, "exp_mv")
                            else obj.exp_bytes)
        return total

    def pretty(nbytes):          # 123456 → '117 MiB'
        return humanize.naturalsize(nbytes, binary=True)

    def print_model_size(tag, model):
        print(f"[{tag}] logical param size = {pretty(model_size_bytes(model))}")

    # ----------------------------------------------------------
    # 1) 载入模型 + get_peft_model 之后，先测一次
    snapshot_mem("before_inject")
    print_model_size("before inject", model) 

    # 2) 执行权重量化压缩
    torch.cuda.reset_peak_memory_stats()
    layer2cps, max_job_size = inject_from_files(model, args.outdir)
    torch.cuda.empty_cache()

    # 3) 再测一次
    snapshot_mem("after_inject")
    print_model_size("after inject", model) 

    gc.collect()
    process = psutil.Process(os.getpid())
    cpu_mem_peak1 = process.memory_info().rss / 1024 / 1024  # 单位: MB
    print(f"Peak CPU memory usage: {cpu_mem_peak1:.2f} MB")
    print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated(device) / 1024 / 1024:.2f} MB")


def report_meta(model, limit=20):
    missing = [n for n, p in model.named_parameters() if p.is_meta]
    print(f"[meta] still {len(missing)} tensors on meta device")
    for n in missing[:limit]:
        print("  •", n)
        if n == "lm_head.weight":
            with torch.no_grad():                         # 避免进 autograd
                model.lm_head.weight = model.model.embed_tokens.weight
report_meta(model)

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
    # for n, m in model.named_modules():
    #     if hasattr(m, "base_layer"):
    #         w = m.base_layer.weight
    #         print(f"{n}.base_layer.weight device: {w.device}, shape: {w.shape}")

    # for n, p in model.named_parameters():
    #     if isinstance(p, CompressedParam):
    #         print(n, p.shape, p.__class__.__name__)

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

    # ---------------- 使用示例 ----------------
    fp = compressed_param_footprint(model, include_gpu=False)
    pretty_print_footprint(fp)

    # --------------------------------------------------------------
    #   3. pipeline prefetch (layer n 计算 ↔ layer n+1 copy) 
    # --------------------------------------------------------------
    # num_workers = min(9, os.cpu_count() // 3) 
    # _pref_q = queue.Queue(maxsize=10)  # 64 jobs per worker
    # def _pref_worker(worker_id):
    #     torch.cuda.set_device(device)
    #     if not hasattr(_tls, "dctx"):
    #         _tls.dctx = zstd.ZstdDecompressor()
    #     while True:
    #         cp = _pref_q.get()
    #         if cp is None:
    #             break
    #         cp.materialize(sync=False)      # ← 各线程自己的 stream
    #         _pref_q.task_done()
    
    # 起线程池
    # workers = [threading.Thread(target=_pref_worker, args=(i,), daemon=True)
    #         for i in range(num_workers)]
    # for w in workers: w.start()

    def prefetch_first_layer(layer2cps, layer_names):
        # 首层预取
        for cp in layer2cps.get(layer_names[0], []):
            cp.materialize(sync=True)
            cp._ready_event.wait()
            cp.CtoD_evt.synchronize()
            cp.set_param()

            # try: 
            #     _pref_q.put_nowait(p)
            # except queue.Full: 
            #     print(f"The queue is full!")
            #     pass
            
    def install_fwd_prefetch_hooks(model, layer_names, layer2cps):
        name2layer = {n: m for n, m in model.named_modules()}
        for cur, nxt in zip(layer_names[:-1], layer_names[1:]):
            nxt_pars = layer2cps.get(nxt, [])
            cur_pars = layer2cps.get(cur, [])

            def _hook_post(_, __, ___, cur_pars=cur_pars):
                for p in cur_pars:
                    p.release()  # 释放当前层的 CompressedParam
                    # p.release_comp()
                torch.cuda.empty_cache()
            def _hook_pre(_, __, nxt_pars=nxt_pars, cur_pars=cur_pars):
                for cp in cur_pars:
                    cp.materialize(sync=True)
                    assert cp._bf16 is not None
                    cp._ready_event.wait()
                    cp.CtoD_evt.synchronize()
                    cp.set_param()
                    # cp.sync_childs()
                    # print(f"{cp.data.shape=}")
                    # cp.release_comp()
                    # print(f'{cp.data.shape=}')

                # for p in nxt_pars:
                #     try: 
                #         _pref_q.put_nowait(p)
                #         # _pref_q.put(p)
                #         # print(f"Pushed into queue")
                #     except queue.Full: 
                #         print(f"The queue is full!")
                #         pass
            
            name2layer[cur].register_forward_pre_hook(_hook_pre)    # overlap current layer's computation with next layer's decompression
            name2layer[cur].register_forward_hook(_hook_post)   # release after calculation
        
        # -------- 最后一层单独处理 --------
        last = layer_names[-1]
        last_pars = layer2cps.get(last, [])

        def _last_pre(_, __, last_pars=last_pars):
            for cp in last_pars:
                cp.materialize(sync=True)
                cp._ready_event.wait()          # 等解压 + H2D
                cp.CtoD_evt.synchronize()
                cp.set_param()                  # 替换成 Tensor
                # cp.sync_childs()
        def _last_post(_, __, ___, last_pars=last_pars):
            for p in last_pars:
                p.release()  # 释放当前层的 CompressedParam
            torch.cuda.empty_cache()
        name2layer[last].register_forward_pre_hook(_last_pre)
        name2layer[last].register_forward_hook(_last_post)

    def install_bwd_prefetch_hooks(model, layer_names, layer2cps):   
        name2layer = {n: m for n, m in model.named_modules()}

        for cur, nxt in zip(layer_names[:-1], layer_names[1:]):
            nxt_pars = layer2cps.get(nxt, [])
            cur_pars = layer2cps.get(cur, [])

            def _hook_post(_, __, ___, cur_pars=cur_pars):
                for p in cur_pars:
                    p.release()  # 释放当前层的 CompressedParam
                torch.cuda.empty_cache()
            def _hook_pre(_, __, nxt_pars=nxt_pars, cur_pars=cur_pars):
                # print(f"Bwd hook pre {cur}")
                for cp in cur_pars:
                    cp.materialize(sync=True)
                    cp._ready_event.wait()
                    cp.CtoD_evt.synchronize()
                    cp.set_param()
                    # cp.sync_childs()
                # for p in nxt_pars:
                #     try: 
                #         _pref_q.put_nowait(p)
                #         # _pref_q.put(p)
                #         # print(f"Pushed into queue")
                #     except queue.Full: 
                #         print(f"The queue is full!")
                #         pass
            
            name2layer[cur].register_full_backward_hook(_hook_post)   # release after calculation
            name2layer[cur].register_full_backward_pre_hook(_hook_pre)    # overlap current layer's computation with next layer's decompression
        
        # -------- 最后一层单独处理 --------
        last = layer_names[-1]
        last_pars = layer2cps.get(last, [])

        def _last_pre(_, __, last_pars=last_pars):
            for cp in last_pars:
                # print(f"Waiting for {cp.parent}-{cp.attr}")
                cp.materialize(sync=True)
                cp._ready_event.wait()          # 等解压 + H2D
                cp.CtoD_evt.synchronize()
                cp.set_param()                  # 替换成 Tensor
                # cp.sync_childs()
        def _last_post(_, __, ___, last_pars=last_pars):
            # print(f"Last post!")
            for p in last_pars:
                p.release()  # 释放当前层的 CompressedParam
            torch.cuda.empty_cache()
        name2layer[last].register_full_backward_pre_hook(_last_pre) # 必须添加这个，为了覆盖 _PTR2CP 中对应的指针，防止访问到非法 tensor
        name2layer[last].register_full_backward_hook(_hook_post)   # release after calculation

   
    # def install_fwd_prefetch_hooks(model, layer_names, layer2cps):
    #     name2layer = {n: m for n, m in model.named_modules()}
    #     for last, cur in zip(layer_names[:-1], layer_names[1:]):
    #         last_pars = layer2cps.get(last, [])
    #         cur_pars = layer2cps.get(cur, [])

    #         # def _hook_post(_, __, ___, cur_pars=cur_pars):
    #         #     for p in cur_pars:
    #         #         p.release()  # 释放当前层的 CompressedParam
    #         #         # p.release_comp()
    #         #     torch.cuda.empty_cache()
    #         def _hook_pre(_, __, last_pars=last_pars, cur_pars=cur_pars):
    #             for cp in last_pars:
    #                 cp.release()
    #             torch.cuda.empty_cache()

    #             for cp in cur_pars:
    #                 cp.materialize(sync=True)
    #                 assert cp._bf16 is not None
    #                 cp._ready_event.wait()
    #                 cp.CtoD_evt.synchronize()
    #                 cp.set_param()
    #                 # cp.sync_childs()
    #                 # print(f"{cp.data.shape=}")
    #                 # cp.release_comp()
    #                 # print(f'{cp.data.shape=}')

    #             # for p in nxt_pars:
    #             #     try: 
    #             #         _pref_q.put_nowait(p)
    #             #         # _pref_q.put(p)
    #             #         # print(f"Pushed into queue")
    #             #     except queue.Full: 
    #             #         print(f"The queue is full!")
    #             #         pass
            
    #         name2layer[cur].register_forward_pre_hook(_hook_pre)    # overlap current layer's computation with next layer's decompression
    #         # name2layer[cur].register_forward_hook(_hook_post)   # release after calculation
        
    #     # -------- 第一层单独处理 --------
    #     last = layer_names[0]
    #     last_pars = layer2cps.get(last, [])

    #     def _last_pre(_, __, last_pars=last_pars):
    #         for cp in last_pars:
    #             cp.materialize(sync=True)
    #             cp._ready_event.wait()          # 等解压 + H2D
    #             cp.CtoD_evt.synchronize()
    #             cp.set_param()                  # 替换成 Tensor
    #             # cp.sync_childs()
    #     name2layer[last].register_forward_pre_hook(_last_pre)
    #     # name2layer[last].register_forward_hook(_last_post)

    #     # -------- 最后一层单独处理 --------
    #     last = layer_names[-1]
    #     last_pars = layer2cps.get(last, [])
    #     def _last_post(_, __, ___, last_pars=last_pars):
    #         for p in last_pars:
    #             p.release()  # 释放当前层的 CompressedParam
    #         torch.cuda.empty_cache()
    #     name2layer[last].register_forward_hook(_last_post)


    # def install_bwd_prefetch_hooks(model, layer_names, layer2cps):   
    #     name2layer = {n: m for n, m in model.named_modules()}

    #     for last, cur in zip(layer_names[:-1], layer_names[1:]):
    #         last_pars = layer2cps.get(last, [])
    #         cur_pars = layer2cps.get(cur, [])

    #         # def _hook_post(_, __, ___, cur_pars=cur_pars):
    #         #     for p in cur_pars:
    #         #         p.release()  # 释放当前层的 CompressedParam
    #         #     torch.cuda.empty_cache()
    #         def _hook_pre(_, __, last_pars=last_pars, cur_pars=cur_pars):
    #             # print(f"Bwd hook pre {cur}")
    #             for cp in last_pars:
    #                 cp.release()
    #             torch.cuda.empty_cache()

    #             for cp in cur_pars:
    #                 cp.materialize(sync=True)
    #                 cp._ready_event.wait()
    #                 cp.CtoD_evt.synchronize()
    #                 cp.set_param()
    #                 # cp.sync_childs()
    #             # for p in nxt_pars:
    #             #     try: 
    #             #         _pref_q.put_nowait(p)
    #             #         # _pref_q.put(p)
    #             #         # print(f"Pushed into queue")
    #             #     except queue.Full: 
    #             #         print(f"The queue is full!")
    #             #         pass
            
    #         # name2layer[cur].register_full_backward_hook(_hook_post)   # release after calculation
    #         name2layer[cur].register_full_backward_pre_hook(_hook_pre)    # overlap current layer's computation with next layer's decompression
        
    #     # -------- 第一层单独处理 --------
    #     last = layer_names[0]
    #     last_pars = layer2cps.get(last, [])

    #     def _last_pre(_, __, last_pars=last_pars):
    #         for cp in last_pars:
    #             # print(f"Waiting for {cp.parent}-{cp.attr}")
    #             cp.materialize(sync=True)
    #             cp._ready_event.wait()          # 等解压 + H2D
    #             cp.CtoD_evt.synchronize()
    #             cp.set_param()                  # 替换成 Tensor
    #             # cp.sync_childs()
    #     name2layer[last].register_full_backward_pre_hook(_last_pre) # 必须添加这个，为了覆盖 _PTR2CP 中对应的指针，防止访问到非法 tensor

    #     last = layer_names[-1]
    #     last_pars = layer2cps.get(last, [])
    #     def _last_post(_, __, ___, last_pars=last_pars):
    #         # print(f"Last post!")
    #         for p in last_pars:
    #             p.release()  # 释放当前层的 CompressedParam
    #         torch.cuda.empty_cache()
    #     name2layer[last].register_full_backward_hook(_hook_post)   # release after calculation


    name2layer = {n: m for n, m in model.named_modules()}

    # # # 3) install prefetch hooks
    layer_names = [f"base_model.model.model.layers.{i}"
                    for i in range(model.config.num_hidden_layers)]

    install_fwd_prefetch_hooks(model, layer_names, layer2cps)
    install_bwd_prefetch_hooks(model, layer_names[::-1], layer2cps)

gc.collect()
process = psutil.Process(os.getpid())
cpu_mem_peak0 = process.memory_info().rss / 1024 / 1024  # 单位: MB
print(f"Peak CPU memory usage: {cpu_mem_peak0:.2f} MB")
print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated(device) / 1024 / 1024:.2f} MB")


# for name, param in model.named_parameters():
#     if not param.requires_grad:
#         print(f"[FROZEN] {name} shape={param.shape} requires_grad={param.requires_grad} is_leaf={param.is_leaf} shape={param.data.shape}")
#     elif "lora_" in name:
#         print(f"[LoRA]   {name} shape={param.shape} requires_grad={param.requires_grad} is_leaf={param.is_leaf} shape={param.shape}")
#     else:
#         print(f"[TRAIN]  {name} shape={param.shape} requires_grad={param.requires_grad} is_leaf={param.is_leaf} shape={param.shape}")


# exit()

# if not args.weight:
#     name2layer = {n: m for n, m in model.named_modules()}
#     def _hook_post(_, __, ___):
#         return
#     def _hook_pre(_, __):
#         return
#     for n in name2layer.keys(): 
#         name2layer[n].register_forward_pre_hook(_hook_pre)    # overlap current layer's computation with next layer's decompression
#         name2layer[n].register_forward_hook(_hook_post)   # release after calculation
        
#         name2layer[n].register_full_backward_hook(_hook_post)   # release after calculation
#         name2layer[n].register_full_backward_pre_hook(_hook_pre)    # overlap current layer's computation with next layer's decompression
        

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

    def __init__(self, stream: torch.cuda.Stream | None = None):
        # Dedicated stream for D→H copy + encode so we don't block the main
        # compute stream.  Feel free to expose this in your API.
        print("init AsyncCompressor")
        self.pool_workers = os.cpu_count() - 2

        # --- NEW: Add Semaphores ---
        # Limit concurrent compressions to control peak CPU/GPU buffer memory
        concurrency_limit = 3
        self.comp_semaphore = threading.Semaphore(value=self.pool_workers)
        self.decomp_semaphore = threading.Semaphore(value=concurrency_limit)

        self._build()

        if not args.asynchronous:
            self.cctx = zstd.ZstdCompressor(level=args.level, threads=-1, write_checksum=False)
            self.dctx = zstd.ZstdDecompressor()

    def _build(self):
        self.compress_pool= fut.ThreadPoolExecutor(self.pool_workers)
        self.decode_pool  = fut.ThreadPoolExecutor(2)
        self.d2h_stream   = torch.cuda.Stream()
        self.h2d_stream   = torch.cuda.Stream()
    
    def _reset(self):
        for pool in (self.compress_pool, self.decode_pool):
            pool.shutdown(wait=True)
        del self.compress_pool
        del self.decode_pool
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()

        self._build()          # 彻底重建所有资源
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

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
        return
    
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
                global comp_time, wait_comp_start
                start = time.time()
                evt.synchronize()
                # del tok.t
                wait_comp_start += time.time() - start

                arr = cpu_exp.numpy()
                cctx = get_cctx(algo, args.level)
                comped_bytes = cctx.compress(arr)
                # tok.comped_cpu_exp = comped_bytes
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
                # del evt
        
        # _encode(tok, cpu_exp, evt)
        fut = self.compress_pool.submit(_encode, cpu_exp, evt)
        # tok.future = fut

        return fut

    def decompress_sync(self, 
            tok: PlaceHolderToken,
            fut):

        # comped_bytes, numel = tok.future.result()
        comped_bytes, numel = fut.result()
        # comped_bytes = tok.comped_cpu_exp
        # numel = tok.numel
        cpu_exp = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
        with self.dctx.stream_reader(memoryview(comped_bytes)) as reader:
            view = memoryview(cpu_exp.numpy())   # numpy() 不复制，只拿 data_ptr
            nread = reader.readinto(view)
            assert nread == numel, "decompress size mismatch"

        sm_bits = tok.sm_bits
        # stream = torch.cuda.current_stream()
        stream = self.h2d_stream
        with torch.cuda.stream(stream):
            rst = fs_sp.merge(cpu_exp, sm_bits, tok.shape, tok.stride, tok.offset, tok.dtype, stream.cuda_stream)
        evt = stream.record_event()
        evt.synchronize()
        
        tok.decomped_data = rst

    def decompress_async(self, 
            tok: PlaceHolderToken,
            fut: fut.Future):

        # # cpu_exp, sm_bits = tok.future.result()  # 阻塞，等待压缩完成
        # cpu_exp = tok.cpu_exp
        # comped_bytes = tok.comped_cpu_exp
        # numel = tok.numel
        def _decode(tok, fut, algo="zstd"):
            self.decomp_semaphore.acquire() # Wait for a free slot
            comped_bytes, numel = None, None
            try:
                # print(f"waiting for compressed data")
                # comped_bytes, numel = tok.future.result()
                comped_bytes, numel = fut.result()

                cpu_exp = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
                dctx = get_dctx(algo)
                with dctx.stream_reader(memoryview(comped_bytes)) as reader:
                    view = memoryview(cpu_exp.numpy())   # numpy() 不复制，只拿 data_ptr
                    nread = reader.readinto(view)
                    assert nread == numel, "decompress size mismatch"
                # print(f"waiting for decompression")

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
    

def _unpack(tok, compressor):
    if args.weight:
        global lid
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
                
        if args.asynchronous:
            if tok.decomped_data is not None and tok.CtoD_copy_evt is None:
            # if tok.future is None and tok.CtoD_copy_evt is None:
            #     assert tok.decomped_data is not None
                return tok.decomped_data
            tok.ready_evt.wait()
            # tok.CtoD_copy_evt.synchronize()
            tok.decomped_data.record_stream(torch.cuda.current_stream())
            tok._clear_after_recover()
        else:
            if tok.decomped_data is None:
                # self.comp.decompress_sync(tok)
                compressor.decompress_sync(tok)
        return tok.decomped_data

    return tok                    # 反向时会被替换成 token，再解压


counter = {}
class DecoderLayerWrapper(torch.nn.Module):
    def __init__(self, layer: nn.Module, compressor: AsyncCompressor):
        super().__init__()
        self.layer = layer
        self.comp = compressor          # 你的 AsyncCompressor
        self.tokens: list[PlaceHolderToken] = []
        self.futures = []

    # ---------- forward ----------
    def forward(self, *inp, **kw):
        # tokens: list[PlaceHolderToken] = []
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
                    # if isinstance(cp, CompressedParam):
                        # print("ahaaha!!!!")
                        # print(f"{len(cp.exp_mv)=}, {cp.sm_gpu.shape=}")

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
            if is_lora_weight(t):
                return t

            # MEM_THRESHOLD_BYTES = 1 * 1024 * 1024  # 1 MB
            # if t.nbytes < MEM_THRESHOLD_BYTES:
            #     return t # For small tensors, do nothing and keep them on GPU.
            
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
                if args.asynchronous:
                    fut = self.comp.kickoff_async(tok, t)
                else:
                    self.comp.kickoff_sync(tok, t)
                self.tokens.append(weakref.ref(tok))
                self.futures.append(fut)
                tok.fut_id = len(self.futures) - 1
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

                # self.comp.kickoff(tok, t)

                if args.asynchronous:
                    self.comp.kickoff_async(tok, t)
                else:
                    self.comp.kickoff_sync(tok, t)
                del t
            
        else:
            # Impl.2
            # print(f"{len(ts)=}")
            for t_ptr in ts:
                t = t_ptr()
                if t is None:
                    continue
                del t
        # for i, tok_ptr in enumerate(self.tokens):
        #     tok = tok_ptr()
        #     tok.future.result()
        # for fut in self.futures:
        #     fut.result()

        del ts
        del seen
        torch.cuda.empty_cache()
        # gc.collect()
        torch._C._host_emptyCache()

        return out

    def decomp_tokens(self):
        print(f"{len(self.tokens)=}")
        for tok_ptr in self.tokens:
            tok = tok_ptr()
            if tok is None:
                continue
            fut = self.futures[tok.fut_id]
            compressor.decompress_async(tok, fut)
            self.futures[tok.fut_id] = None

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
        if args.asynchronous:
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

    return model

if args.hook:
    compressor = AsyncCompressor()
    inject_async_compression(model, compressor)
# if not args.hook:
#     def _hook_post(_, __, ___):
#         torch.cuda.empty_cache()
#     for n, m in model.named_modules():
#         m.register_forward_hook(_hook_post)
#         m.register_full_backward_hook(_hook_post)

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
# tokenizer.pad_token = tokenizer.unk_token
# tokenizer.pad_token_id =  tokenizer.unk_token_id
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
    # if args.weight:
    #     prefetch_first_layer(layer2cps, layer_names)
    
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
        torch._C._host_emptyCache()
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
                    # out = model(**inputs)
                    # loss = out[0] if isinstance(out, tuple) else out.loss
                    if "gemma" in args.model:
                        out = model(**inputs, return_dict=False)
                        loss = out[0] if isinstance(out, tuple) else out.loss
                    else:   
                        loss = model(**inputs).loss

            print(f"{loss=}")
            tf = time.time()

            if not EVAL:
                # if args.hook and args.asynchronous:
                #     model.model.model.layers[-1].decomp_tokens()
                loss.backward()
                # print(f"After backward")
                optimizer.step()
                # print(f"After step")
                optimizer.zero_grad()
                if args.hook:
                    compressor._reset()
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