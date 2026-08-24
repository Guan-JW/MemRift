import argparse
import math
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Local model path or Hugging Face model ID")
parser.add_argument("--checkpoint", help="Prepared MemRift compressed-weight directory")
parser.add_argument("--results-dir", default="/results/memrift")
parser.add_argument("--dataset-cache", default="/cache/huggingface")
parser.add_argument("--save-model-dir", default="/results/models/memrift")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--tegrastats-bin", default="/usr/bin/tegrastats")
parser.add_argument("--disable-tegrastats", action="store_true")
parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
parser.add_argument("--dataset", default="timdettmers/openassistant-guanaco", help="Dataset to use for training")
parser.add_argument(
    "--synthetic-data",
    action="store_true",
    help="Use deterministic local text examples instead of loading a dataset (smoke tests only)",
)
parser.add_argument("--finetune_type", choices=["full", "lora", "qlora"], default="lora", help="Type of finetuning")
parser.add_argument("--autocast_context", action="store_true", help="Set torch.amp.autocast")
parser.add_argument("--gradient_checkpointing", action="store_true", help="Run with gradient checkpointing")
parser.add_argument("--check_diff", action="store_true",
                    help="同时加载原始模型并比对差值（耗显存）")
parser.add_argument("--hook", action="store_true", help="Run with compression hooks")
parser.add_argument("--debug", action="store_true", help="Run with debug")
parser.add_argument("--print_ratio", action="store_true", help="Print activation's compression ratio")
parser.add_argument("--print_time", action="store_true", help="Print activation's compression ratio")
parser.add_argument("--weight", default=False, action="store_true", help="Switch on weight compression")
parser.add_argument("--weight_async", default=False, action="store_true", help="Switch on asynchronous weight decompression")
parser.add_argument(
    "--weight_async_concurrency",
    type=int,
    default=1,
    help="Maximum concurrent async weight materialization jobs.",
)
parser.add_argument(
    "--act_compact_concurrency",
    type=int,
    default=1,
    help="Maximum activation compact jobs admitted before split/encode.",
)
parser.add_argument(
    "--act_decode_concurrency",
    type=int,
    default=1,
    help="Maximum concurrent activation decode jobs.",
)
parser.add_argument(
    "--gc_keep_recompute_weights",
    default=False,
    action="store_true",
    help="With GC, keep weights materialized after recompute forward for the immediately following backward.",
)
parser.add_argument(
    "--gc_no_recompute_prefetch",
    default=False,
    action="store_true",
    help="With GC, do not async-prefetch next-layer weights during checkpoint recompute forwards.",
)
parser.add_argument("--weight_df11", default=False, action="store_true", help="Switch on asynchronous weight decompression")
parser.add_argument("--activation", default=False, action="store_true", help="Switch on activation compression")
parser.add_argument("--layerwise", default=False, action="store_true", help="Switch on layerwise activation compression")
parser.add_argument("--act_async", default=False, action="store_true", help="Switch on asynchronous activation (de)compression")
parser.add_argument("--attn", default=False, action="store_true", help="Switch on asynchronous activation (de)compression")
parser.add_argument("--mlp", default=True, action="store_true", help="Switch on asynchronous activation (de)compression")
parser.add_argument("--save_model", default=False, action="store_true", help="Switch on to save finetuned model weights")
parser.add_argument(
        "--level", type=int, default=1, help="Zstd compression level (-131072 through 22)"
    )
parser.add_argument(
        "--round", type=int, default=2, help="# training cycles"
    )
parser.add_argument(
        "--warmup_rounds", type=int, default=1, help="# initial rounds excluded from timing summary"
    )
parser.add_argument(
        "--max_length", type=int, default=2048, help="Input length"
    )
parser.add_argument(
        "--batch_size", type=int, default=1, help="Input batch size"
    )
parser.add_argument(
        "--tegrastats_interval_ms", type=int, default=500, help="tegrastats sampling interval"
    )
parser.add_argument(
        "--tegra-csv", dest="tegra_csv", default="", help="Optional raw tegrastats CSV filename under results-dir"
    )
args = parser.parse_args()
for name in ("round", "batch_size", "max_length", "weight_async_concurrency", "act_compact_concurrency", "act_decode_concurrency"):
    if getattr(args, name) < 1:
        parser.error(f"--{name} must be at least 1")
if args.warmup_rounds < 0 or args.warmup_rounds >= args.round:
    parser.error("--warmup_rounds must be non-negative and less than --round")
if re.fullmatch(r"cuda(?::\d+)?", args.device) is None:
    parser.error("--device must name a CUDA device, for example cuda:0")
if args.weight and not args.hook:
    parser.error("--weight requires --hook")
if args.weight_async and not args.weight:
    parser.error("--weight_async requires --weight")
if args.activation and not args.hook:
    parser.error("--activation requires --hook")
if args.act_async and not args.activation:
    parser.error("--act_async requires --activation")
if not -131072 <= args.level <= 22:
    parser.error("--level must be a valid Zstd compression level (-131072 through 22)")
if args.weight and not args.checkpoint:
    parser.error("--checkpoint is required when --weight is enabled")
if args.checkpoint:
    args.checkpoint = os.path.abspath(args.checkpoint)
if args.weight and not os.path.isfile(os.path.join(args.checkpoint, "index.json")):
    parser.error(f"compressed checkpoint index is missing: {os.path.join(args.checkpoint, 'index.json')}")
os.makedirs(args.results_dir, exist_ok=True)
if args.tegra_csv:
    if os.path.isabs(args.tegra_csv):
        parser.error("--tegra-csv must be a filename relative to --results-dir")
    results_root = os.path.realpath(args.results_dir)
    tegra_csv = os.path.realpath(os.path.join(results_root, args.tegra_csv))
    if os.path.commonpath((results_root, tegra_csv)) != results_root:
        parser.error("--tegra-csv must remain within --results-dir")
    args.tegra_csv = tegra_csv

import contextlib, weakref, psutil
import json, mmap, queue, threading, struct, gc, types, re, pprint, time, subprocess
import concurrent.futures as fut
import functools
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Union, List, Optional

try:
    import humanize
except ModuleNotFoundError:
    class humanize:
        @staticmethod
        def naturalsize(nbytes, binary=False):
            step = 1024 if binary else 1000
            units = ["Bytes", "KiB", "MiB", "GiB", "TiB"] if binary else ["Bytes", "KB", "MB", "GB", "TB"]
            size = float(nbytes)
            for unit in units:
                if abs(size) < step or unit == units[-1]:
                    return f"{size:.1f} {unit}" if unit != "Bytes" else f"{int(size)} Bytes"
                size /= step

import imagecodecs
import numpy as np
import pandas as pd
import torch
import bitsandbytes as bnb
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from torch.utils._pytree import tree_map
from torch.utils.cpp_extension import load
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from transformers import MistralModel, MistralForCausalLM, MistralConfig, LlamaModel, LlamaForCausalLM, LlamaConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from accelerate import init_empty_weights
from data_prep import data_prepare
import float_split_stride as fs_sp
import zstandard as zstd
import lz4.frame as lz4f, lz4.block as lz4b
import torch.cuda.nvtx as nvtx

if not torch.cuda.is_available():
    parser.error("--device requires CUDA, but CUDA is not available")
device = torch.device(args.device)
device_index = torch.cuda.current_device() if device.index is None else device.index
if device_index < 0 or device_index >= torch.cuda.device_count():
    parser.error(
        f"--device references CUDA device {device_index}, but only {torch.cuda.device_count()} device(s) are available"
    )

try:
    import wandb
except ModuleNotFoundError:
    if args.wandb_mode != "disabled":
        raise RuntimeError("wandb is required for --wandb-mode offline or online")

    class _DisabledWandb:
        summary = {}

        @staticmethod
        def init(**_kwargs):
            return None

        @staticmethod
        def define_metric(*_args, **_kwargs):
            return None

        @staticmethod
        def log(*_args, **_kwargs):
            return None

    wandb = _DisabledWandb()

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex
except (ImportError, OSError):
    nvmlInit = nvmlDeviceGetHandleByIndex = None
target_modules = []
if args.attn:
    target_modules += ["q_proj", "k_proj", "v_proj", "o_proj"]
if args.mlp:
    target_modules += ["gate_proj", "up_proj", "down_proj"]
if args.finetune_type in ["lora", "qlora"]:
    assert len(target_modules) > 0

PHASE_LOAD   = "weight_load_end"   # 权重加载完
PHASE_EPOCH  = "round_end"         # 每轮训练完

if nvmlInit is not None:
    try:
        nvmlInit()
        nvmlDeviceGetHandleByIndex(torch.device(args.device).index or 0)
    except Exception as exc:
        print(f"[MemRift] NVML unavailable: {exc}", file=sys.stderr, flush=True)
proj = ""
if "LongForm" in args.dataset:
    proj = "LongForm_"
if "oasst1" in args.dataset:
    proj = "oasst1_"
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
proj = f"{proj}memrift_act_weight_jetson"
if args.max_length != 2048:
    proj = f"{proj}_{args.max_length}"
if args.batch_size > 1:
    proj = f"{proj}_bs{args.batch_size}"
if args.gradient_checkpointing:
    proj = f"{proj}_gc"

if "Mistral-7B" in args.model:
    proj = f"Mistral-7B_{proj}"
elif "Llama-3.1-8B" in args.model:
    proj = f"Llama-3.1-8B_{proj}"
elif "gemma-2-2b-it" in args.model:
    proj = f"gemma-2-2b-it_{proj}"
elif "Llama-3.2-3B-Instruct" in args.model:
    proj = f"Llama-3.2-3B-Instruct_{proj}"
elif 'gpt2' in args.model:
    proj = f"gpt2_{proj}"

wandb.init(project=proj, config=vars(args), group="group1", mode=args.wandb_mode, dir=args.results_dir)
wandb.define_metric("timestamp")          # 用绝对时间
wandb.define_metric("*", step_metric="timestamp")

T0 = time.time()
tegra_stats = {
    "ram_used_MB_max": None,
    "process_rss_MB_max": 0,
    "process_rss_bytes_max": 0,
    "system_available_MB_min": None,
    "system_available_bytes_min": None,
    "cpu_util_sum": 0.0,
    "gpu_util_sum": 0.0,
    "samples": 0,
}
tegra_stats_lock = threading.Lock()
# ──────────────────────────────────────────
# 2.  采样线程：tegrastats + torch
PAT = re.compile(
    r'RAM\s+(\d+)/(\d+)MB'           # RAM used / total
    r'.*?CPU\s+\[([^\]]+)\]'         # 整个  [...]  片段
    r'.*?GR3D_FREQ\s+(\d+)%',        # GPU util %
    re.I)


def tegra_loop():
    cmd = [args.tegrastats_bin, '--interval', str(args.tegrastats_interval_ms)]
    csv_f = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                text=True, bufsize=1)
        if args.tegra_csv:
            os.makedirs(os.path.dirname(args.tegra_csv) or ".", exist_ok=True)
            csv_f = open(args.tegra_csv, "w", buffering=1)
            csv_f.write("timestamp_ms,ram_used_MB,ram_total_MB,cpu_util,gpu_util,gpu_alloc_MB,gpu_reserved_MB,cpu_proc_MB,raw\n")
    except (FileNotFoundError, PermissionError, OSError) as exc:
        print(f"[MemRift] tegrastats unavailable: {exc}", file=sys.stderr, flush=True)
        return

    for line in proc.stdout:
        m = PAT.search(line)
        # print(f"{m=}")
        if not m:
            continue                          # 依旧安全兜底

        ram_u, ram_t, cpu_blk, gpu_u = m.groups()

        # ① CPU 方括号里提取所有 “99%@freq”
        cpu_vals = [int(x) for x in re.findall(r'(\d+)%@', cpu_blk)]
        cpu_avg  = sum(cpu_vals) / len(cpu_vals) if cpu_vals else 0
        with tegra_stats_lock:
            previous_ram = tegra_stats["ram_used_MB_max"]
            tegra_stats["ram_used_MB_max"] = int(ram_u) if previous_ram is None else max(previous_ram, int(ram_u))
            process_rss_bytes = psutil.Process().memory_info().rss
            available_bytes = psutil.virtual_memory().available
            process_rss_MB = process_rss_bytes // 2**20
            available_MB = available_bytes // 2**20
            tegra_stats["process_rss_MB_max"] = max(tegra_stats["process_rss_MB_max"], process_rss_MB)
            tegra_stats["process_rss_bytes_max"] = max(tegra_stats["process_rss_bytes_max"], process_rss_bytes)
            current_min = tegra_stats["system_available_MB_min"]
            tegra_stats["system_available_MB_min"] = available_MB if current_min is None else min(current_min, available_MB)
            current_bytes_min = tegra_stats["system_available_bytes_min"]
            tegra_stats["system_available_bytes_min"] = available_bytes if current_bytes_min is None else min(current_bytes_min, available_bytes)
            tegra_stats["cpu_util_sum"] += cpu_avg
            tegra_stats["gpu_util_sum"] += int(gpu_u)
            tegra_stats["samples"] += 1

        ts_ms = int((time.time()-T0)*1000)
        gpu_alloc_MB = torch.cuda.memory_allocated() // 2**20
        gpu_reserved_MB = torch.cuda.memory_reserved() // 2**20 or -1
        cpu_proc_MB = process_rss_MB
        if csv_f is not None:
            raw = line.strip().replace('"', '""')
            csv_f.write(f'{ts_ms},{int(ram_u)},{int(ram_t)},{cpu_avg},{int(gpu_u)},{gpu_alloc_MB},{gpu_reserved_MB},{cpu_proc_MB},"{raw}"\n')

        wandb.log({
            "timestamp": ts_ms,   # ms step
            "ram_used_MB": int(ram_u),
            "ram_total_MB": int(ram_t),
            "cpu_util":  cpu_avg,                      # 平均 %
            "gpu_util":  int(gpu_u),
            "gpu_alloc_MB": gpu_alloc_MB,
            "gpu_reserved_MB": gpu_reserved_MB,
            "cpu_proc_MB": cpu_proc_MB,
        }, commit=True)
    proc.stdout.close()
    if csv_f is not None:
        csv_f.close()


if not args.disable_tegrastats:
    threading.Thread(target=tegra_loop, daemon=True).start()

print(f"\n\n{args.model=}, {args.checkpoint=}, {args.weight=}, {args.activation=}, {args.max_length=}")
rid = 0

# Set GPU!
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
gradient_checkpointing = args.gradient_checkpointing
_PTR2CP = weakref.WeakValueDictionary()
# fromPT = False
# if "gpt2" in args.model:
#     fromPT = True

if args.hook and args.weight:
    # 0. 仅用 config 起“空架”，避免 from_pretrained 拉满权重进显存
    cfg    = AutoConfig.from_pretrained(args.model, cache_dir=args.dataset_cache)
    with init_empty_weights():
        # if "gemma" in args.model:
        #     model  = AutoModelForCausalLM.from_config(cfg, torch_dtype=MODEL_TYPE, attn_implementation='eager')
        # else:
        model  = AutoModelForCausalLM.from_config(cfg, torch_dtype=MODEL_TYPE)
elif args.weight_df11:
    from dfloat11 import DFloat11Model
    model = DFloat11Model.from_pretrained(args.model, device_map="auto")
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
                device_map={"": args.device},
                cache_dir=args.dataset_cache,
        )
        model = prepare_model_for_kbit_training(model,
                use_gradient_checkpointing=gradient_checkpointing)
    else:
        # if "gemma" in args.model:
        #     model = AutoModelForCausalLM.from_pretrained(args.model,
        #                     torch_dtype=MODEL_TYPE, device_map={"": 0}, attn_implementation='eager')
        # else:
        model = AutoModelForCausalLM.from_pretrained(args.model,
                        torch_dtype=MODEL_TYPE, device_map={"": args.device}, cache_dir=args.dataset_cache)

GC_BOUNDARY_HIDDEN_SIZE = getattr(model.config, "hidden_size", None)
GC_BOUNDARY_ONLY_ACT = bool(args.gradient_checkpointing and args.activation)
if GC_BOUNDARY_ONLY_ACT:
    print(
        "[MemRift] gradient checkpointing is enabled; compressing only "
        "decoder-boundary hidden states for activation compression.",
        flush=True,
    )

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
        self._materialize_error = None
        self.CtoD_evt   = None
        self._hooked = False
        self._materialize_started = False

        # if args.debug:
        #     self.parent, self.attr = parent, attr
        #     self.layer_id = layer_id

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
        global weight_materialize_sync_count
        weight_materialize_sync_count += 1
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
        self._materialize_error = None
        self._materialize_started = True
        try:
            if self._bf16 is not None:
                return self._bf16   # double-checked

            # CPU 解压
            numel = int(np.prod(self.orig_shape))

            # 1) 解压 exponent → pinned uint8
            # self._exp_host = fs_sp.acquire_pin(numel, torch.uint8)
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
            # if args.debug:
            #     print(f"Setted event! layer={self.layer_id}: {self.parent}-{self.attr}")

            if sync:                       # ------- 同步路径 (漏预取) -------
                self.CtoD_evt.synchronize()

                # return self._bf16
        except BaseException as e:
            self._materialize_error = e
            self._materialize_started = False
            import traceback, sys
            traceback.print_exc(file=sys.stderr)
            raise                # 重新抛给上层，让你能看到
        finally:
            self._ready_event.set()

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
            # fs_sp.release_pin(self._exp_host)
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

        global weight_release_count
        weight_release_count += 1

        if delref:  # called after backward, 防止之后重用空间的tensor 又识别到 cp
            del _PTR2CP[self._bf16.data_ptr()]

        fs_sp.release_cuda(self._bf16)
        self._bf16 = None

        self.data = torch.empty(0, dtype=torch.bfloat16, device=self.sm_gpu.device)
        self._ready_event.clear()
        self._materialize_error = None
        self._hooked = False
        self.CtoD_evt = None
        self._materialize_started = False

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
    decomp_started: bool = False
    decomp_error: BaseException | None = None
    # CtoD_copy_evt: torch.cuda.Event | None = None

    def _clear_after_recover(self):
        # if hasattr(self, "cpu_exp"):
        #     del self.cpu_exp
        # if hasattr(self, "sm_bits"):
        #     del self.sm_bits
        if hasattr(self, "fut_id"):
            del self.fut_id
        if hasattr(self, "future"):
            del self.future
        # self.future = None
        self.ready_evt.clear()
        if hasattr(self, "CtoD_copy_evt"):
            del self.CtoD_copy_evt
        self.CtoD_copy_evt = None

decomp_time = comp_time = 0
wait_comp_done = wait_comp_start = 0
orig_len = new_len = 0
act_pack_count = act_pack_bytes = 0
act_pack_boundary_count = act_pack_boundary_bytes = 0
weight_materialize_sync_count = 0
weight_materialize_async_count = 0
weight_release_count = 0
c3_lock = threading.Lock()
c3_stats = defaultdict(int)

def _c3_add_current(name, delta):
    with c3_lock:
        cur_key = f"{name}_current_bytes"
        max_key = f"{name}_max_bytes"
        c3_stats[cur_key] += int(delta)
        if c3_stats[cur_key] > c3_stats[max_key]:
            c3_stats[max_key] = c3_stats[cur_key]

def _c3_dec_current(name, delta):
    with c3_lock:
        cur_key = f"{name}_current_bytes"
        c3_stats[cur_key] = max(0, c3_stats[cur_key] - int(delta))

def _c3_inc(name, delta=1):
    with c3_lock:
        c3_stats[name] += int(delta)

def _c3_inc_current(name, delta=1):
    with c3_lock:
        cur_key = f"{name}_current"
        max_key = f"{name}_max"
        c3_stats[cur_key] += int(delta)
        if c3_stats[cur_key] > c3_stats[max_key]:
            c3_stats[max_key] = c3_stats[cur_key]

def _c3_max(name, value):
    with c3_lock:
        c3_stats[name] = max(c3_stats[name], int(value))

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
        # print("init AsyncCompressor")

        self._async_errors = []
        self._async_errors_lock = threading.Lock()
        if not args.act_async:
            self.cctx = zstd.ZstdCompressor(level=args.level, threads=-1, write_checksum=False)
            self.dctx = zstd.ZstdDecompressor()
        if args.act_async:
            self.act_compact_semaphore = threading.Semaphore(args.act_compact_concurrency)
            self.act_decode_semaphore = threading.Semaphore(args.act_decode_concurrency)
        if args.weight_async:
            self.weight_materialize_semaphore = threading.Semaphore(args.weight_async_concurrency)

        self._build()

    def _report_async_error(self, error):
        with self._async_errors_lock:
            self._async_errors.append(error)

    def _watch_future(self, future):
        def record_failure(done):
            if not done.cancelled() and done.exception() is not None:
                self._report_async_error(done.exception())
        future.add_done_callback(record_failure)
        return future

    def _build(self):
        if args.act_async:
            self.compress_pool = fut.ThreadPoolExecutor(args.act_compact_concurrency, thread_name_prefix="memrift-act-compact")
            self.act_decode_pool = fut.ThreadPoolExecutor(args.act_decode_concurrency, thread_name_prefix="memrift-act-decode")
        if args.weight_async:
            self.weight_materialize_pool = fut.ThreadPoolExecutor(args.weight_async_concurrency, thread_name_prefix="memrift-weight")
        self.d2h_stream   = torch.cuda.Stream()
        self.h2d_stream   = torch.cuda.Stream()

    def _reset(self):
        for name in ("compress_pool", "act_decode_pool", "weight_materialize_pool"):
            pool = getattr(self, name, None)
            if pool is not None:
                pool.shutdown(wait=True)
                delattr(self, name)
        with self._async_errors_lock:
            errors = self._async_errors[:]
            self._async_errors.clear()

        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        # torch._C._host_emptyCache()

        self._build()          # 彻底重建所有资源
        if errors:
            raise RuntimeError("asynchronous MemRift worker failed") from errors[0]
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


        self.act_compact_semaphore.acquire()
        nvtx.range_push("memrift_act_split")
        try:
            self.d2h_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.d2h_stream):
                cpu_exp, sm_bits = fs_sp.split(t, self.d2h_stream.cuda_stream)
                evt = self.d2h_stream.record_event()   # 拷贝结束事件
                t.record_stream(self.d2h_stream)
        except Exception:
            self.act_compact_semaphore.release()
            raise
        finally:
            nvtx.range_pop()
        # evt.synchronize()
        tok.sm_bits = sm_bits
        # tok.t = t
        compact_bytes = int(t.nbytes + sm_bits.nbytes + cpu_exp.nbytes)
        _c3_add_current("act_compact_transition", compact_bytes)
        _c3_inc_current("act_compact_jobs")

        def _encode(cpu_exp, evt, algo="zstd"):
            # self.comp_semaphore.acquire() # Wait for a free slot
            nvtx.range_push("memrift_act_encode")
            arr = None
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
                nvtx.range_pop()
                _c3_dec_current("act_compact_transition", compact_bytes)
                _c3_inc_current("act_compact_jobs", -1)
                self.act_compact_semaphore.release() # Release the slot for others
                del cpu_exp
                # fs_sp.release_pin(cpu_exp)
                if arr is not None:
                    del arr
                del evt

        # _encode(tok, cpu_exp, evt)
        try:
            fut = self._watch_future(self.compress_pool.submit(_encode, cpu_exp, evt))
        except BaseException:
            _c3_dec_current("act_compact_transition", compact_bytes)
            _c3_inc_current("act_compact_jobs", -1)
            self.act_compact_semaphore.release()
            fs_sp.release_cuda(sm_bits)
            raise

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
            comped_bytes, numel, decode_bytes, cpu_exp = None, None, 0, None
            range_pushed = semaphore_acquired = False
            try:
                nvtx.range_push("memrift_act_decode")
                range_pushed = True
                self.act_decode_semaphore.acquire()
                semaphore_acquired = True
                comped_bytes, numel = fut.result()

                cpu_exp = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
                out_bytes = int(np.prod(tok.shape)) * torch.tensor([], dtype=tok.dtype).element_size()
                decode_bytes = int(len(comped_bytes) + cpu_exp.nbytes + tok.sm_bits.nbytes + out_bytes)
                _c3_add_current("act_decode_transition", decode_bytes)
                _c3_inc_current("act_decode_jobs")
                # cpu_exp = fs_sp.acquire_pin(numel, torch.uint8)
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
                tok.decomped_data = rst

                # del tok.future
                # tok.future = None
                # del fut
            except BaseException as e:
                tok.decomp_error = e
                self._report_async_error(e)
                import traceback, sys
                traceback.print_exc(file=sys.stderr)
                raise
            finally:
                try:
                    if range_pushed:
                        nvtx.range_pop()
                    if semaphore_acquired:
                        self.act_decode_semaphore.release()
                    if decode_bytes:
                        _c3_dec_current("act_decode_transition", decode_bytes)
                        _c3_inc_current("act_decode_jobs", -1)
                    fs_sp.release_cuda(tok.sm_bits)
                except BaseException as cleanup_error:
                    if tok.decomp_error is None:
                        tok.decomp_error = cleanup_error
                    self._report_async_error(cleanup_error)
                    raise
                finally:
                    tok.ready_evt.set()

        # _decode(tok)
        try:
            tok.decode_future = self.act_decode_pool.submit(_decode, tok, fut)
        except BaseException as exc:
            tok.decomp_error = exc
            self._report_async_error(exc)
            tok.ready_evt.set()
            raise
        self._watch_future(tok.decode_future)

    def materialize_async(self,
                            cp: CompressedParam,
                            sync: bool =True):
        global weight_materialize_async_count
        weight_materialize_async_count += 1
        if cp._bf16 is not None:
            return cp._bf16   # double-checked
        if cp._materialize_started:
            return None
        cp._materialize_error = None
        cp._materialize_started = True

        def c_contiguous_strides(shape):
            strides = [1] * len(shape)
            running = 1
            for i in range(len(shape) - 2, -1, -1):   # 从倒数第二维向前推
                running *= shape[i + 1]
                strides[i] = running
            return tuple(strides)

        def _materialize(cp, sync, algo="zstd"):
            mat_bytes = 0
            ev = None
            range_pushed = semaphore_acquired = False
            try:
                nvtx.range_push("memrift_weight_materialize")
                range_pushed = True
                self.weight_materialize_semaphore.acquire()
                semaphore_acquired = True
                # CPU 解压
                numel = int(np.prod(cp.orig_shape))

                # 1) 解压 exponent → pinned uint8
                dctx = get_dctx(algo)
                cp._exp_host = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
                bf16_bytes = numel * torch.tensor([], dtype=MODEL_TYPE).element_size()
                mat_bytes = int(len(cp.exp_mv) + cp._exp_host.nbytes + cp.sm_gpu.nbytes + bf16_bytes)
                _c3_add_current("weight_materialize_transition", mat_bytes)
                _c3_inc_current("weight_materialize_jobs")
                # cp._exp_host = fs_sp.acquire_pin(numel, torch.uint8)
                with dctx.stream_reader(memoryview(cp.exp_mv)) as reader:
                    view = memoryview(cp._exp_host.numpy())   # numpy() 不复制，只拿 data_ptr
                    nread = reader.readinto(view)
                    assert nread == numel, "decompress size mismatch"

                # 2) 异步 H2D copy
                stride = c_contiguous_strides(cp.orig_shape)
                with torch.cuda.stream(self.h2d_stream):
                    cp._bf16 = fs_sp.merge(cp._exp_host, cp.sm_gpu, cp.orig_shape, stride, 0, MODEL_TYPE, self.h2d_stream.cuda_stream)
                ev = self.h2d_stream.record_event()

                cp.CtoD_evt = ev
                if sync:                       # ------- 同步路径 (漏预取) -------
                    cp.CtoD_evt.synchronize()

            except BaseException as e:
                cp._materialize_error = e
                self._report_async_error(e)
                cp._materialize_started = False
                import traceback, sys
                traceback.print_exc(file=sys.stderr)
                raise
            finally:
                try:
                    if range_pushed:
                        nvtx.range_pop()
                    if semaphore_acquired:
                        self.weight_materialize_semaphore.release()
                    if mat_bytes:
                        _c3_dec_current("weight_materialize_transition", mat_bytes)
                        _c3_inc_current("weight_materialize_jobs", -1)
                    if sync and hasattr(cp, "_exp_host"):
                        del cp._exp_host
                except BaseException as cleanup_error:
                    if cp._materialize_error is None:
                        cp._materialize_error = cleanup_error
                    self._report_async_error(cleanup_error)
                    raise
                finally:
                    cp._ready_event.set()

        # _materialize(cp, sync)
        # self.materialize_pool.submit(_materialize, cp, sync)
        try:
            cp._materialize_future = self.weight_materialize_pool.submit(_materialize, cp, sync)
        except BaseException as exc:
            cp._materialize_error = exc
            self._report_async_error(exc)
            cp._materialize_started = False
            cp._ready_event.set()
            raise
        self._watch_future(cp._materialize_future)

    def ensure_materialize_async(self, cp: CompressedParam):
        if cp._bf16 is None and not cp._ready_event.is_set():
            self.materialize_async(cp, sync=False)

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
            if tok.decomped_data is not None and getattr(tok, "CtoD_copy_evt", None) is None:
            # if tok.future is None and tok.CtoD_copy_evt is None:
            #     assert tok.decomped_data is not None
                return tok.decomped_data
            # Gradient checkpointing recomputes the layer forward inside backward.
            # Tokens created during recomputation are unpacked after the layer's
            # backward pre-hook has already fired, so schedule decompression here.
            if not tok.decomp_started:
                fut_obj = getattr(tok, "future", None)
                if fut_obj is not None:
                    tok.decomp_started = True
                    compressor.decompress_async(tok, fut_obj)
            tok.ready_evt.wait()
            if tok.decomp_error is not None:
                raise RuntimeError("asynchronous activation decode failed") from tok.decomp_error
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

def _wait_materialized(cp: CompressedParam):
    cp._ready_event.wait()
    if cp._materialize_error is not None:
        raise RuntimeError("asynchronous weight materialization failed") from cp._materialize_error


def _checkpoint_entries(comp_dir, *, legacy=False):
    index_path = os.path.join(comp_dir, "index.json")
    with open(index_path) as index_file:
        idx = json.load(index_file)
    if not isinstance(idx, list):
        raise ValueError(f"checkpoint index must contain a list: {index_path}")

    root = os.path.realpath(comp_dir)
    validated = []
    for entry_number, it in enumerate(idx):
        where = f"{index_path} entry {entry_number}"
        if not isinstance(it, dict):
            raise ValueError(f"{where} must be an object")
        for field_name in ("name", "file", "shape"):
            if field_name not in it:
                raise ValueError(f"{where} is missing {field_name!r}")
        if not isinstance(it["name"], str) or not it["name"].rpartition(".")[0]:
            raise ValueError(f"{where} has an invalid parameter name")
        if not isinstance(it["file"], str) or not it["file"] or os.path.isabs(it["file"]):
            raise ValueError(f"{where} has an invalid relative file path")
        shape = it["shape"]
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) > 4
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape)
        ):
            raise ValueError(f"{where} has an invalid shape (at most four non-negative dimensions are supported)")
        scheme = it.get("scheme", "legacy_split_zstd" if legacy else None)
        allowed_schemes = {"split_zstd", "legacy_split_zstd"} if legacy else {"split_zstd", "raw_torch"}
        if scheme not in allowed_schemes:
            raise ValueError(f"{where} uses unsupported checkpoint scheme {scheme!r}")
        if not legacy and scheme == "split_zstd" and it.get("dtype") not in ("bfloat16", "float32"):
            raise ValueError(f"{where} has unsupported split_zstd dtype {it.get('dtype')!r}")

        file_path = os.path.realpath(os.path.join(root, it["file"]))
        if os.path.commonpath((root, file_path)) != root:
            raise ValueError(f"{where} references a file outside the checkpoint directory")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"checkpoint file listed by index is missing: {file_path}")
        validated.append((it, file_path, tuple(shape), scheme))
    return validated


def _validate_checkpoint_target(modules, module, attr, shape, where):
    if module not in modules:
        raise ValueError(f"{where} references unknown module {module!r}")
    mod = modules[module]
    if attr not in mod._parameters:
        raise ValueError(f"{where} references unknown parameter {module}.{attr}")
    expected = mod._parameters[attr]
    if expected is not None and tuple(expected.shape) != shape:
        raise ValueError(
            f"{where} shape {shape} does not match model parameter shape {tuple(expected.shape)}"
        )
    return mod

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

            if GC_BOUNDARY_ONLY_ACT:
                # With gradient checkpointing, internal tensors are recomputed.
                # Compress only layer-boundary hidden states retained by GC.
                if (
                    GC_BOUNDARY_HIDDEN_SIZE is None
                    or t.dim() != 3
                    or t.shape[-1] != GC_BOUNDARY_HIDDEN_SIZE
                ):
                    return t
                global act_pack_boundary_count, act_pack_boundary_bytes
                act_pack_boundary_count += 1
                act_pack_boundary_bytes += t.nbytes

            global act_pack_count, act_pack_bytes
            act_pack_count += 1
            act_pack_bytes += t.nbytes

            # MEM_THRESHOLD_BYTES = 1 * 1024 * 1024  # 1 MB
            # if t.nbytes < MEM_THRESHOLD_BYTES:
            #     return t # For small tensors, do nothing and keep them on GPU.

            key = (t.data_ptr(), t.nbytes)

            if key in seen:
                tok_ref, t_ref = seen[key]
                if t_ref() is not None:
                    # if args.debug:
                    #     return tok_ref(), t
                    return tok_ref()
            tok = PlaceHolderToken(t.dtype, t.shape, t.stride(), t.storage_offset())
            seen[key] = (weakref.ref(tok), weakref.ref(t))
            if args.act_async:
                fut = self.comp.kickoff_async(tok, t)
                tok.future = fut
                self.futures.append(fut)
                tok.fut_id = len(self.futures) - 1
            else:
                self.comp.kickoff_sync(tok, t)
            self.tokens.append(weakref.ref(tok))
            ts.append(weakref.ref(t))

            # if args.debug:
            #     return tok, t                      # 给 autograd 的就是它
            return tok

        unpack_fn = functools.partial(_unpack, compressor=self.comp)
        with torch.autograd.graph.saved_tensors_hooks(_pack, unpack_fn):
            out = self.layer(*inp, **kw)      # 真正计算

        # -------- 2) layer 结束：开始异步 D→H + 压缩 -------

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
                self.layer2cps, max_job_size = self.inject_from_files_old(args.checkpoint)
            else:
                self.layer2cps, max_job_size = self.inject_from_files(args.checkpoint)
            self._report_meta()

            self._EMPTY_STEP = 55
            if not args.act_async:
                self._EMPTY_STEP = 5
            self.bwd_counter = 0

        if args.finetune_type in ["lora", "qlora"]:
            peft_config = LoraConfig(
                        lora_alpha=16,
                        lora_dropout=0.0,
                        r=16,
                        bias="none",
                        task_type="CAUSAL_LM",
                        target_modules=target_modules
                )
            self.model = get_peft_model(self.model, peft_config,
                                autocast_adapter_dtype=True)   # set this to keep the adapters in bfloat16
        torch.cuda.empty_cache()
        gc.collect()

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
        if hasattr(container, "transformer"):
            container = container.transformer
        # print(f"type(model) = {type(model)}")       # LlamaForCausalLM
        # print(f"type(model.model) = {type(model.model)}")  # LlamaModel
        # print(f"type(container) = {type(container)}")       # 期望也是 LlamaModel

        if "gpt2" in args.model:
            layers = container.h
        else:
            layers = container.layers
        for i, layer in enumerate(layers):
            do_empty = False
            if i % 10 == 1:
                do_empty = True
            layers[i] = DecoderLayerWrapper(layer, compressor, do_empty)
            if args.activation and args.act_async:
                if i >= 0:
                    def _bwd_pre_hook(_, __, layer=layers[i], compressor=compressor):
                        for tok_ptr in layer.tokens[::-1]:
                            tok = tok_ptr()
                            if tok is None:
                                continue
                            if tok.decomp_started:
                                continue
                            fut = layer.futures[tok.fut_id]
                            tok.decomp_started = True
                            compressor.decompress_async(tok, fut)
                            layer.futures[tok.fut_id] = None
                    layers[i].register_full_backward_pre_hook(_bwd_pre_hook)
                def _bwd_hook(_, __, ___, layer=layers[i]):
                    for tok_ptr in layer.tokens:
                        tok = tok_ptr()
                        if tok is not None:
                            del tok
                    layer.tokens.clear()
                    layer.futures.clear()
                    # torch.cuda.empty_cache()
                    # torch._C._host_emptyCache()
                    # gc.collect()
                layers[i].register_full_backward_hook(_bwd_hook)

    def inject_from_files(self, comp_dir):
        layer2cps = {}  # {"base_model.model.model.layers.3": [cp1, cp2, ...]}
        max_job_size = 0
        modules = dict(self.model.named_modules())
        entries = _checkpoint_entries(comp_dir)
        for entry_number, (it, _, shape, _) in enumerate(entries):
            module, _, attr = it["name"].rpartition(".")
            _validate_checkpoint_target(modules, module, attr, shape, f"checkpoint index entry {entry_number}")

        for entry_number, (it, file_path, shape, scheme) in enumerate(entries):
            module, _, attr = it["name"].rpartition(".")
            mod = modules[module]

            if scheme == "split_zstd":
                with open(file_path, "rb") as f:
                    header = f.read(8)
                    if len(header) != 8:
                        raise ValueError(f"checkpoint file has a truncated header: {file_path}")
                    numel, = struct.unpack("<Q", header)
                    expected_numel = math.prod(shape)
                    if numel != expected_numel:
                        raise ValueError(
                            f"checkpoint file element count {numel} does not match index shape {shape}: {file_path}"
                        )
                    sm_size = numel * (1 if it["dtype"] == "bfloat16" else 3)
                    sm_bytes = f.read(sm_size)
                    if len(sm_bytes) != sm_size:
                        raise ValueError(f"checkpoint file has truncated sign/mantissa data: {file_path}")
                    sm     = np.frombuffer(sm_bytes, dtype=np.uint8)
                    exp    = f.read()
                    if not exp:
                        raise ValueError(f"checkpoint file has no compressed exponent data: {file_path}")

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
                cp = CompressedParam(shape, sm_gpu, exp, parent=mod, attr=attr, layer_id=layer_id, typ=dtype)

                if layer_name is None:
                    print(f"Materializing {module}")
                    cp.materialize(sync=True)
                    _wait_materialized(cp)
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

            elif scheme == "raw_torch":
                print(f"raw_torch")
                raw = torch.load(file_path, map_location=device)
                if not isinstance(raw, torch.Tensor):
                    raise ValueError(f"raw_torch checkpoint is not a tensor: {file_path}")
                if tuple(raw.shape) != shape:
                    raise ValueError(
                        f"raw_torch tensor shape {tuple(raw.shape)} does not match index shape {shape}: {file_path}"
                    )
                if attr in mod._parameters:
                    del mod._parameters[attr]
                    mod._parameters[attr] = torch.nn.Parameter(raw, requires_grad=False)
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

        modules = dict(self.model.named_modules())
        entries = _checkpoint_entries(comp_dir, legacy=True)
        prepared_entries = []
        for entry_number, (it, file_path, shape, scheme) in enumerate(entries):
            module, _, attr = it["name"].rpartition(".")
            module = rm_prefixes(module)
            mod = _validate_checkpoint_target(
                modules, module, attr, shape, f"checkpoint index entry {entry_number}"
            )
            with open(file_path, "rb") as f:
                header = f.read(4)
            if len(header) != 4:
                raise ValueError(f"legacy checkpoint file has a truncated header: {file_path}")
            numel, = struct.unpack("<I", header)
            expected_numel = math.prod(shape)
            if numel != expected_numel:
                raise ValueError(
                    f"legacy checkpoint element count {numel} does not match index shape {shape}: {file_path}"
                )
            if os.path.getsize(file_path) <= 4 + numel:
                raise ValueError(f"legacy checkpoint file is truncated or has no exponent data: {file_path}")
            prepared_entries.append((it, file_path, shape, module, attr, mod))

        for it, file_path, shape, module, attr, mod in prepared_entries:
            # --- 打开文件并一次性读到内存 ---------------------------------
            with open(file_path, "rb") as f:
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
            cp = CompressedParam(shape, sm_gpu, exp_bytes, parent=mod, attr=attr, layer_id=layer_id, typ=torch.bfloat16)
            # print(f"{layer_name=}")

            if layer_name is None:
                print(f"Materializing {module}")
                cp.materialize(sync=True)
                _wait_materialized(cp)
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

        return layer2cps, max_job_size

    def install_fwd_prefetch_hooks(self):
        name2layer = {n: m for n, m in self.model.named_modules()}
        for cur, nxt in zip(self.layer_names[:-1], self.layer_names[1:]):
            nxt_pars = self.layer2cps.get(nxt, [])
            cur_pars = self.layer2cps.get(cur, [])

            def _hook_post(_, __, ___, cur_pars=cur_pars):
                if (
                    args.gradient_checkpointing
                    and args.gc_keep_recompute_weights
                    and torch.is_grad_enabled()
                ):
                    return
                for p in cur_pars:
                    p.release()  # 释放当前层的 CompressedParam
                    # p.release_comp()
                # torch.cuda.empty_cache()
            def _hook_pre(_, __, nxt_pars=nxt_pars, cur_pars=cur_pars, cur=cur):
                # print(f"Fwd hook pre {cur}")
                is_gc_recompute = args.gradient_checkpointing and torch.is_grad_enabled()
                if args.weight_async and not (args.gc_no_recompute_prefetch and is_gc_recompute):
                    for cp in nxt_pars:
                        self.compressor.materialize_async(cp, sync=False)
                for cp in cur_pars:
                    if not args.weight_async:
                        cp.materialize(sync=True)
                    else:
                        self.compressor.ensure_materialize_async(cp)
                    _wait_materialized(cp)
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
                else:
                    self.compressor.ensure_materialize_async(cp)
                _wait_materialized(cp)
                cp.CtoD_evt.synchronize()
                cp.set_param()                  # 替换成 Tensor
        def _last_post(_, __, ___, last_pars=last_pars):
            if (
                args.gradient_checkpointing
                and args.gc_keep_recompute_weights
                and torch.is_grad_enabled()
            ):
                return
            for p in last_pars:
                p.release()  # 释放当前层的 CompressedParam
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
                self.bwd_counter += 1
                if self.bwd_counter % self._EMPTY_STEP == 0:
                    torch.cuda.empty_cache()
                    # torch._C._host_emptyCache()
            def _hook_pre(_, __, nxt_pars=nxt_pars, cur_pars=cur_pars, cur=cur):
                # print(f"Bwd hook pre {cur}")
                if args.weight_async:
                    for cp in nxt_pars:
                        self.compressor.materialize_async(cp, sync=False)
                for cp in cur_pars:
                    if not args.weight_async:
                        cp.materialize(sync=True)
                    else:
                        self.compressor.ensure_materialize_async(cp)
                    _wait_materialized(cp)
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
                else:
                    self.compressor.ensure_materialize_async(cp)
                _wait_materialized(cp)
                cp.CtoD_evt.synchronize()
                cp.set_param()                  # 替换成 Tensor
                # cp.sync_childs()
        def _last_post(_, __, ___, last_pars=last_pars):
            for p in last_pars:
                p.release()  # 释放当前层的 CompressedParam
            # torch.cuda.empty_cache()
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
                    target_modules=target_modules
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
cpu_mem_at_load = process.memory_info().rss / 1024 / 1024
print(f"Process RSS after model load: {cpu_mem_at_load:.2f} MB")
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

#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, cache_dir=args.dataset_cache)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
tokenizer.padding_side = 'left'


if args.synthetic_data:
    synthetic_base = (
        "### Instruction:\nSummarize deterministic systems behavior.\n\n"
        "### Response:\nMemory compression preserves tensor values while reducing transfer volume. "
    )
    longest_texts = [
        (synthetic_base + f"Synthetic example {i}. ") * max(2, args.max_length // 16)
        for i in range(args.batch_size)
    ]
else:
    longest_texts = data_prepare(args.dataset, args.batch_size, cache_dir=args.dataset_cache)
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
    global model
    if args.hook:
        model = runtime.model

    global T_load
    global decomp_time, comp_time, pack_time, comp_done_time, decomp_done_time, merge_time, prep_time1, prep_time2, prep_time3, prep_time4
    global unpack_time, post_time1, post_time2, post_time3
    global pp1_time, pp2_time, pp3_time, pp4_time, wait_comp_start

    round_times = np.zeros(n)
    round_peak_gpu_MB = np.zeros(n)
    round_peak_gpu_reserved_MB = np.zeros(n)
    round_peak_gpu_bytes = np.zeros(n, dtype=np.int64)
    round_peak_gpu_reserved_bytes = np.zeros(n, dtype=np.int64)
    t_start = time.time()
    if args.weight_async:
        runtime.prefetch_layer(0)
    for i in range(n):
        nvtx.range_push(f"memrift_round_{i}")

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

        # t0 = time.time()
        if EVAL:
            with torch.no_grad():
                loss = model(**inputs).loss
        else:
            nvtx.range_push(f"memrift_forward_{i}")
            with bf16_ctx:
                if "gemma" in args.model:
                    out = model(**inputs, return_dict=False)
                    loss = out[0] if isinstance(out, tuple) else out.loss
                else:
                    loss = model(**inputs).loss
            nvtx.range_pop()

        print(f"{loss=}")
        tf = time.time()

        if not EVAL:
            nvtx.range_push(f"memrift_backward_step_{i}")
            # if args.weight_async:
            #     runtime.prefetch_layer(-1)
            loss.backward()
            # print(f"After backward")
            optimizer.step()
            # print(f"After step")
            optimizer.zero_grad()
            if args.hook:
                runtime._reset()
            # print(f"After zero grad")
            nvtx.range_pop()

        # t = (tb - t0)
        torch.cuda.synchronize()
        tb = time.time()
        # t1 = (time.time() - t0)

        if EVAL:
            nvtx.range_pop()
            break

        dur   = time.time() - t_start               # 纯训练耗时
        dur_f = tf - t_start               # forward
        dur_b = tb - tf               # backward
        peak_gpu_bytes = torch.cuda.max_memory_allocated(device)
        peak_gpu_reserved_bytes = torch.cuda.max_memory_reserved(device)
        peak_gpu_MB = peak_gpu_bytes / 1024 / 1024
        peak_gpu_reserved_MB = peak_gpu_reserved_bytes / 1024 / 1024
        with tegra_stats_lock:
            process_rss_bytes = process.memory_info().rss
            available_bytes = psutil.virtual_memory().available
            process_rss_MB = process_rss_bytes // 2**20
            available_MB = available_bytes // 2**20
            tegra_stats["process_rss_MB_max"] = max(tegra_stats["process_rss_MB_max"], process_rss_MB)
            tegra_stats["process_rss_bytes_max"] = max(tegra_stats["process_rss_bytes_max"], process_rss_bytes)
            current_min = tegra_stats["system_available_MB_min"]
            tegra_stats["system_available_MB_min"] = available_MB if current_min is None else min(current_min, available_MB)
            current_bytes_min = tegra_stats["system_available_bytes_min"]
            tegra_stats["system_available_bytes_min"] = available_bytes if current_bytes_min is None else min(current_bytes_min, available_bytes)

        wandb.log({
            "timestamp": int((time.time()-T0)*1000),
            "phase": PHASE_EPOCH,
            "round": i,
            "round_sec": dur,
            "round_fwd_sec": dur_f,
            "round_bwd_sec": dur_b,
            "round_peak_gpu_alloc_MB": peak_gpu_MB,
            "round_peak_gpu_reserved_MB": peak_gpu_reserved_MB,
            "loss": loss.item(),
        })
        t_start = time.time()

        round_times[i] = dur
        round_peak_gpu_MB[i] = peak_gpu_MB
        round_peak_gpu_reserved_MB[i] = peak_gpu_reserved_MB
        round_peak_gpu_bytes[i] = peak_gpu_bytes
        round_peak_gpu_reserved_bytes[i] = peak_gpu_reserved_bytes
        nvtx.range_pop()

    summary_start = min(args.warmup_rounds, max(n - 1, 0))
    measured_times = round_times[summary_start:]
    measured_gpu_peaks = round_peak_gpu_MB[summary_start:]
    measured_gpu_reserved_peaks = round_peak_gpu_reserved_MB[summary_start:]
    measured_gpu_byte_peaks = round_peak_gpu_bytes[summary_start:]
    measured_gpu_reserved_byte_peaks = round_peak_gpu_reserved_bytes[summary_start:]
    avg_t  = np.mean(measured_times)
    std_t  = np.std(measured_times)
    peak_gpu_MB = np.max(measured_gpu_peaks) if measured_gpu_peaks.size else float(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
    peak_gpu_reserved_MB = np.max(measured_gpu_reserved_peaks) if measured_gpu_reserved_peaks.size else float(torch.cuda.max_memory_reserved(device) / 1024 / 1024)
    peak_gpu_bytes = np.max(measured_gpu_byte_peaks) if measured_gpu_byte_peaks.size else torch.cuda.max_memory_allocated(device)
    peak_gpu_reserved_bytes = np.max(measured_gpu_reserved_byte_peaks) if measured_gpu_reserved_byte_peaks.size else torch.cuda.max_memory_reserved(device)
    with tegra_stats_lock:
        tegra_samples = tegra_stats["samples"]
        peak_ram_MB = tegra_stats["ram_used_MB_max"]
        avg_cpu_util = tegra_stats["cpu_util_sum"] / tegra_samples if tegra_samples else None
        avg_gpu_util = tegra_stats["gpu_util_sum"] / tegra_samples if tegra_samples else None
        process_rss_MB_max = tegra_stats["process_rss_MB_max"]
        process_rss_bytes_max = tegra_stats["process_rss_bytes_max"]
        system_available_MB_min = tegra_stats["system_available_MB_min"]
        system_available_bytes_min = tegra_stats["system_available_bytes_min"]
    wandb.summary["round_time_mean_sec"] = avg_t
    wandb.summary["round_time_std_sec"]  = std_t
    wandb.summary["round_peak_gpu_alloc_MB_max"] = peak_gpu_MB
    wandb.summary["round_peak_gpu_reserved_MB_max"] = peak_gpu_reserved_MB
    wandb.summary["ram_used_MB_max"] = peak_ram_MB
    if avg_cpu_util is not None:
        wandb.summary["cpu_util_mean"] = avg_cpu_util
        wandb.summary["gpu_util_mean"] = avg_gpu_util
    wandb.summary["weight_load_sec"]  = T_load
    result = {
        "model": os.path.basename(os.path.normpath(args.model)),
        "checkpoint": None if args.checkpoint is None else os.path.basename(os.path.normpath(args.checkpoint)),
        "dataset": args.dataset,
        "synthetic_data": args.synthetic_data,
        "finetune_type": args.finetune_type,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "rounds": n,
        "warmup_rounds": args.warmup_rounds,
        "gradient_checkpointing": args.gradient_checkpointing,
        "hook": args.hook,
        "weight": args.weight,
        "weight_async": args.weight_async,
        "activation": args.activation,
        "act_async": args.act_async,
        "round_time_mean_sec": float(avg_t),
        "round_time_std_sec": float(std_t),
        "round_peak_gpu_alloc_MB_max": float(peak_gpu_MB),
        "round_peak_gpu_reserved_MB_max": float(peak_gpu_reserved_MB),
        "round_peak_gpu_alloc_bytes_max": int(peak_gpu_bytes),
        "round_peak_gpu_reserved_bytes_max": int(peak_gpu_reserved_bytes),
        "ram_used_MB_max": None if peak_ram_MB is None else int(peak_ram_MB),
        "ram_used_bytes_max": None if peak_ram_MB is None else int(peak_ram_MB) * 2**20,
        "process_rss_MB_max": int(process_rss_MB_max),
        "process_rss_bytes_max": int(process_rss_bytes_max),
        "system_available_MB_min": None if system_available_MB_min is None else int(system_available_MB_min),
        "system_available_bytes_min": None if system_available_bytes_min is None else int(system_available_bytes_min),
        "cpu_util_mean": None if avg_cpu_util is None else float(avg_cpu_util),
        "gpu_util_mean": None if avg_gpu_util is None else float(avg_gpu_util),
        "weight_load_sec": float(T_load),
        "act_pack_count": int(act_pack_count),
        "act_pack_bytes": int(act_pack_bytes),
        "act_pack_boundary_count": int(act_pack_boundary_count),
        "act_pack_boundary_bytes": int(act_pack_boundary_bytes),
        "weight_materialize_sync_count": int(weight_materialize_sync_count),
        "weight_materialize_async_count": int(weight_materialize_async_count),
        "weight_release_count": int(weight_release_count),
    }
    with c3_lock:
        result.update({f"c3_{k}": int(v) for k, v in c3_stats.items()})
    print("MEMRIFT_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    result_path = os.path.join(args.results_dir, "result.json")
    with open(result_path, "w") as result_file:
        json.dump(result, result_file, indent=2, sort_keys=True)

    # Save model weights
    if args.save_model:
        for name, param in model.named_parameters():
            if isinstance(param, CompressedParam):
                print(f"mat {name}")
                param.materialize(sync=True)  # 确保所有 CompressedParam 都被解压到 GPU
                param.CtoD_evt.synchronize()
                param.set_param()                  # 替换成 Tensor
                param.release_comp()

        def remove_all_hooks(model: torch.nn.Module):
            """
            Remove every hook from `model`, including its sub-modules.
            """
            for m in model.modules():
                # PyTorch 1.13+ 的 hook 容器名称
                for attr in (
                    "_forward_hooks",
                    "_forward_pre_hooks",
                    "_backward_hooks",
                    "_full_backward_hooks",
                ):
                    hooks = getattr(m, attr, None)
                    if hooks:
                        hooks.clear()          # 直接清空字典
                # TorchScript 2.1 可能还有 _state_dict_hooks, _load_state_dict_pre_hooks
            torch.cuda.empty_cache()            # 如有需要

        MERGE_OUT = args.save_model_dir

        if args.hook:
            remove_all_hooks(model)
            container = model.model if hasattr(model, "model") else model
            if hasattr(container, "model"):          # PeftModel 里还有一层
                container = container.model
            if hasattr(container, "transformer"):    # GPT-2 家族
                container = container.transformer
            layers = container.layers                # 列表(32,) 或 (40,)

            for i, wrap in enumerate(layers):
                if isinstance(wrap, DecoderLayerWrapper):
                    layers[i] = wrap.layer           # 只留下真 · MistralDecoderLayer

        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(MERGE_OUT, safe_serialization=True)
        # tokenizer 直接拷贝基座即可
        tokenizer.save_pretrained(MERGE_OUT)

measure()
