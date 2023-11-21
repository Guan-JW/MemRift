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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from accelerate import init_empty_weights
import float_split_stride_pin as fs_sp

import zstandard as zstd
import lz4.frame as lz4f, lz4.block as lz4b
import io
from dataclasses import dataclass

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/Mistral-7B-v0.1")
parser.add_argument("--outdir", default="./weight_comp/prepare_weight/zstd_comped_weights_level21")
parser.add_argument("--finetune_type", choices=["full", "lora", "qlora"], default="lora", help="Type of finetuning")
parser.add_argument("--check_diff", action="store_true",
                    help="同时加载原始模型并比对差值（耗显存）")
parser.add_argument("--hook", action="store_true", help="Run with compression hooks")
parser.add_argument("--debug", action="store_true", help="Run with debug")
parser.add_argument("--print_ratio", action="store_true", help="Print activation's compression ratio")
parser.add_argument("--print_time", action="store_true", help="Print activation's compression time")
parser.add_argument("--weight", default=False, action="store_true", help="Switch on weight compression")
parser.add_argument("--activation", default=False, action="store_true", help="Switch on activation compression")
parser.add_argument(
        "--level", type=int, default=1, help="Zstd compression level (<22)"
    )
parser.add_argument(
        "--round", type=int, default=5, help="# training cycles"
    )
parser.add_argument(
        "--max_length", type=int, default=512, help="Input length"
    )
parser.add_argument(
        "--batch_size", type=int, default=1, help="Input batch size"
    )
args = parser.parse_args()
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

# cfg = AutoConfig.from_pretrained(args.model)
# with init_empty_weights():            # 所有 Parameter 都在 device='meta'
#     model = AutoModelForCausalLM.from_config(cfg)
#     print(f"{model.dtype=}")
# model.tie_weights()
def materialize_trainables(model, device):
    for name, p in model.named_parameters():
        if p.requires_grad and p.device.type == "meta":
            # print(f"{name=}")
            # 创建同形状实张量并做一次正常初始化
            new_p = torch.nn.Parameter(
                torch.empty_like(p, device=device),
                requires_grad=True,
            )
            torch.nn.init.kaiming_uniform_(new_p, a=5**0.5)  # LoRA 默认
            # 把新权重塞回原模块
            parent_name, _, attr = name.rpartition(".")
            parent = dict(model.named_modules())[parent_name]
            setattr(parent, attr, new_p)
    return model

# 在构造优化器 **之前** 调用
# model = materialize_trainables(model, device)

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
# --------------------------------------------------------------
#           2. Model Weight Compression & Injection 
# --------------------------------------------------------------
if args.weight:
    index = []                                  # <layer name, binary file, shape, dtype>
        # 模块级别，全局只有这一份；属性却会被线程隔离
    def decompress_into_pinned(bytes_like, numel):
        """把 exponent 压缩数据直接解压到 pinned Tensor"""
        if args.print_time:
            t0 = time.time()
        # 1) 申请 pinned buffer
        buf = torch.empty(numel, dtype=torch.uint8, pin_memory=True)

        # 2) 用 streaming API 直接写入
        # dctx = zstd.ZstdDecompressor()
        # if not hasattr(_tls, "dctx"):        # 该线程第一次用，创建解压器
        #     _tls.dctx = zstd.ZstdDecompressor()
        # zrec = _tls.dctx.decompress(bytes_like)
        # arr = np.frombuffer(zrec, dtype=np.uint8)
        
        # buf = torch.empty(arr.shape, dtype=torch.uint8,
        #                     device='cpu', pin_memory=True)
        # buf.numpy()[:] = arr

        tls = _tls                          # 取到当前线程的 thread-local 对象
        if not hasattr(tls, "dctx"):        # 该线程第一次用，创建解压器
            tls.dctx = zstd.ZstdDecompressor()
        with tls.dctx.stream_reader(memoryview(bytes_like)) as reader:
            view = memoryview(buf.numpy())   # numpy() 不复制，只拿 data_ptr
            nread = reader.readinto(view)
            assert nread == numel, "decompress size mismatch"

        if args.print_time:
            global decomp_time, decomp_lock
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
        def __new__(cls, orig_shape, sm_gpu, exp_mv, parent, attr, layer_id):
            dummy = torch.empty(     # 0-element，占不了显存
                0, dtype=MODEL_TYPE, device=sm_gpu.device
            )
            # 这里只能传 (data, requires_grad) 两个位置参数
            return super().__new__(cls, dummy, requires_grad=False)

        def __init__(self, orig_shape, sm_gpu, exp_mv, parent, attr, layer_id):
            super().__init__()

            self.orig_shape   = tuple(orig_shape)
            self.sm_gpu    = sm_gpu            # uint8 pinned
            self.exp_mv    = exp_mv               # compressed exponent mv
            self._exp_host = None
            self._bf16     = None

            self.parent, self.attr = parent, attr
            self._ready_event = threading.Event()
            self.CtoD_evt   = None
            self._hooked = False

            if args.debug:
                self.layer_id = layer_id

        # ------------ on-demand materialize ------------------------
        def materialize(self, sync=True):
            """
            解压 exponent → 重建 bf16 → H2D copy
            调用方负责与计算流同步（可 wait_stream）。
            """
            try:
                # if self._bf16 is not None:
                #     # print(f"Already materialized")
                #     return self._bf16

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
                # current_stream = torch.cuda.current_stream()
                current_stream = torch.cuda.Stream()
                bf16 = fs_sp.merge(self._exp_host, self.sm_gpu, self.orig_shape, stride, 0, MODEL_TYPE, current_stream.cuda_stream)
                self._bf16 = bf16.view(self.orig_shape)
                # evt = torch.cuda.Event()
                # evt.record()          # ensure all copies/kernels before event
                evt = current_stream.record_event()   # 拷贝结束事件

                self.CtoD_evt = evt
                self._ready_event.set()
                if args.debug:
                    print(f"Setted event! layer={self.layer_id}: {self.parent}-{self.attr}")

                if sync:                       # ------- 同步路径 (漏预取) -------
                    self.CtoD_evt.synchronize()

                # del sign_gpu, exp_gpu, sign_i32, exp_i32
                    
                return self._bf16
            except Exception as e:
                import traceback, sys
                traceback.print_exc(file=sys.stderr)
                raise                # 重新抛给上层，让你能看到

        def set_param(self):
            # 现在才把最终对象挂回父模块
            if self._hooked:
                return
                
            assert self._bf16 is not None

            if self._exp_host is not None:
                del self._exp_host
                self._exp_host = None
            self._bf16._source_cp = self
            self.data = self._bf16
            _PTR2CP[self._bf16.data_ptr()] = self

            # 3) 只挂一次 backward hook
            self._hooked = True

        def release(self, delref=True):
            "回收 GPU 张量，恢复到压缩状态"
            if args.debug:
                print(f"Releasing layer={self.layer_id}: {self.parent}.{self.attr} {self.orig_shape=}")

            if delref:  # called after backward, 防止之后重用空间的tensor 又识别到 cp
                del _PTR2CP[self._bf16.data_ptr()]

            self._bf16 = None
            self.data = torch.empty(0, dtype=torch.bfloat16, device=self.sm_gpu.device)
            # if self._bf16 is not None:
            #     del self._bf16
            #     self._bf16 = None
            self._ready_event.clear()
            self._hooked = False


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
            cp = CompressedParam(it["shape"], sm_gpu, exp_bytes, parent=mod, attr=attr, layer_id=layer_id)
            # print(f"{layer_name=}")

            if layer_name is None:
                if args.debug:
                    print(f"Materializing {module}")
                cp.materialize(sync=True)
                # cp._ready_event.wait()
                # cp.CtoD_evt.synchronize()
                cp.set_param()
            else:
                layer2cps.setdefault(layer_name, []).append(cp)
                max_job_size = max(len(layer2cps[layer_name]), max_job_size)
                
                if attr in mod._parameters:
                    # del mod._parameters[attr]   # 删除原来的 Parameter
                    mod._parameters[attr] = cp           # property 安全
                else:
                    setattr(mod, attr, cp)


        idx.clear()    

        return layer2cps, max_job_size

        print(f"✅  injected {len(idx)} frozen tensors (mmap freed)")

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

    # 3) 再测一次
    snapshot_mem("after_inject")
    print_model_size("after inject", model) 

    gc.collect()
    process = psutil.Process(os.getpid())
    cpu_mem_peak1 = process.memory_info().rss / 1024 / 1024  # 单位: MB
    print(f"Peak CPU memory usage: {cpu_mem_peak1:.2f} MB")
    print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated(device) / 1024 / 1024:.2f} MB")


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


    # print(f"{max_job_size=}, {layer2cps=}")
    # --------------------------------------------------------------
    #   3. pipeline prefetch (layer n 计算 ↔ layer n+1 copy) 
    # --------------------------------------------------------------
    num_workers = min(9, os.cpu_count() // 3) 
    _pref_q = queue.Queue(maxsize=10)  # 64 jobs per worker
    def _pref_worker(worker_id):
        torch.cuda.set_device(device)
        # stream = _pref_streams[worker_id]
        # stream = _pref_stream
        if not hasattr(_tls, "dctx"):
            _tls.dctx = zstd.ZstdDecompressor()
        while True:
            cp = _pref_q.get()
            if cp is None:
                break
            cp.materialize(sync=False)      # ← 各线程自己的 stream
            _pref_q.task_done()

    # 起线程池
    workers = [threading.Thread(target=_pref_worker, args=(i,), daemon=True)
            for i in range(num_workers)]
    for w in workers: w.start()

    def prefetch_first_layer(layer2cps, layer_names):
        # 首层预取
        for p in layer2cps.get(layer_names[0], []):
            try: 
                _pref_q.put_nowait(p)
            except queue.Full: 
                print(f"The queue is full!")
                pass

    def prefetch_last_layer(layer2cps, layer_names):
        # 尾层预取
        for p in layer2cps.get(layer_names[-1], []):
            try: 
                _pref_q.put_nowait(p)
            except queue.Full: 
                print(f"The queue is full!")
                pass

            
    def install_fwd_prefetch_hooks(model, layer_names, layer2cps):
        name2layer = {n: m for n, m in model.named_modules()}
        for cur, nxt in zip(layer_names[:-1], layer_names[1:]):
            nxt_pars = layer2cps.get(nxt, [])
            cur_pars = layer2cps.get(cur, [])

            def _hook_post(_, __, ___, cur_pars=cur_pars):
                # print(f"Running hook!")
                for p in cur_pars:
                    # before = torch.cuda.memory_allocated()
                    p.release()  # 释放当前层的 CompressedParam
                    # torch.cuda.synchronize()
                    # delta = before - torch.cuda.memory_allocated()
                    # print("freed", delta/1024**2, "MB")   # 应看到 ~320 MB
            def _hook_pre(_, __, nxt_pars=nxt_pars, cur_pars=cur_pars):
                # if args.debug:
                #     print(f"_hook_pre: {cur_pars=}, {nxt_pars=}")
                for cp in cur_pars:
                    cp._ready_event.wait()
                    cp.CtoD_evt.synchronize()
                    cp.set_param()

                for p in nxt_pars:
                    try: 
                        _pref_q.put_nowait(p)
                        # _pref_q.put(p)
                        # print(f"Pushed into queue")
                    except queue.Full: 
                        print(f"The queue is full!")
                        pass
            name2layer[cur].register_forward_pre_hook(_hook_pre)    # overlap current layer's computation with next layer's decompression
            name2layer[cur].register_forward_hook(_hook_post)   # release after calculation
        
        # -------- 最后一层单独处理 --------
        last = layer_names[-1]
        last_pars = layer2cps.get(last, [])

        def _last_pre(_, __, last_pars=last_pars):
            for cp in last_pars:
                cp._ready_event.wait()          # 等解压 + H2D
                cp.CtoD_evt.synchronize()
                cp.set_param()                  # 替换成 Tensor
        def _last_post(_, __, ___, last_pars=last_pars):
            # print(f"Running hook!")
            for p in last_pars:
                p.release()  # 释放当前层的 CompressedParam
        name2layer[last].register_forward_pre_hook(_last_pre)
        # name2layer[last].register_forward_hook(_last_post)

    def install_bwd_prefetch_hooks(model, layer_names, layer2cps):   
        name2layer = {n: m for n, m in model.named_modules()}

        for cur, nxt in zip(layer_names[:-1], layer_names[1:]):
            nxt_pars = layer2cps.get(nxt, [])
            cur_pars = layer2cps.get(cur, [])

            def _hook_post(_, __, ___, cur_pars=cur_pars):
                # print(f"Running hook!")
                for p in cur_pars:
                    p.release(delref=True)  # 释放当前层的 CompressedParam
            def _hook_pre(_, __, nxt_pars=nxt_pars, cur_pars=cur_pars):
                for cp in cur_pars:
                    cp._ready_event.wait()
                    cp.CtoD_evt.synchronize()
                    cp.set_param()

                for p in nxt_pars:
                    try: 
                        _pref_q.put_nowait(p)
                        # _pref_q.put(p)
                        # print(f"Pushed into queue")
                    except queue.Full: 
                        print(f"The queue is full!")
                        pass
            
            name2layer[cur].register_full_backward_hook(_hook_post)   # release after calculation
            name2layer[cur].register_full_backward_pre_hook(_hook_pre)    # overlap current layer's computation with next layer's decompression
        
        # -------- 最后一层单独处理 --------
        last = layer_names[-1]
        last_pars = layer2cps.get(last, [])

        def _last_pre(_, __, last_pars=last_pars):
            # print(f"Last pre!")
            for cp in last_pars:
                # print(f"Waiting for {cp.parent}-{cp.attr}")
                cp._ready_event.wait()          # 等解压 + H2D
                cp.CtoD_evt.synchronize()
                cp.set_param()                  # 替换成 Tensor
        def _last_post(_, __, ___, last_pars=last_pars):
            print(f"Last post!")
            for p in last_pars:
                p.release()  # 释放当前层的 CompressedParam
        # name2layer[last].register_full_backward_hook(_last_post)
        name2layer[last].register_full_backward_pre_hook(_last_pre) # 必须添加这个，为了覆盖 _PTR2CP 中对应的指针，防止访问到非法 tensor
        
    name2layer = {n: m for n, m in model.named_modules()}

    # # 3) install prefetch hooks
    layer_names = [f"base_model.model.model.layers.{i}"
                    for i in range(model.config.num_hidden_layers)]

    install_fwd_prefetch_hooks(model, layer_names, layer2cps)
    install_bwd_prefetch_hooks(model, layer_names[::-1], layer2cps)
    # for cp in layer2cps.get(layer_names[0], []):
    #     print(f"{cp.orig_shape=}")


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
    def __init__(self, future, cpu_exp_buf, sm_bits, evt, rid):
        self.future = future            # 便于重复获取压缩数据
        self.sm_bits = sm_bits          # 用于恢复数据
        # self.gpu_exp = gpu_exp          # 在数据拷贝期间存放 gpu 上的 exp_bits，防止拷贝出错
        self.cpu_exp_buf = cpu_exp_buf  # 在数据拷贝和压缩期间存放 cpu（pin_memory） 上的 exp_bits，防治拷贝出错
        self.refcount = 1           # 计数器。todo: no need for an explicit refcount?
        self.DtoC_copy_evt = evt

        self.scheduled = False          # ➜ 已提交 decode 任务？
        self.lock = threading.Lock()    # ➜ 防止并发重复调度
        # self.decoded = None
        self.cpu_exp = None # todo : for decompressed data   
        self.st_ref = None  # todo

        # self.CtoD_copy_evt = torch.cuda.Event(blocking=False)
        # self.decode_start_evt = threading.Event()  # 压缩完成，开启解压 ⇒ set()
        # self.decode_done_evt   = threading.Event()  # 解压完成 ⇒ set()
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
        # self.DtoC_copy_evt = None
        
    def release_payload(self):
        self.cpu_exp = None
        self.sm_bits = None
    
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

def get_cctx(algo="zstd", level=-1):
    try:
        return _tls.cctx
    except AttributeError:
        if algo == "zstd":
            _tls.cctx = zstd.ZstdCompressor(level=level)
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


class HookRuntime:
    def __init__(self, pool_workers=None, window=None):
        self.pool_workers = pool_workers or os.cpu_count() // 3
        self._build()
        self.window = window or 10
    
    # ---------------- internal helpers ----------------
    def _build(self):
        # 1. 线程 / 进程池
        # self.wait_pool    = fut.ThreadPoolExecutor(max_workers=self.pool_workers)
        self.compress_pool= fut.ThreadPoolExecutor(max_workers=self.pool_workers * 2)
        self.decode_pool  = fut.ThreadPoolExecutor(max_workers=self.pool_workers)

        # 2. CUDA stream（必须 *先* 再构造，否则 reset 时还在用旧 stream）
        # self.d2h_stream   = torch.cuda.Stream(priority=0)
        # self.h2d_stream   = torch.cuda.Stream(priority=0)
        # self.d2h_stream   = torch.cuda.Stream()
        # self.h2d_stream   = torch.cuda.Stream()

        # 3. 运行期状态
        self.tensor_cache = weakref.WeakValueDictionary()   # (ptr,bytes,off) -> Token
        self.storage_gen  = defaultdict(int)               # ptr -> generation
        self.seq_counter  = 0
        self.activ_q      = deque()                        # (seq_id, token)

        # 4. 轻量级同步原语
        self.q_lock       = threading.Lock()

    def reset(self):
        """在一个 epoch（或想要的颗粒度）结束后调用"""

        # a. 等待/关闭线程池，保证后台任务完成
        for pool in (self.compress_pool, self.decode_pool):
            pool.shutdown(wait=True)

        self.tensor_cache.clear()
        self.storage_gen.clear()
        self.activ_q.clear()
        self.seq_counter = 0

        # b. 清空 CUDA 资源
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 重新创建新 stream，旧 stream 让 GC 处理
        # （不能直接 .reset()，得重新 new）
        # ------------------------------------------------
        self._build()          # 彻底重建所有资源
        # ------------------------------------------------

        # c. 额外可选：清 peak mem 计数器
        torch.cuda.reset_peak_memory_stats()

        # 可选，主动触发 GC（调试用）
        import gc
        gc.collect()
        print('reset done, forced gc')

    def next_generation(self, ptr_key):
        self.storage_gen[ptr_key] += 1
        return self.storage_gen[ptr_key]
    
    # unpack 时调用并挂载到 future 上
    def _prefetch(self, token, tmeta, seq_id):
        fut_jxl = token.future          # 压缩已提交，fut_jxl 完成时拿到 jxl bytes
        
        def _decode_and_copy(fut, algo="zstd"):      # 在线程池中执行
            try:
                # global decomp_time, decomp_cnt, wait_comp_done, jxl_decode_done
                if args.print_time:
                    start = time.time()
            
                current_stream = torch.cuda.Stream()
                current_stream.wait_event(token.DtoC_copy_evt)
                
                if args.debug:
                    comped_bytes, exp_arr, numel = fut.result()
                else:
                    comped_bytes, numel = fut.result()
                # t1 = time.time()
                # comp_done_time = t1 - start
                # token.decode_start_evt.set()    # <- 关键：告诉 _unpack "我开始解压了"
                
                # --- CPU 解码 ---
                # print(f"_decode_and_copy: Decompressing {seq_id}")
                if algo == "JXL":
                    exp_arr = imagecodecs.jpegxl_decode(comped_bytes).ravel().view(np.uint8) # do not cut here
                    # assert (exp_arr == exp_rec).all(), "JPEG-XL decode error"
                    cpu_exp = torch.from_numpy(exp_arr)
                elif algo == "zstd":
                    # dctx = zstd.ZstdDecompressor()
                    dctx = get_dctx(algo)
                    # zrec = dctx.decompress(comped_bytes)
                    # # exp_arr = np.frombuffer(zrec, dtype=np.uint8)
                    # cpu_exp = torch.frombuffer(zrec, dtype=torch.uint8)
                    # # cpu_exp = raw.pin_memory()
                    cpu_exp = torch.empty(numel, dtype=torch.uint8,
                                        device='cpu', pin_memory=True)
                    with dctx.stream_reader(memoryview(comped_bytes)) as reader:
                        view = memoryview(cpu_exp.numpy())
                        nread = reader.readinto(view)
                elif algo == "lz4":
                    # dctx = lz4f.LZ4FrameDecompressor()
                    dctx = get_dctx(algo)
                    zrec = dctx.decompress(comped_bytes)
                    arr = np.frombuffer(zrec, dtype=np.uint8)
                    if args.debug:
                        if not (arr == exp_arr).all():
                            print(f"{np.frombuffer(comped_bytes, dtype=np.uint8)=}")
                            print(f"{arr=}, {exp_arr=}")
                        assert (arr == exp_arr).all()
                    cpu_exp = torch.empty(arr.shape, dtype=torch.uint8,
                                        device='cpu', pin_memory=True)
                    cpu_exp.numpy()[:] = arr

                token.free_future() # delete the compression future to free memory

                # current_stream = torch.cuda.current_stream()  # 获取当前 stream
                with torch.cuda.stream(current_stream):
                    rst = fs_sp.merge(cpu_exp, token.sm_bits, tmeta.shape, tmeta.stride, tmeta.offset, tmeta.dtype, current_stream.cuda_stream)
                    rst.record_stream(current_stream)
                # evt = torch.cuda.Event()
                # evt.record()
                evt = current_stream.record_event()   # 拷贝结束事件

                token.cpu_exp       = cpu_exp     # 为了保险，把 pinned tensor保存到 token，等 copy 结束再删除
                token.rst           = rst
                token.CtoD_copy_evt = evt
                token.ready_evt.set()             # <- 关键：告诉 _unpack "我好了"

                # assert hasattr(token, "cpu_exp"), "cpu_exp not set!"
                
                if args.print_time:
                    global decomp_time, decomp_lock
                    with decomp_lock:
                        decomp_time += time.time() - start
                #     decomp_cnt += 1
                #     wait_comp_done += comp_done_time
                #     jxl_decode_done += jxl_decode_time
            except Exception as e:
                import traceback, sys
                traceback.print_exc(file=sys.stderr)
                raise                # 重新抛给上层，让你能看到

        # 把整个 decode+H2D 拷贝作为回调挂到 fut_jxl 上
        # fut_jxl.add_done_callback(lambda fut: jpegxl_pool.submit(_decode_and_copy, fut)) # 此处 fut 就是完成了 compress 事件的 fut_jxl 本身

        def _schedule_decode(fut):
            try:
                # 压缩完成，删掉 token 中登记的 gpu_exp 和 cpu_exp_buf，节省内存
                # token.free_gpu_exp()
                # token.free_cpu_exp_buf()
                self.decode_pool.submit(_decode_and_copy, fut, "zstd")
            except Exception as e:
                print(f"[ERROR] Failed to schedule decode: {e}")
            
            
        fut_jxl.add_done_callback(_schedule_decode)

        return True


    def _prefetch_batch(self):
        # global PREFETCH_ID
        # ACTIV_Q 尾部 seq 最大
        # lookahead = sorted(ACTIV_Q)[-WINDOW:]   # 取后 WINDOW 个 seq_id
        lookahead = list(self.activ_q)[-self.window:]    # 取后 WINDOW 个 token
        for seq, tmeta, tok in lookahead[::-1]:   # 反向遍历
            if tok.scheduled:
                # print(f"Already pre-fetched {seq=},")
                continue
            self._prefetch(tok, tmeta, seq)
            with tok.lock:
                tok.scheduled = True

    def flush_prefetch_tail(self):
        # global PREFETCH_ID
        # PREFETCH_ID = len(ACTIV_Q)
        with self.q_lock:
            # print(f"flushing tail: {len(ACTIV_Q)}")
            if self.activ_q:                         # 队列非空
                self._prefetch_batch()

    def pop_token(self, seq_id_expected, ensure_prefetch=True):
        """从队尾向前扫描 ≤ WINDOW 步，取到匹配的 seq_id"""
        for i in range(1, min(self.window, len(self.activ_q)) + 1):
            seq, tmeta, tok = self.activ_q[-i]
            # assert seq not in PREFETCHED, f"seq_id {seq_id_expected} has not been processed"
            if ensure_prefetch and not tok.scheduled:
                self._prefetch_batch()

            if seq == seq_id_expected:
                self.activ_q.remove((seq, tmeta, tok))   # O(W) 内部线性扫描
                return tok
        # 没找到：说明窗口不够大，走兜底
        # print(f"pop_token: seq_id {seq_id_expected} not found in ACTIV_Q")
        return None
    
    def get_token(self, seq_id_expected):
        for i in range(len(self.activ_q)-1, -1, -1):
            seq, tmeta, tok = self.activ_q[i]
            if seq == seq_id_expected:
                print(f"Found seq_id {seq_id_expected} in ACTIV_Q, at position {i}")
                return tok

        return None

def make_ptr_key(t: torch.Tensor):
    return (t.data_ptr(), t.nbytes)

def make_tensor_key(t: torch.Tensor, gen: int):
    return (t.data_ptr(), gen, t.nbytes)

# -------------------------------------------------------------------------
#                           Pack & Unpack hooks
# -------------------------------------------------------------------------

wait_comp_done = 0
jxl_decode_done = 0
wait_time = decode_time = copy_time = 0
pack_time = unpack_time = 0
prep_time1 = prep_time2 = prep_time3 = prep_time4 = prep_time5 = 0
post_time1 = post_time2 = post_time3 = post_time4 = post_time5 = 0
rid = 0

lid = -1
uniq_seq_id = 0
def build_hooks(rt: HookRuntime):
    def _pack(t: torch.Tensor):

        # ----------------------------------------------------
        #               1. Check frozen weights
        # ----------------------------------------------------
        if args.weight:
            if isinstance(t, CompressedParam):
                if args.debug:
                    print(f"[pack]: layer {lid} is Compressed Param! layer={t.layer_id} orig_shape={t.orig_shape}, _bf16=None? ({t._bf16 is None})")
                return t
            
            if not t.requires_grad:
                cp = _PTR2CP.get(t.data_ptr(), None)
                if cp is not None:
                # if isinstance(cp, CompressedParam):
                    # print("ahaaha!!!!")
                    # print(f"{len(cp.exp_mv)=}, {cp.sm_gpu.shape=}")
                    if args.debug:
                        print(f"[pack]: layer {lid} is Compressed Param! layer={cp.layer_id}, orig_shape={cp.orig_shape}, t.shape={t.shape}, _bf16=None? ({cp._bf16 is None})")
                        # return (cp, t)

                    return (cp, t.shape, t.stride())    # 就目前的参数来看，没有对参数做中间 view 的，都是整个转置，故不存 offset


        # ----------------------------------------------------
        #               2. Check activations
        # ----------------------------------------------------
        if not args.activation:
            return t

        if is_lora_weight(t):   # 模型权重 tensor
            # print(f"[skip] Detected model weight: shape={t.shape}, dtype={t.dtype}, is_leaf={t.is_leaf}, requires_grad={t.requires_grad}, grad_fn={type(t.grad_fn).__name__ if t.grad_fn else None}")
            return t

        if not t.dtype in (torch.float32, torch.bfloat16) or t.numel() < 4096:
            # print(f"Got type {t.dtype} (bytes={t.nbytes}), not fp32 or bf16")
            return t
            # raise TypeError(f"only fp32 / bf16 supported, got {t.dtype}, grad_fn: {type(t.grad_fn).__name__}")
        
        if not torch.all(torch.isfinite(t)):
            # print(f"_pack: Found NaN or Inf in unpacked tensor: {t.shape=}")
            return t
        
        global prep_time1, prep_time2, prep_time3, prep_time4, pack_time
        start = time.time()

        ptr_key = make_ptr_key(t)                   # (t.data_ptr(), t.nbytes)
        gen = rt.storage_gen.setdefault(ptr_key, 0)
        key = make_tensor_key(t, gen)               # (t.data_ptr(), gen, t.nbytes)

        new_key = True
        if key in rt.tensor_cache:
            if rt.tensor_cache[key].st_ref() is not None:  # to avoid duplicate compression
                rt.tensor_cache[key].inc_refcount()   # 计数器+1, 避免被 GC
                token = rt.tensor_cache[key]   # 重新获取 token
                new_key = False
            else:
                # print(f"Create new token for old key {key}")
                # print(f"{token.t._version=}, {t._version=}")
                gen = rt.next_generation(ptr_key)
                gen += 1
        p1 = time.time()
        prep_time1 += p1 - start
        
        orig_t = t
        # if key not in tensor_cache:
        if new_key:
            key = make_tensor_key(t, gen)   # (t.data_ptr(), gen, t.nbytes), update gen

            global uniq_seq_id
            uniq_seq_id += 1
            # print(f"Adding new {key=}")

            # stream = torch.cuda.Stream()
            # stream.wait_stream(torch.cuda.current_stream())
            # with torch.cuda.stream(stream):
            stream = torch.cuda.current_stream()
            cpu_pin_buf, sm_bits = fs_sp.split(t, stream.cuda_stream)
            evt = stream.record_event()   # 拷贝结束事件
            evt.synchronize()
            
            p2 = time.time()
            prep_time2 += p2 - p1   

            def _encode_token(token, algo="JXL"):

                # global comp_time, comp_cnt
                start = time.time()
                # stream = torch.cuda.Stream()  # 获取当前 stream
                # stream.wait_event(token.DtoC_copy_evt)  # 等待 DtoC 拷贝完成
                token.DtoC_copy_evt.synchronize()

                arr = token.cpu_exp_buf.reshape(-1).numpy()
                numel = arr.size

                if algo == "JXL":
                    # print("using jxl")
                    width = 1024    # tunable
                    pad   = (-arr.size) % width
                    img   = np.pad(arr, (0, pad)).reshape(-1, width)   # C-contiguous
                    assert img.dtype == np.uint8, "image must be uint8"
                    # if not arr.data.contiguous:
                        # print(f"arr is not contiguous, shape = {arr.shape}, strides = {arr.strides}")
                    
                    comped_bytes = imagecodecs.jpegxl_encode(
                        img,
                        lossless=1,
                        bitspersample=8,
                        photometric='minisblack',        # 1-channel grayscale
                        # planar=True,                    # default; keep interleaved
                    )
                    # # Remember: first view to uint8 then reshape !!!
                    # rec = imagecodecs.jpegxl_decode(comped_bytes).ravel().view(np.uint8)
                    # exp_arr = rec[:arr.size].reshape(arr.shape)
                    # # if not (arr == exp_arr).all():
                    #     # print(f"{arr.shape=}, {exp_arr.shape=}")
                    #     # print(f"{arr=}, {exp_arr=}")
                    #     # exit()
                    # assert (arr == exp_arr).all(), "JPEG-XL encode error"

                    # ratio = arr.nbytes / len(comped_bytes)
                    # # print(f"{ratio=}")
                elif algo == "zstd":
                    cctx = get_cctx(algo, args.level)
                    comped_bytes = cctx.compress(arr)

                    if 1:
                        dctx = get_dctx()
                        zrec = dctx.decompress(comped_bytes)

                        exp_arr = np.frombuffer(zrec, dtype=np.uint8)
                        if not (arr == exp_arr).all():
                            print(f"{arr=}, {exp_arr=}")
                        assert (arr == exp_arr).all()
                    
                elif algo == "lz4":
                    cctx = get_cctx(algo, args.level)
                    # cctx = lz4f.LZ4FrameCompressor(compression_level=args.level)
                    comped_bytes = lz4f.compress(arr)

                    if args.debug:
                        dctx = lz4f.LZ4FrameDecompressor()
                        zrec = dctx.decompress(comped_bytes)
                        exp_arr = np.frombuffer(zrec, dtype=np.uint8)
                        if not (arr == exp_arr).all():
                            # print(f"{np.frombuffer(comped_bytes, dtype=np.uint8)=}")
                            print(f"{arr=}, {exp_arr=}")
                        assert (arr == exp_arr).all()

                if args.print_ratio:
                    global orig_len, new_len
                    orig_len += arr.nbytes
                    new_len += len(comped_bytes)
                
                token.free_cpu_exp_buf()    # 压缩完成，free 掉 cpu 上的 exp_bits 内存
                
                if args.debug:
                    return comped_bytes, arr, numel
                else:
                    return comped_bytes, numel

            # ---- 3) 生成 token ----
            token = Token(None, cpu_pin_buf, sm_bits, evt, rid)
            token.st_ref = weakref.ref(orig_t.untyped_storage())     # weakref to the original "orig_t" (t), not the newly created contiguous t
            rt.tensor_cache[key] = token

            fut_exp = rt.compress_pool.submit(_encode_token, token, algo="zstd")   # 提交压缩任务
            token.future = fut_exp
            if 1:
                token.t = t

            p3 = time.time()
            prep_time3 += p3 - p2

        # ---- 4) 生成 tensor 元数据 ----
        p5 = time.time()
        tmeta = TensorLayout.from_tensor(t)
        with rt.q_lock:
            seq_id = rt.seq_counter
            rt.activ_q.append((seq_id, tmeta, token))   # 记录顺序
            rt.seq_counter += 1
        
        meta = {
                "key": key,
                # "shape": t.shape,
                # "stride": t.stride(),
                # "offset": t.storage_offset(),
                # "dtype": t.dtype,
                "token": token,  # 把 token 挂在 meta 上，传递给 autograd 进行存储，则不用单独存在 dict 中。用 weakref 将内存管理交给 autograd
                "seq_id": seq_id,   # 用于恢复顺序
            }
        prep_time4 += time.time() - p5

        # # with pack_lock:
        pack_time += time.time() - start
        # pack_cnt += 1
        return meta

    def _unpack(meta):
        # print("[UNPACK]")
        # ----------------------------------------------------
        #               1. Check frozen weights
        # ----------------------------------------------------
        
        if args.weight:
            global lid
            if isinstance(meta, CompressedParam):
                if args.debug:
                    print(f"[unpack]: layer {lid} is Compressed Param! {meta.data.shape=}")
                meta._ready_event.wait()
                meta.CtoD_evt.synchronize()
                meta.set_param()
                if args.debug:
                    print(f"[unpack-done]: layer {lid} is Compressed Param! {meta.data.shape=}")
                return meta.data
            elif isinstance(meta, tuple):
                cp, shape, stride = meta
                # print(f"[unpack-waiting]: layer {lid} {cp.shape=}, {shape=}")
                cp._ready_event.wait()
                cp.CtoD_evt.synchronize()
                cp.set_param()
                # print(f"[unpack]: layer {lid} {cp.shape=}, {shape=}")
                # print(f"{cp.data.shape=}, {cp.orig_shape=}, {shape=}, {stride=}")
                rec_t = cp.data.as_strided(shape, stride, 0)    # 就目前的参数来看，没有对参数做中间 view 的，都是整个转置
                return rec_t

        # ----------------------------------------------------
        #               2. Check activations
        # ----------------------------------------------------
        if not args.activation:
            return meta


        if isinstance(meta, torch.Tensor):
            # print(f"[UNPACK] t.shape={meta.shape}, is_leaf={meta.is_leaf}, grad_fn={type(meta.grad_fn).__name__ if meta.grad_fn else None}, {meta.requires_grad=}")
            return meta

        # global unpack_time, unpack_cnt, wait_time, copy_time, decode_time
        global post_time1, post_time2, post_time3, post_time4, post_time5, unpack_time
        start = time.time()

        seq_id = meta["seq_id"]
        # -------- 3.1 取出当前 token --------
        with rt.q_lock:
            token = rt.pop_token(seq_id)
            # -------- 3.2 peek 下一条并预取 --------
            if rt.activ_q:                       # 队列非空
                rt._prefetch_batch()
        p1 = time.time()
        post_time1 += p1 - start

        # ---- 1) 获取 token 元数据 ----
        key = meta['key']
        # shape = meta['shape']
        # dtype = meta['dtype']
        # stride = meta['stride']
        # offset = meta['offset']

        token.ready_evt.wait()  # 等待当前 token 解压完成
        ct = time.time()
        post_time2 += ct - p1

        # rst = fs_sp.merge(token.cpu_exp, token.sm_bits, shape, stride, offset, dtype)

        # event = torch.cuda.Event()          # ① 创建事件
        # event.record()                      # ② 记录到 *当前流*（merge 的 kernel 前面）
        # event.synchronize()                 # ③ 等流上所有已排 op 全部完成
        if token.CtoD_copy_evt:
            # 让主线程当前流等合并完成，但不阻塞 CPU
            torch.cuda.current_stream().wait_event(token.CtoD_copy_evt)  # 等待 CtoD 拷贝完成
            # token.CtoD_copy_evt.synchronize()
            token.CtoD_copy_evt = None
        rst = token.rst

        post_time3 += time.time() - ct

        # if not args.debug and token.dec_refcount():
        #     token.release_payload()

        unpack_time += time.time() - start
        if 1:
            t = token.t
            if 1:
            # if not torch.allclose(rst, t, rtol=1e-5, atol=1e-8, equal_nan=True):
            #     # print(f"{seq_id=}, {rst.shape=}, {t.shape=}, {token.refcount=}, {dtype=}")
            #     print(f"{t.requires_grad=}")
            #     print(f"{rst=}")
            #     print(f"{t=}")
            #     # print(f"{rst.shape=}, {rst.stride()=}, {rst.storage_offset()=}")
            #     print(f"{t.shape=}, {t.stride()=}, {t.storage_offset()=}")
            #     print(f"{rst.shape=}, {rst.stride()=}, {rst.storage_offset()=}")
                current_stream = torch.cuda.current_stream()
                exp, sm = fs_sp.split(t, current_stream.cuda_stream)
                rec_t = fs_sp.merge(exp, sm, t.shape, t.stride(), t.storage_offset(), t.dtype, current_stream.cuda_stream)

                # print(f"{torch.equal(exp, token.cpu_exp)=}")
                if not torch.equal(exp, token.cpu_exp):
                    print(f"{exp=}")
                    print(f"{token.cpu_exp=}")
                    print(f"{exp.shape=}, {token.cpu_exp.shape=}")
                if not torch.equal(sm, token.sm_bits):
                    print(f"{sm=}")
                    print(f"{token.sm_bits=}")
                # print(f"{torch.equal(sm, token.sm_bits)=}")
                # print(f"{torch.equal(rec_t, rst)=}")
                # # print(f"{torch.equal(rec_t, rec_t1)=}")
                # print(f"{torch.equal(rec_t, t)=}")
                # print(f"{torch.equal(rec, t)=}")
                # print()
                # assert torch.equal(token.exp, exp)
                # diff_mask = rst != t
                # diff_indices = torch.nonzero(diff_mask, as_tuple=False)
                # print(f"{diff_indices.shape=}")
                # print(f"{rst[diff_indices]=}")
                # print(f"{t[diff_indices]=}")
            # assert torch.equal(rst, t)
            assert torch.allclose(rst, t, rtol=1e-5, atol=1e-8, equal_nan=True)


        return rst
        # return rec

    return _pack, _unpack


def build_ptr_tables(model):
    param_ptr2name  = {}
    buffer_ptr2name = {}
    for n, p in model.named_parameters(recurse=True):
        param_ptr2name[p.untyped_storage().data_ptr()] = n          # e.g. "layers.0.mlp.fc1.weight"
    for n, b in model.named_buffers(recurse=True):
        buffer_ptr2name[b.untyped_storage().data_ptr()] = n         # e.g. "layers.0.norm.bias"
    return param_ptr2name, buffer_ptr2name
param_ptr2name, buffer_ptr2name = build_ptr_tables(model)

def classify_leaf(t: torch.Tensor) -> str:
    ptr = t.untyped_storage().data_ptr()
    if ptr in param_ptr2name:
        if t.requires_grad:
            return "trainable_param"
        else:
            return "frozen_param"

    if ptr in buffer_ptr2name:
        return "buffer_const"           # running_mean, running_var, mask…
    if t.dtype == torch.bool:
        return "mask"
    if t.numel() < 1024:
        return "small_const"            # dropout 比例、eps、γβ view 等
    return "input_or_detached"



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
    trainable = (
        p for p in model.parameters()
        if isinstance(p, torch.nn.Parameter) and p.requires_grad
    )
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)

if gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False  # for mistral or LLaMA, 在大多数 HuggingFace 的 decoder-only 模型（如 Mistral、LLaMA）中，use_cache=True 会导致模型跳过中间状态保存，从而禁用 gradient checkpointing。

def measure(n=args.round):
    if args.weight:
        prefetch_first_layer(layer2cps, layer_names)

    for i in range(n):
        print(f"\n\n*************** {i=} ****************")
        global rid
        rid = i
        if args.print_ratio:
            global orig_len, new_len
            orig_len = new_len = 0
        if args.print_time:
            global decomp_time
            decomp_time = 0

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peak0 = torch.cuda.max_memory_allocated(device)

        if args.activation:
            runtime = HookRuntime()
        else:
            runtime = None
        pack_fn, unpack_fn = build_hooks(runtime)
        hook_ctx = saved_tensors_hooks(pack_fn, unpack_fn) if args.hook else contextlib.nullcontext()

        with hook_ctx:
            t0 = time.time()
            if EVAL:
                with torch.no_grad():
                    loss = model(**inputs).loss
            else:
                loss = model(**inputs).loss

            print(f"{loss=}")
            peak1 = torch.cuda.max_memory_allocated(device)
            print(f"Round {i}: peak1={(peak1)/1024/1024:.1f} MB")

            process = psutil.Process(os.getpid())
            cpu_mem_peak = process.memory_info().rss / 1024 / 1024  # 单位: MB
            print(f"Peak CPU memory usage: {cpu_mem_peak:.2f} MB")
            
            if not EVAL:
                if runtime is not None:
                    runtime.flush_prefetch_tail()   #  预取最后一条
                # prefetch_last_layer(layer2cps, layer_names)
                loss.backward()
                # print(f"After backward")
                optimizer.step()
                # print(f"After step")
                optimizer.zero_grad()
                # print(f"After zero grad")
                
            t = (time.time() - t0)
            torch.cuda.synchronize()

            peak = torch.cuda.max_memory_allocated(device)

            print(f"Round {i}: {t*1000:.1f} ms , peak Δ={peak/1024/1024:.1f} MB")

            cpu_mem_peak = process.memory_info().rss / 1024 / 1024  # 单位: MB
            print(f"Peak CPU memory usage: {(cpu_mem_peak-cpu_mem_peak0):.2f} MB")
            if args.print_time:
                print(f"Decompression time: {decomp_time*1000:.1f} ms")
            print(f"Pack time: {pack_time*1000:.1f} ms")
            print(f"\tPrep time 1: {prep_time1*1000:.1f} ms")
            print(f"\tPrep time 2: {prep_time2*1000:.1f} ms")
            print(f"\tPrep time 3: {prep_time3*1000:.1f} ms")
            print(f"\tPrep time 4: {prep_time4*1000:.1f} ms")

            print(f"Unpack time: {unpack_time*1000:.1f} ms")
            print(f"\tPost time 1: {post_time1*1000:.1f} ms")
            print(f"\tPost time 2: {post_time2*1000:.1f} ms")
            print(f"\tPost time 3: {post_time3*1000:.1f} ms")
            if args.print_ratio and orig_len > 0 and new_len > 0:
                print(f"{orig_len=}, {new_len=}, ratio={orig_len/new_len}")
        if EVAL:
            break

        if runtime is not None:
            runtime.reset() 
        runtime = None
        uniq_seq_id = 0
measure()