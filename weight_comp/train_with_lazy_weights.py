import os, psutil, contextlib, weakref
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json, mmap, queue, threading, struct, argparse, gc, humanize, sys, types, re, pprint, json, mmap
import numpy as np, torch, zstandard as zstd, time
import pandas as pd
from torch import nn
from torch.autograd.graph import saved_tensors_hooks
from torch.utils._pytree import tree_map 

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from accelerate import init_empty_weights
import float_split_stride_pin as fs_sp

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/Mistral-7B-v0.1")
parser.add_argument("--outdir", default="./prepare_weight/zstd_comped_weights")
parser.add_argument("--check_diff", action="store_true",
                    help="同时加载原始模型并比对差值（耗显存）")
parser.add_argument("--hook", action="store_true", help="Run with compression hooks")
parser.add_argument("--debug", action="store_true", help="Run with debug")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)
print(f"\n\n{args.model=}, {args.outdir=}")
rid = 0

# Set GPU!
device = torch.device("cuda:0") 
torch.cuda.set_device(device) 

# --------------------------------------------------------------
#                1. Get Base model and LoRA Adapter 
# --------------------------------------------------------------
MODEL_TYPE = torch.bfloat16
_PTR2CP = weakref.WeakValueDictionary()
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=MODEL_TYPE, device_map={"": 0})

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


# --------------------------------------------------------------
#           2. Model Weight Compression & Injection 
# --------------------------------------------------------------
index = []                                  # <layer name, binary file, shape, dtype>

decomp_time = 0
_tls = threading.local()    # 模块级别，全局只有这一份；属性却会被线程隔离
def decompress_into_pinned(bytes_like, numel):
    """把 exponent 压缩数据直接解压到 pinned Tensor"""
    global decomp_time
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
            bf16 = fs_sp.merge(self._exp_host, self.sm_gpu, self.orig_shape, stride, 0, MODEL_TYPE)
            self._bf16 = bf16.view(self.orig_shape)

            ev = torch.cuda.Event()
            ev.record()          # ensure all copies/kernels before event
            self.CtoD_evt = ev
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

gc.collect()
process = psutil.Process(os.getpid())
cpu_mem_peak0 = process.memory_info().rss / 1024 / 1024  # 单位: MB
print(f"Peak CPU memory usage: {cpu_mem_peak0:.2f} MB")
print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated(device) / 1024 / 1024:.2f} MB")

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
num_workers = min(9, os.cpu_count()) 
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


lid = -1
def pack(t):
    global lid
    lid += 1
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
        # else:
        #     print(f"Address reused by other tensor!!!!")
    return t

def unpack(t):
    global lid
    if isinstance(t, CompressedParam):
        # t.materialize(sync=True)
        # t.set_param()
        if args.debug:
            print(f"[unpack]: layer {lid} is Compressed Param! {t.data.shape=}")
        t._ready_event.wait()
        t.CtoD_evt.synchronize()
        t.set_param()
        if args.debug:
            print(f"[unpack-done]: layer {lid} is Compressed Param! {t.data.shape=}")
        return t.data
    elif isinstance(t, tuple):
        if 0:
            cp, orig_t = t
            print(f"[unpack-waiting]: layer {lid} {cp.shape=}, {orig_t.shape=}, {cp.stride()=}, {orig_t.stride()=}")
            cp._ready_event.wait()
            cp.CtoD_evt.synchronize()
            cp.set_param()
            rec_t = cp.data.as_strided(orig_t.shape, orig_t.stride(), orig_t.storage_offset())
            print(f"[unpack]: layer {lid} {cp.shape=}, {orig_t.shape=}, {cp.stride()=}, {orig_t.stride()=}")
            assert torch.equal(rec_t, orig_t)
        else:
            cp, shape, stride = t
            # print(f"[unpack-waiting]: layer {lid} {cp.shape=}, {shape=}")
            cp._ready_event.wait()
            cp.CtoD_evt.synchronize()
            cp.set_param()
            # print(f"[unpack]: layer {lid} {cp.shape=}, {shape=}")
            # print(f"{cp.data.shape=}, {cp.orig_shape=}, {shape=}, {stride=}")
            rec_t = cp.data.as_strided(shape, stride, 0)    # 就目前的参数来看，没有对参数做中间 view 的，都是整个转置

        return rec_t
    else:
        return t

hook_ctx = saved_tensors_hooks(pack, unpack) if args.hook else contextlib.nullcontext()

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

dataset = load_dataset("timdettmers/openassistant-guanaco")
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

gradient_checkpointing = False
if gradient_checkpointing:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False  # for mistral or LLaMA, 在大多数 HuggingFace 的 decoder-only 模型（如 Mistral、LLaMA）中，use_cache=True 会导致模型跳过中间状态保存，从而禁用 gradient checkpointing。

def measure(n=5):
    prefetch_first_layer(layer2cps, layer_names)

    for i in range(n):
        print(f"\n\n*************** {i=} ****************")
        global rid, decomp_time
        rid = i
        decomp_time = 0

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        peak0 = torch.cuda.max_memory_allocated(device)

        # stats = compressed_param_memory(model, verbose=True)
        # print("Logical CPU bytes  :", humanize.naturalsize(stats["total_cpu"], binary=True))
        # print("Logical GPU bytes  :", humanize.naturalsize(stats["total_gpu"], binary=True))

        with hook_ctx:
            t0 = time.time()
            if EVAL:
                with torch.no_grad():
                    loss = model(**inputs).loss
            else:
                loss = model(**inputs).loss

            print(f"{loss=}")
            t = (time.time() - t0)
            peak1 = torch.cuda.max_memory_allocated(device)
            print(f"Round {i}: peak1={(peak1)/1024/1024:.1f} MB")

            # stats = compressed_param_memory(model, verbose=True)
            # print("Logical CPU bytes  :", humanize.naturalsize(stats["total_cpu"], binary=True))
            # print("Logical GPU bytes  :", humanize.naturalsize(stats["total_gpu"], binary=True))

            if not EVAL:
                # prefetch_last_layer(layer2cps, layer_names)
                loss.backward()
                # print(f"After backward")
                optimizer.step()
                # print(f"After step")
                optimizer.zero_grad()
                # print(f"After zero grad")
                
            torch.cuda.synchronize()

            peak = torch.cuda.max_memory_allocated(device)

            print(f"Round {i}: {t*1000:.1f} ms , peak Δ={peak/1024/1024:.1f} MB")

            process = psutil.Process(os.getpid())
            cpu_mem_peak = process.memory_info().rss / 1024 / 1024  # 单位: MB
            print(f"Peak CPU memory usage: {(cpu_mem_peak-cpu_mem_peak0):.2f} MB")
            print(f"Decompression time: {decomp_time*1000:.1f} ms")
        if EVAL:
            break
measure()