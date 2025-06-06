#!/usr/bin/env python3
# ==============================================================
#  restore_from_compressed.py
#
#  * 加载压缩权重目录（prepare_bf16_weights.py 生成）
#  * 解压复原 → 写回模型
#  * 简单 forward 验证
# ==============================================================
import os, json, struct, argparse, time, zstandard as zstd, numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# ----------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/Mistral-7B-v0.1")
parser.add_argument("--compdir", default="./zstd_comped_weights",
                    help="prepare_bf16_weights.py 生成的目录")
parser.add_argument("--test_text", default="Hello, world",
                    help="随便来一句做 forward 验证")
parser.add_argument("--check_diff", action="store_true",
                    help="同时加载原始模型并比对差值（耗显存）")
args = parser.parse_args()

# ----------------- 1. 加载模型 + LoRA -----------------
print("→ loading base model …")
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
)
peft_cfg = LoraConfig(
    lora_alpha=16, lora_dropout=0.0, r=16, bias="none",
    task_type="CAUSAL_LM", target_modules=["gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_cfg, autocast_adapter_dtype=True)
model.eval()                  # 只测试推理

# ----------------- 2. 恢复冻结权重 -----------------
print("→ restoring frozen bf16 parameters …")
dctx  = zstd.ZstdDecompressor()
index = json.load(open(os.path.join(args.compdir, "index.json")))
decomp_time = 0
for item in index:
    bin_path = os.path.join(args.compdir, item["file"])
    with open(bin_path, "rb") as f:
        numel, = struct.unpack("<I", f.read(4))
        sign_mant = np.frombuffer(f.read(numel),   dtype=np.uint8)   # 1B/elem
        exp_comp  = f.read()                                         # rest bytes
    t0 = time.time()
    decomp = dctx.decompress(exp_comp, numel)
    decomp_time += time.time() - t0

    exp_u8 = np.frombuffer(decomp, dtype=np.uint8)
    # ---- uint8 → uint16 拼接 ----
    # ---- 正确复原：把 sign 从 bit7 移到 bit15 ----
    sm_u16   = sign_mant.astype(np.uint16)
    sign_u16 = (sm_u16 & 0x80) << 8          # bit7 → bit15
    mant_u16 =  sm_u16 & 0x7F                # bit0–6 保持
    exp_u16  =  exp_u8.astype(np.uint16) << 7

    recon_u16  = sign_u16 | exp_u16 | mant_u16
    recon_bf16 = torch.from_numpy(recon_u16).view(torch.bfloat16)
    recon_bf16 = recon_bf16.view(item["shape"])                    # reshape

    # ---- 覆写到模型 ----
    module, _, attr = item["name"].rpartition(".")
    getattr(dict(model.named_modules())[module], attr).data.copy_(recon_bf16)

print(f"✅ restored {len(index)} tensors")

# ----------------- 3. (可选) 与原始模型对比 -----------------
if args.check_diff:
    print("→ loading a fresh copy for diff …")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    ref_model = get_peft_model(ref_model, peft_cfg, autocast_adapter_dtype=True)

    max_abs = 0.0
    for n, p in ref_model.named_parameters():
        q = dict(model.named_parameters())[n]
        if not p.requires_grad:               # 只比冻结权重
            diff = (p.data - q.data).abs().max().item()
            max_abs = max(max_abs, diff)
    print(f"🔎 max |Δ| on frozen params = {max_abs:e}")

# ----------------- 4. 一次 forward 自验 -----------------
tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
tok.pad_token = tok.unk_token
tok.padding_side = "left"

inputs = tok(args.test_text, return_tensors="pt").to(torch.bfloat16)
with torch.no_grad():
    outs = model(**{k: v.to(model.device) for k, v in inputs.items()})
    logits_sum = outs.logits.float().sum().item()
print(f"🤖 forward OK, logits sum = {logits_sum:.4f}, decompression time = {decomp_time*1000:.2f} ms")
