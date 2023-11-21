import torch, os
from torch.utils.cpp_extension import load
import float_split as mod

# ---------- demo ----------
x_f32  = torch.randn(2_000_000, device="cuda", dtype=torch.float32).contiguous()
exp, sm = mod.split(x_f32)
x_rec   = mod.merge(exp, sm, torch.float32)

assert torch.equal(x_f32, x_rec)      # bit-wise identical
print("FP32 round-trip OK")

x_bf16  = torch.randn(2_000_000, device="cuda", dtype=torch.bfloat16).contiguous()
exp, sm = mod.split(x_bf16)
x_rec   = mod.merge(exp, sm, torch.bfloat16)
assert torch.equal(x_bf16, x_rec)
print("BF16 round-trip OK")
