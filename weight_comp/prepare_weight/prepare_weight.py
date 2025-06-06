# === prepare_compressed_weights.py =================================================
import torch, zstandard as zstd, numpy as np, struct, json, os, argparse
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/opt/models/hf/Mistral-7B-v0.1")
parser.add_argument("--outdir", default="./zstd_comped_weights")
parser.add_argument("--level", default=1)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

compressor = zstd.ZstdCompressor(level=int(args.level))
index = []                                  # <layer name, binary file, shape, dtype>

print(f"{args.model=}, {args.outdir=}")
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
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

# for name, param in model.named_parameters():
#     if param is None or not isinstance(param, torch.nn.Parameter):
#         print("Not parameter!")
#     if not param.requires_grad:
#         print(f"[FROZEN] {name} shape={param.shape} requires_grad={param.requires_grad} is_leaf={param.is_leaf}")
#     elif "lora_" in name:
#         print(f"[LoRA]   {name} shape={param.shape} requires_grad={param.requires_grad} is_leaf={param.is_leaf}")
#     else:
#         print(f"[TRAIN]  {name} shape={param.shape} requires_grad={param.requires_grad} is_leaf={param.is_leaf}")


comp_time = 0
for name, w in model.named_parameters():
    if w.requires_grad or w.dtype != torch.bfloat16:
        continue                                # 只处理冻结 bf16

    w_cpu = w.detach().cpu().contiguous()
    arr_u16_t = w_cpu.view(torch.uint16)
    arr = arr_u16_t.numpy()

    sign = (arr >> 15) & 0x1
    mant = arr & 0x007F
    sign_mant = ((sign << 7) | mant).astype(np.uint8)
    # sign_mant = (arr & 0x807f).astype(np.uint16) # bit15 sign + bit6-0 mantissa
    exp       = ((arr >> 7) & 0x00ff).astype(np.uint8)

    fn = f"{len(index):05d}.bin"
    path = os.path.join(args.outdir, fn)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", arr.size))          # 4B 元素个数
        f.write(sign_mant.tobytes())                   # 2 bytes/elem
        t0 = time.time()
        comped = compressor.compress(exp.tobytes())
        comp_time += time.time() - t0
        f.write(comped)          # 压缩后 exponent

    index.append(dict(
        name=name,
        file=fn,
        shape=list(w.shape),
    ))

json.dump(index, open(os.path.join(args.outdir, "index.json"), "w"))
print(f"✅  wrote {len(index)} frozen tensors → “{args.outdir}”, compression time: {comp_time*1000:.2f} ms")
