import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch import nn
from transformers import MistralModel, MistralForCausalLM, MistralConfig
import float_split_stride_pin as fs_sp

import wandb, subprocess, re
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import torch.cuda.nvtx as nvtx

device = torch.device("cuda:0") 
torch.cuda.set_device(device) 


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
        "--max_length", type=int, default=512, help="Input length"
    )
parser.add_argument(
        "--batch_size", type=int, default=1, help="Input batch size"
    )
args = parser.parse_args()


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
        self.stream = stream or torch.cuda.Stream()
        print("init AsyncCompressor")

    # ---------------------------------------------------------------------
    #  Interfaces you need to flesh out
    # ---------------------------------------------------------------------
    def compress_async(self, tensor: torch.Tensor):
        """Kick off async copy/encode.  Returns an opaque token."""
        token: dict = {}
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            # Metadata needed for reconstruction
            token["shape"] = tuple(tensor.shape)
            token["dtype"] = tensor.dtype
            token["device"] = tensor.device
            token["stride"] = tensor.stride()
            token["offset"] = tensor.storage_offset()
            # 🔻 Replace the next line with your `split()` + encoder pipeline
            # token["payload"] = tensor.detach().cpu().numpy()
            token["payload"] = tensor
            cpu_pin_buf, sm_bits = fs_sp.split(tensor, self.stream.cuda_stream)
            evt = self.stream.record_event()   # 拷贝结束事件
            token["pin_exp"] = cpu_pin_buf
            token["gpu_sm"] = sm_bits
            token["split_evt"] = evt

        return token

    def decompress_sync(self, token: dict, device: torch.device):
        """Decode + move back to the target device (blocking)."""
        # Make sure async encode finished
        self.stream.synchronize()
        # 🔻 Replace with decoder + your `unpack()`
        arr = token["payload"]
        shape = token["shape"]
        dtype = token["dtype"]
        stride = token["stride"]
        offset = token["offset"]
        MODEL_TYPE = torch.bfloat16

        token["split_evt"].synchronize()
        with torch.cuda.stream(self.stream):
            bf16 = fs_sp.merge(token["pin_exp"], token["gpu_sm"], shape, stride, 0, MODEL_TYPE, self.stream.cuda_stream)
            ev = self.stream.record_event()
            # self._bf16 = bf16.view(shape)
        ev.synchronize()
        assert torch.equal(arr, bf16)
        return bf16
        # return torch.from_numpy(arr).to(device=device, dtype=token["dtype"])  # (B, hidden)


# -------------------------------------------------------------------------
#  Autograd glue: identity op that frees the saved activation & brings it
#  back (de)compressed only when grad is needed.
# -------------------------------------------------------------------------
class _CompressFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, compressor: AsyncCompressor):
        print(f"_CompressFunction.forward")
        ctx.compressor = compressor
        ctx.token = compressor.compress_async(x)
        # We purposely *don't* save `x` in ctx so PyTorch won't keep it alive.
        return x  # downstream layers use it right away

    @staticmethod
    def backward(ctx, grad_output):
        print(f"_CompressFunction.backward")
        x = ctx.compressor.decompress_sync(ctx.token, grad_output.device)
        # grad_output already matches the forward identity, so just pass it back
        return grad_output, None


# Convenience wrapper so the call site is cleaner
compress_activation = _CompressFunction.apply


# -------------------------------------------------------------------------
#  Plug the compressor into every decoder layer ― zero changes to HF code
# -------------------------------------------------------------------------
class DecoderLayerWrapper(nn.Module):
    """Thin wrapper around a `MistralDecoderLayer` that transparently
    compresses its output activation using `_CompressFunction`."""

    def __init__(self, layer: nn.Module, compressor: AsyncCompressor):
        super().__init__()
        self.layer = layer
        self.compressor = compressor
        # print(f"here")

    def forward(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        if isinstance(out, tuple):
            hidden, *extra = out
            hidden = compress_activation(hidden, self.compressor)
            return (hidden, *extra)
        return compress_activation(out, self.compressor)


def inject_async_compression(model: MistralModel | MistralForCausalLM,
                             compressor: AsyncCompressor):
    """Recursively wrap every decoder layer inside `model` with
    `DecoderLayerWrapper`.  Works for `MistralForCausalLM` and bare `MistralModel`."""

    if hasattr(model, "model"):
        container = model.model
    else:  # bare MistralModel
        container = model

    for i, layer in enumerate(container.layers):
        container.layers[i] = DecoderLayerWrapper(layer, compressor)
    return model


# -------------------------------------------------------------------------
#  Minimal demo ― replace with your real training script (train1.py)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_TYPE = torch.bfloat16
    base = MistralForCausalLM.from_pretrained(args.model, 
                    torch_dtype=MODEL_TYPE, device_map={"": 0})

    compressor = AsyncCompressor()
    print("finish")
    model = inject_async_compression(base, compressor)

    # Dummy pass
    tok = torch.randint(0, base.config.vocab_size, (1, 32), device="cuda")
    out = model(input_ids=tok)
    loss = out.logits.sum()
    loss.backward()

    print("forward/backward succeeded with async compression :) ")
