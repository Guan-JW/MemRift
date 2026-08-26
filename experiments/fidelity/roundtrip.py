#!/usr/bin/env python3
"""Perform direct bitwise EBC-Zstd round-trip checks for paper Table 5."""

import argparse
import json
import sys
from pathlib import Path


def bitwise_equal(left, right):
    import torch

    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    return torch.equal(left.detach().contiguous().view(torch.uint8), right.detach().contiguous().view(torch.uint8))


class FidelityCounts:
    def __init__(self):
        self.tensors = 0
        self.bytes = 0
        self.mismatches = 0

    def add(self, source, restored):
        self.tensors += 1
        self.bytes += source.nbytes
        self.mismatches += int(not bitwise_equal(source, restored))

    def as_dict(self):
        return {"tensors": self.tensors, "bytes": self.bytes, "mismatches": self.mismatches}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--dataset-revision", default="dce01c9b08f87459cf36a430d809084718273017")
    parser.add_argument("--dataset-cache", default="/cache/huggingface")
    parser.add_argument("--synthetic-data", action="store_true", help="Use local text for codec smoke tests only")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.max_length, args.batch_size, args.steps) < 1:
        raise SystemExit("length, batch size, and steps must be positive")

    import torch
    import zstandard as zstd
    import float_split_stride as split_stride
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from data_prep import data_prepare

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    model = get_peft_model(model, LoraConfig(
        lora_alpha=16, lora_dropout=0.0, r=16, bias="none", task_type="CAUSAL_LM",
        target_modules=["gate_proj", "up_proj", "down_proj"],
    ), autocast_adapter_dtype=True)
    optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-5)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if args.synthetic_data:
        texts = ["Memory compression preserves tensor values exactly. " * 64] * args.batch_size
    else:
        texts = data_prepare(args.dataset, args.batch_size, cache_dir=args.dataset_cache, revision=args.dataset_revision)
    encoded = tokenizer(texts, max_length=args.max_length, padding="max_length", truncation=True, return_tensors="pt")
    inputs = {key: value.to("cuda:0") for key, value in encoded.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    compressor = zstd.ZstdCompressor(level=args.zstd_level, write_checksum=True)
    decompressor = zstd.ZstdDecompressor()
    weights = FidelityCounts()
    activations = FidelityCounts()

    def roundtrip(tensor, counts):
        stream = torch.cuda.current_stream()
        exponent, sign_mantissa = split_stride.split(tensor, stream.cuda_stream)
        stream.synchronize()
        raw_exponent = exponent.numpy().tobytes()
        compressed = compressor.compress(raw_exponent)
        decoded = decompressor.decompress(compressed, max_output_size=len(raw_exponent))
        decoded_exponent = torch.frombuffer(bytearray(decoded), dtype=torch.uint8).pin_memory()
        restored = split_stride.merge(decoded_exponent, sign_mantissa, tensor.shape, tensor.stride(),
                                      tensor.storage_offset(), tensor.dtype, stream.cuda_stream)
        stream.synchronize()
        counts.add(tensor, restored)
        split_stride.release_cuda(sign_mantissa)

    model.train()
    step_rows = []
    for step in range(args.steps):
        weight_start = weights.tensors
        activation_start = activations.tensors
        for parameter in model.parameters():
            if parameter.dtype in (torch.bfloat16, torch.float32) and not parameter.requires_grad:
                roundtrip(parameter, weights)
        seen = set()

        def pack(tensor):
            key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tensor.nbytes)
            if tensor.dtype in (torch.bfloat16, torch.float32) and tensor.requires_grad and not tensor.is_leaf and key not in seen:
                seen.add(key)
                roundtrip(tensor, activations)
            return tensor

        optimizer.zero_grad(set_to_none=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = model(**inputs).loss
        loss.backward()
        optimizer.step()
        step_rows.append({
            "step": step + 1,
            "weight_tensors": weights.tensors - weight_start,
            "activation_tensors": activations.tensors - activation_start,
            "cumulative_mismatches": weights.mismatches + activations.mismatches,
        })
        if step_rows[-1]["cumulative_mismatches"]:
            break

    result = {
        "schema_version": "1.0", "model_logical_id": args.model_id,
        "dataset": args.dataset, "dataset_revision": args.dataset_revision, "synthetic_data": args.synthetic_data,
        "steps_requested": args.steps, "steps_completed": len(step_rows), "zstd_level": args.zstd_level,
        "weights": weights.as_dict(), "activations": activations.as_dict(),
        "tensor_mismatches": weights.mismatches + activations.mismatches, "per_step": step_rows,
        "scope": "direct EBC-Zstd codec round trips during LoRA training; LM-Eval is a separate phase",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["tensor_mismatches"] == 0 and len(step_rows) == args.steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
