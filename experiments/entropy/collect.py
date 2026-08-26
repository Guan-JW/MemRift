#!/usr/bin/env python3
"""Collect the per-tensor BF16 field entropy reported in paper Table 1."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def shannon_entropy(symbols, alphabet_size):
    import torch

    values = symbols.detach().flatten().to(device="cpu", dtype=torch.int64)
    counts = torch.bincount(values, minlength=alphabet_size).double()
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * torch.log2(probabilities)).sum())


def tensor_entropies(tensor):
    import torch

    contiguous = tensor.detach().to(device="cpu").contiguous()
    if contiguous.dtype != torch.bfloat16:
        raise TypeError("Table 1 entropy collection requires BF16 tensors")
    bits = contiguous.view(torch.uint16).to(torch.int32)
    return {
        "raw_per_8": shannon_entropy(contiguous.view(torch.uint8), 256),
        "sign_per_1": shannon_entropy((bits >> 15) & 1, 2),
        "exponent_per_8": shannon_entropy((bits >> 7) & 0xFF, 256),
        "mantissa_per_7": shannon_entropy(bits & 0x7F, 128),
    }


class MeanEntropies:
    def __init__(self):
        self.count = 0
        self.sums = {key: 0.0 for key in ("raw_per_8", "sign_per_1", "exponent_per_8", "mantissa_per_7")}

    def add(self, tensor):
        values = tensor_entropies(tensor)
        self.count += 1
        for key, value in values.items():
            self.sums[key] += value

    def result(self):
        if self.count == 0:
            raise RuntimeError("no eligible BF16 tensors were observed")
        return {"tensor_count": self.count, **{key: value / self.count for key, value in self.sums.items()}}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-cache", default="/cache/huggingface")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.max_length, args.batch_size, args.rounds) < 1:
        raise SystemExit("length, batch size, and rounds must be positive")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from data_prep import data_prepare

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}
    )
    weights = MeanEntropies()
    for name, parameter in model.named_parameters():
        if parameter.dtype == torch.bfloat16 and "norm" not in name and "embed" not in name:
            weights.add(parameter)

    model = get_peft_model(
        model,
        LoraConfig(
            lora_alpha=16,
            lora_dropout=0.0,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["gate_proj", "up_proj", "down_proj"],
        ),
        autocast_adapter_dtype=True,
    )
    parameter_pointers = {parameter.untyped_storage().data_ptr() for parameter in model.parameters()}
    activations = MeanEntropies()

    def pack(tensor):
        if (
            tensor.dtype == torch.bfloat16
            and tensor.requires_grad
            and not tensor.is_leaf
            and tensor.untyped_storage().data_ptr() not in parameter_pointers
        ):
            activations.add(tensor)
        return tensor

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    texts = data_prepare(
        args.dataset,
        args.batch_size,
        cache_dir=args.dataset_cache,
        revision=args.dataset_revision,
    )
    encoded = tokenizer(
        texts,
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: value.to("cuda:0") for key, value in encoded.items()}
    inputs["labels"] = inputs["input_ids"].clone()
    model.train()
    for _ in range(args.rounds):
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            model(**inputs).loss

    rows = [
        {"model_logical_id": args.model_id, "scope": "W", **weights.result()},
        {"model_logical_id": args.model_id, "scope": "A", **activations.result()},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".csv":
        with args.output.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
