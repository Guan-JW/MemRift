#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

try:
    from experiments.model_loading.driver_utils import enter_process_group, environment_record, utc_now
except ModuleNotFoundError:
    from driver_utils import enter_process_group, environment_record, utc_now


def validate_checkpoint(method, checkpoint):
    if method == "prequant" and not checkpoint:
        raise ValueError("--checkpoint is required for method prequant")
    if method == "online" and checkpoint:
        raise ValueError("--checkpoint is not used for method online")


def logits_reference(output_path):
    return Path(output_path).with_suffix(".logits.pt").name


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("online", "prequant"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None):
    enter_process_group()
    started_at = utc_now()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_checkpoint(args.method, args.checkpoint)
    except ValueError as error:
        parser.error(str(error))

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    torch.manual_seed(1234)
    if args.method == "online":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map={"": args.device},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint, torch_dtype=torch.bfloat16, device_map={"": args.device}
        )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
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
    input_ids = torch.randint(
        0, model.config.vocab_size, (1, args.sequence_length), device=args.device
    )

    model.eval()
    device_type = torch.device(args.device).type
    with torch.no_grad(), torch.amp.autocast(device_type, dtype=torch.bfloat16):
        validation_logits = model(input_ids=input_ids).logits[:, (-1, 0), :].float().cpu()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logits_path = output_path.with_suffix(".logits.pt")
    torch.save(validation_logits, logits_path)

    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=2e-4
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=args.device)
    started = time.perf_counter()
    with torch.amp.autocast(device_type, dtype=torch.bfloat16):
        loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device=args.device)
    step_seconds = time.perf_counter() - started

    result = {
        "method": args.method,
        "device": args.device,
        "cache_state": "warm",
        "cache_dropped": False,
        "started_at": started_at,
        "finished_at": utc_now(),
        "command": [sys.executable, str(Path(__file__)), *(argv if argv is not None else sys.argv[1:])],
        "environment": environment_record(),
        "loss": loss.item(),
        "training_step_seconds": step_seconds,
        "training_peak_torch_bytes": torch.cuda.max_memory_allocated(device=args.device),
        "logits": logits_reference(output_path),
    }
    with output_path.open("w") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
