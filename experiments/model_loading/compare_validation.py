#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

try:
    from experiments.model_loading.driver_utils import enter_process_group, environment_record, utc_now
except ModuleNotFoundError:
    from driver_utils import enter_process_group, environment_record, utc_now


def resolve_record_path(record_file, artifact_path):
    path = Path(artifact_path)
    return path if path.is_absolute() else Path(record_file).parent / path


def compare(online, prequant, online_file, prequant_file):
    import torch

    online_logits = torch.load(
        resolve_record_path(online_file, online["logits"]), map_location="cpu", weights_only=True
    )
    prequant_logits = torch.load(
        resolve_record_path(prequant_file, prequant["logits"]), map_location="cpu", weights_only=True
    )
    difference = (online_logits - prequant_logits).abs()
    return {
        "logits_allclose_atol_1e-5": torch.allclose(
            online_logits, prequant_logits, atol=1e-5, rtol=0
        ),
        "logits_max_absolute_difference": difference.max().item(),
        "logits_mean_absolute_difference": difference.mean().item(),
        "training_step_seconds_ratio_prequant_over_online": prequant["training_step_seconds"]
        / online["training_step_seconds"],
        "training_peak_ratio_prequant_over_online": prequant["training_peak_torch_bytes"]
        / online["training_peak_torch_bytes"],
        "loss_absolute_difference": abs(prequant["loss"] - online["loss"]),
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", required=True)
    parser.add_argument("--prequant", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    enter_process_group()
    started_at = utc_now()
    args = build_parser().parse_args(argv)
    with open(args.online) as source:
        online = json.load(source)
    with open(args.prequant) as source:
        prequant = json.load(source)
    result = compare(online, prequant, args.online, args.prequant)
    result.update(
        {
            "cache_state": "warm",
            "cache_dropped": False,
            "started_at": started_at,
            "finished_at": utc_now(),
            "command": [sys.executable, str(Path(__file__)), *(argv if argv is not None else sys.argv[1:])],
            "environment": environment_record(),
        }
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
