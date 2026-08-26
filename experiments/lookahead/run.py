#!/usr/bin/env python3
"""Run the Figure 11 readiness-lookahead matrix serially and safely."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = ROOT / "experiments" / "gradient_checkpointing" / "run.py"


def load_driver():
    spec = importlib.util.spec_from_file_location("memrift_gradient_driver", DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_values(value):
    values = [int(item) for item in value.split(",")]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("lookahead values must be comma-separated non-negative integers")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--dataset-revision", default="dce01c9b08f87459cf36a430d809084718273017")
    parser.add_argument("--dataset-cache", default="/cache/huggingface")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--lookaheads", type=parse_values, default=parse_values("0,1,2,4,8"))
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--min-available-mb", type=int, default=4096)
    parser.add_argument("--disable-tegrastats", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    return parser.parse_args(argv)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    if args.context < 1 or args.batch_size < 1 or args.rounds < 2:
        raise SystemExit("context and batch size must be positive; rounds must be at least two")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    driver = load_driver()
    driver_args = driver.parse_args([
        "--model", args.model, "--checkpoint", args.checkpoint,
        "--dataset", args.dataset, "--dataset-revision", args.dataset_revision,
        "--dataset-cache", args.dataset_cache, "--results-dir", str(args.results_dir),
        "--matched-context", str(args.context), "--batch-size", str(args.batch_size),
        "--rounds", str(args.rounds), "--warmup-rounds", str(args.warmup_rounds),
        "--timeout-sec", str(args.timeout_sec), "--min-available-mb", str(args.min_available_mb),
        "--variants", "lora", "qlora", "memrift",
        *(["--disable-tegrastats"] if args.disable_tegrastats else []),
        *(["--synthetic-data"] if args.synthetic_data else []),
    ])
    runner = driver.WorkerRunner()
    rows = []
    output_csv = args.results_dir / "readiness_sweep.csv"
    for variant in ("lora", "qlora"):
        driver_args.weight_lookahead = 1
        driver_args.activation_lookahead = 0
        row = driver.run_one(driver_args, runner, variant, args.context, args.rounds, "figure11_baseline")
        rows.append({"model_logical_id": args.model_id, "method": variant, "Nw": "", "Np": "", **row})
        write_csv(output_csv, rows)
    qlora_time = next((row.get("round_time_mean_sec") for row in rows if row["method"] == "qlora"), None)
    for row in rows:
        measured = row.get("round_time_mean_sec")
        row["step_time_vs_qlora"] = measured / qlora_time if measured and qlora_time else ""
    write_csv(output_csv, rows)
    for weight_lookahead in args.lookaheads:
        for activation_lookahead in args.lookaheads:
            driver_args.weight_lookahead = weight_lookahead
            driver_args.activation_lookahead = activation_lookahead
            tag = f"figure11_nw{weight_lookahead}_np{activation_lookahead}"
            row = driver.run_one(driver_args, runner, "memrift", args.context, args.rounds, tag)
            measured = row.get("round_time_mean_sec")
            rows.append({"model_logical_id": args.model_id, "method": "memrift", "Nw": weight_lookahead,
                         "Np": activation_lookahead,
                         "step_time_vs_qlora": measured / qlora_time if measured and qlora_time else "", **row})
            write_csv(output_csv, rows)
    (args.results_dir / "figure11_manifest.json").write_text(json.dumps({
        "model_logical_id": args.model_id, "dataset": args.dataset,
        "dataset_revision": args.dataset_revision, "context": args.context,
        "batch_size": args.batch_size, "lookaheads": args.lookaheads, "row_count": len(rows),
    }, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
