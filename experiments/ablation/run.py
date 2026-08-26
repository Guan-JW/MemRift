#!/usr/bin/env python3
"""Run the Figure 8 weight/activation pipeline ablation."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_driver():
    path = ROOT / "experiments" / "gradient_checkpointing" / "run.py"
    spec = importlib.util.spec_from_file_location("memrift_ablation_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--dataset-revision", default="dce01c9b08f87459cf36a430d809084718273017")
    parser.add_argument("--dataset-cache", default="/cache/huggingface")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--min-available-mb", type=int, default=4096)
    parser.add_argument("--disable-tegrastats", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    driver = load_driver()
    base = driver.parse_args([
        "--model", args.model, "--checkpoint", args.checkpoint,
        "--dataset", args.dataset, "--dataset-revision", args.dataset_revision,
        "--dataset-cache", args.dataset_cache, "--results-dir", str(args.results_dir),
        "--matched-context", str(args.context), "--batch-size", str(args.batch_size),
        "--rounds", str(args.rounds), "--warmup-rounds", str(args.warmup_rounds),
        "--timeout-sec", str(args.timeout_sec), "--min-available-mb", str(args.min_available_mb),
        "--variants", "lora", "memrift_weight", "memrift",
        *(["--disable-tegrastats"] if args.disable_tegrastats else []),
        *(["--synthetic-data"] if args.synthetic_data else []),
    ])
    runner = driver.WorkerRunner()
    rows = []
    for variant in ("lora", "memrift_weight", "memrift"):
        row = driver.run_one(base, runner, variant, args.context, args.rounds, "figure8")
        rows.append({"model_logical_id": args.model_id, "method": variant, **row})
    baseline = rows[0].get("ram_used_MB_max")
    for row in rows:
        peak = row.get("ram_used_MB_max")
        row["peak_memory_reduction_percent"] = 100 * (baseline - peak) / baseline if baseline and peak else ""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.results_dir / "figure8_ablation.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.results_dir / "figure8_manifest.json").write_text(json.dumps({
        "model_logical_id": args.model_id, "dataset": args.dataset,
        "dataset_revision": args.dataset_revision, "context": args.context, "batch_size": args.batch_size,
    }, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
