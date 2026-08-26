#!/usr/bin/env python3
"""Run the paper Table 6 activation-compression backend comparison."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ("lz4", "zstd", "ebc-lz4", "ebc-zstd")


def load_driver():
    path = ROOT / "experiments" / "gradient_checkpointing" / "run.py"
    spec = importlib.util.spec_from_file_location("memrift_table6_base", path)
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
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=2400)
    parser.add_argument("--min-available-mb", type=int, default=4096)
    parser.add_argument("--disable-tegrastats", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.compression_level <= 16:
        parser.error("Table 6 requires a compression level valid for both LZ4 and Zstd (0..16)")
    return args


def table_row(model_id, backend, result, baseline_time):
    step_time = result.get("round_time_mean_sec")
    peak_bytes = result.get("ram_used_bytes_max")
    return {
        "model_logical_id": model_id,
        "backend": backend,
        "weight_backend": "ebc-zstd",
        "status": result.get("status"),
        "compression_ratio": result.get("activation_compression_ratio"),
        "activation_original_bytes": result.get("activation_original_bytes"),
        "activation_stored_bytes": result.get("activation_stored_bytes"),
        "peak_system_memory_gib": peak_bytes / 2**30 if peak_bytes is not None else "",
        "round_time_mean_sec": step_time,
        "normalized_step_time": step_time / baseline_time if step_time and baseline_time else "",
        "baseline_round_time_mean_sec": baseline_time,
        "run_dir": result.get("run_dir"),
    }


def main(argv=None):
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    driver = load_driver()
    common = [
        "--model", args.model, "--checkpoint", args.checkpoint,
        "--dataset", args.dataset, "--dataset-revision", args.dataset_revision,
        "--dataset-cache", args.dataset_cache, "--results-dir", str(args.results_dir),
        "--matched-context", str(args.context), "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--rounds", str(args.rounds), "--warmup-rounds", str(args.warmup_rounds),
        "--compression-level", str(args.compression_level),
        "--timeout-sec", str(args.timeout_sec), "--min-available-mb", str(args.min_available_mb),
        "--activation-compaction-concurrency", "16",
        "--activation-decode-concurrency", "4",
        "--weight-materialization-concurrency", "4",
        "--variants", "lora", "memrift",
        *(["--disable-tegrastats"] if args.disable_tegrastats else []),
        *(["--synthetic-data"] if args.synthetic_data else []),
    ]
    base = driver.parse_args(common)
    runner = driver.WorkerRunner()
    baseline = driver.run_one(base, runner, "lora", args.context, args.rounds, "table6_baseline")
    baseline_time = baseline.get("round_time_mean_sec") if baseline.get("status") == "ok" else None
    rows = []
    for backend in BACKENDS:
        base.activation_backend = backend
        result = driver.run_one(base, runner, "memrift", args.context, args.rounds, f"table6_{backend}")
        rows.append(table_row(args.model_id, backend, result, baseline_time))
        with (args.results_dir / "table6_backends.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "model_logical_id": args.model_id,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "context": args.context,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "compression_level": args.compression_level,
        "scope": "Activation backend comparison with static weights held at EBC-Zstd.",
        "historical_discrepancy": "Located historical runs used unpinned timdettmers/openassistant-guanaco; the AE default is pinned Alpaca.",
        "baseline": baseline,
    }
    (args.results_dir / "table6_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0 if baseline.get("status") == "ok" and all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
