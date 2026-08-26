#!/usr/bin/env python3
"""Produce AE-grade TinyLlama evidence for paper Tables 2 and 3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "tatsu-lab/alpaca"
DATASET_REVISION = "dce01c9b08f87459cf36a430d809084718273017"
MODEL_ID = "tinyllama-1.1b-chat-v1.0"
EXPECTED = {
    "table2_memrift_step_sec": 19.32,
    "table2_memrift_load_sec": 2.66,
    "table2_activation_share_percent": 91.2,
    "table2_context_expansion_percent": 25.0,
    "table3_lora_peak_gb": 28.28,
    "table3_qlora_reduction_percent": -4.24,
    "table3_memrift_reduction_percent": 14.96,
    "table3_activation_share_percent": 89.5,
    "table3_memrift_storage_gb": 1.4,
}


def load_module(relative_path: str, name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def file_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def directory_sha256(path: Path, excluded_names=()) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file() and candidate.name not in excluded_names):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "little"))
        with item.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def relative_error(actual, expected):
    return abs(actual - expected) / expected if actual is not None and expected else None


def reduction_percent(baseline, value):
    return 100 * (baseline - value) / baseline if baseline and value is not None else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def command_output(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": repr(error)}


def hardware_snapshot(paths=()) -> dict[str, object]:
    l4t = Path("/etc/nv_tegra_release")
    online = Path("/sys/devices/system/cpu/online")
    thermal = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            thermal.append({"type": (zone / "type").read_text().strip(), "millidegrees_c": int((zone / "temp").read_text().strip())})
        except (OSError, TypeError, ValueError):
            pass
    mem_available = None
    try:
        mem_available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    except (OSError, StopIteration, ValueError):
        pass
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "l4t_release": l4t.read_text().strip() if l4t.is_file() else None,
        "online_cpus": online.read_text().strip() if online.is_file() else None,
        "configured_cpu_count": os.cpu_count(),
        "power_mode": os.environ.get("MEMRIFT_NVP_MODEL", "unknown-unrecorded"),
        "thermal_zones": thermal,
        "mem_available_bytes": mem_available,
        "disk_usage": {str(path): dict(zip(("total", "used", "free"), shutil.disk_usage(path))) for path in paths},
        "nvpmodel": command_output(["nvpmodel", "-q"]),
        "container_image_digest": os.environ.get("MEMRIFT_IMAGE_DIGEST", "unknown"),
    }


def validate_inputs(args) -> tuple[dict[str, object], dict[str, object]]:
    prepare = load_module("scripts/prepare_weights.py", "tables23_prepare_weights")
    validate = load_module("scripts/validate_environment.py", "tables23_validate_environment")
    model = prepare.select_model(args.model_manifest, args.model_id)
    prepare.validate_source(model, args.model)
    errors = validate.validate_checkpoint(args.checkpoint)
    errors.extend(validate.validate_dataset_cache(args.dataset_cache, args.dataset, args.dataset_revision))
    metadata = json.loads((args.checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_logical_id") != args.model_id:
        errors.append("training checkpoint logical model does not match --model-id")
    if metadata.get("source_revision") != model["revision"]:
        errors.append("training checkpoint revision does not match the model manifest")
    if metadata.get("zstd_level") != 18:
        errors.append("Table 3 storage requires the Zstd-18 training checkpoint")
    loading_metadata = json.loads((args.loading_prepared / "memrift" / "metadata.json").read_text(encoding="utf-8"))
    if loading_metadata.get("zstd_level") != 3:
        errors.append("Table 2 loading requires a Zstd-3 MemRift checkpoint")
    nf4_config = json.loads((args.loading_prepared / "nf4" / "config.json").read_text(encoding="utf-8"))
    quantization = nf4_config.get("quantization_config", {})
    if quantization.get("bnb_4bit_quant_type") != "nf4" or not quantization.get("bnb_4bit_use_double_quant"):
        errors.append("loading NF4 checkpoint is not double-quantized NF4")
    for method in ("nf4", "memrift"):
        path = args.loading_prepared / method
        receipt_path = path / "preparation.json"
        if not receipt_path.is_file():
            errors.append(f"loading {method} checkpoint lacks preparation.json")
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_receipt = {
            "method": method, "model_logical_id": args.model_id,
            "source_revision": model["revision"],
            "source_weight_sha256": model["sha256"]["model.safetensors"],
        }
        for key, expected in expected_receipt.items():
            if receipt.get(key) != expected:
                errors.append(f"loading {method} receipt {key} does not match the model manifest")
        actual_digest = directory_sha256(path, {"preparation.json"})
        if receipt.get("prepared_directory_sha256") != actual_digest:
            errors.append(f"loading {method} checkpoint content digest does not match its receipt")
    if errors:
        raise ValueError("; ".join(errors))
    receipt = json.loads((args.dataset_cache / "memrift-dataset-receipt.json").read_text(encoding="utf-8"))
    dataset = next(item for item in receipt["datasets"] if item.get("huggingface_id") == args.dataset and item.get("revision") == args.dataset_revision)
    return model, dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-manifest", type=Path, default=ROOT / "manifests/models.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--loading-prepared", type=Path, required=True)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=int, default=2400)
    parser.add_argument("--min-available-mb", type=int, default=4096)
    parser.add_argument("--loading-python", default="/opt/venvs/loading/bin/python")
    parser.add_argument("--loading-runs", type=int, default=5)
    parser.add_argument("--loading-seed", type=int, default=20260821)
    parser.add_argument("--skip-loading", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use reduced non-reportable matrix dimensions")
    parser.add_argument("--synthetic-data", action="store_true", help="Smoke only; output is marked non-reportable")
    parser.add_argument("--disable-tegrastats", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.model_id != MODEL_ID:
        parser.error(f"this AE workflow supports only {MODEL_ID}")
    if args.dataset != DATASET_ID or args.dataset_revision != DATASET_REVISION:
        parser.error("Tables 2-3 AE runs require the pinned Alpaca ID and revision")
    if args.rounds < 2 or not 0 <= args.warmup_rounds < args.rounds:
        parser.error("round and warmup configuration is invalid")
    if args.loading_runs != 5 and not args.skip_loading:
        parser.error("reportable Table 2 loading requires exactly five runs")
    if args.smoke and not (args.synthetic_data and args.skip_loading):
        parser.error("--smoke requires --synthetic-data and --skip-loading")
    return args


def prepare_results(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"results directory is not empty: {path}; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    marker = path / ".incomplete"
    marker.write_text("Tables 2-3 invocation has not completed.\n", encoding="utf-8")
    return marker


def training_args(args, driver):
    return driver.parse_args([
        "--model", str(args.model), "--checkpoint", str(args.checkpoint),
        "--dataset", args.dataset, "--dataset-revision", args.dataset_revision,
        "--dataset-cache", str(args.dataset_cache), "--results-dir", str(args.results_dir / "training"),
        "--matched-context", "2048", "--batch-size", "1",
        "--rounds", str(args.rounds), "--warmup-rounds", str(args.warmup_rounds),
        "--seed", str(args.seed), "--variants", "lora", "qlora", "memrift",
        "--activation-compaction-concurrency", "16", "--activation-decode-concurrency", "4",
        "--weight-materialization-concurrency", "4", "--weight-lookahead", "1",
        "--activation-lookahead", "0", "--activation-backend", "ebc-zstd",
        "--compression-level", "1", "--timeout-sec", str(args.timeout_sec),
        "--min-available-mb", str(args.min_available_mb),
        *(["--synthetic-data"] if args.synthetic_data else []),
        *(["--disable-tegrastats"] if args.disable_tegrastats else []),
    ])


def run_loading(args) -> tuple[str, dict[str, object]]:
    if args.skip_loading:
        return "skipped", {}
    output_root = args.results_dir / "loading"
    command = [
        args.loading_python, str(ROOT / "experiments/model_loading/run_benchmarks.py"),
        "--name", "TinyLlama-1.1B-AE", "--model", str(args.model),
        "--prepared", str(args.loading_prepared), "--output-root", str(output_root),
        "--runs", str(args.loading_runs), "--seed", str(args.loading_seed),
        "--methods", "lora", "qlora-online", "qlora-prequant", "memrift",
        "--worker-timeout-seconds", str(args.timeout_sec),
        "--min-mem-available-bytes", str(args.min_available_mb * 2**20), "--overwrite",
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (args.results_dir / "loading.log").write_text(result.stdout, encoding="utf-8")
    summary_path = output_root / "TinyLlama-1.1B-AE" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if result.returncode == 0 and summary_path.is_file() else {}
    return ("ok" if result.returncode == 0 else "failed"), summary


def main(argv=None):
    args = parse_args(argv)
    marker = prepare_results(args.results_dir, args.overwrite)
    model, dataset = validate_inputs(args)
    snapshot = hardware_snapshot((args.model, args.checkpoint, args.results_dir))
    preflight = {"schema_version": "1.0", "model": model, "dataset": dataset, "hardware": snapshot}
    (args.results_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    if args.preflight_only:
        marker.unlink()
        return 0
    driver = load_module("experiments/gradient_checkpointing/run.py", "tables23_training_driver")
    base = training_args(args, driver)
    runner = driver.WorkerRunner()
    runs = {}
    context = 256 if args.smoke else 2048
    points = ((1, "table3"), (1, "table2")) if args.smoke else ((4, "table3"), (5, "table2"))
    for batch_size, table in points:
        base.batch_size = batch_size
        for method in ("lora", "qlora", "memrift"):
            key = f"{table}_{method}"
            runs[key] = driver.run_one(base, runner, method, context, args.rounds, table)
            (args.results_dir / "training_runs.json").write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n")

    loading_status, loading = run_loading(args)
    t2_lora, t2_qlora, t2_memrift = (runs[f"table2_{name}"] for name in ("lora", "qlora", "memrift"))
    t3_lora, t3_qlora, t3_memrift = (runs[f"table3_{name}"] for name in ("lora", "qlora", "memrift"))
    memrift_load = loading.get("memrift", {}).get("load_to_ready_seconds_median")
    memrift_step = t2_memrift.get("round_time_mean_sec")
    boundary_reproduced = (
        not args.smoke and t3_lora.get("status") == "ok"
        and t2_lora.get("status") == "oom" and t2_memrift.get("status") == "ok"
    )
    expansion = 25.0 if boundary_reproduced else None
    canonical_protocol = (
        not args.synthetic_data and not args.smoke and not args.skip_loading and not args.disable_tegrastats
        and args.rounds == 7 and args.warmup_rounds == 1 and args.loading_runs == 5
        and str(snapshot["container_image_digest"]).startswith("sha256:")
        and snapshot["power_mode"] != "unknown-unrecorded"
    )
    successful_telemetry = all(
        run.get("status") == "ok" and run.get("tegrastats_samples", 0) > 0 and run.get("ram_used_MB_max") is not None
        for run in (t2_memrift, t3_lora, t3_qlora, t3_memrift)
    )
    protocol_reportable = canonical_protocol and successful_telemetry and loading_status == "ok"
    table2 = {
        "model_logical_id": args.model_id, "sequence_length_tokens": context, "batch_size": points[1][0],
        "paper_activation_share_percent": EXPECTED["table2_activation_share_percent"],
        "activation_share_measured": False,
        "expected_context_expansion_over_lora_percent": EXPECTED["table2_context_expansion_percent"],
        "context_expansion_over_lora_percent": expansion, "lora_boundary_reproduced": boundary_reproduced,
        "lora_status": t2_lora.get("status"), "qlora_status": t2_qlora.get("status"),
        "memrift_status": t2_memrift.get("status"), "memrift_step_time_mean_sec": memrift_step,
        "memrift_step_time_std_sec": t2_memrift.get("round_time_std_sec"),
        "lora_load_median_sec": loading.get("lora", {}).get("load_to_ready_seconds_median"),
        "qlora_online_load_median_sec": loading.get("qlora-online", {}).get("load_to_ready_seconds_median"),
        "qlora_prequant_load_median_sec": loading.get("qlora-prequant", {}).get("load_to_ready_seconds_median"),
        "memrift_load_median_sec": memrift_load, "loading_runs": args.loading_runs,
        "loading_status": loading_status, "cache_state": "warm", "cache_dropped": False,
        "expected_memrift_step_sec": EXPECTED["table2_memrift_step_sec"],
        "expected_memrift_load_sec": EXPECTED["table2_memrift_load_sec"],
        "step_relative_error": relative_error(memrift_step, EXPECTED["table2_memrift_step_sec"]),
        "load_relative_error": relative_error(memrift_load, EXPECTED["table2_memrift_load_sec"]),
        "memrift_training_point_completed": not args.smoke and t2_memrift.get("status") == "ok",
        "timing_within_acceptance": bool(memrift_step and relative_error(memrift_step, EXPECTED["table2_memrift_step_sec"]) <= 0.10),
        "loading_within_acceptance": bool(memrift_load and relative_error(memrift_load, EXPECTED["table2_memrift_load_sec"]) <= 0.15),
        "attention_implementation": t2_memrift.get("attention_implementation"),
        "reportable_ae_configuration": protocol_reportable,
    }
    table2["numeric_acceptance_met"] = bool(
        table2["memrift_training_point_completed"] and table2["timing_within_acceptance"]
        and table2["loading_within_acceptance"] and table2["lora_boundary_reproduced"]
    )
    table2["full_paper_row_reproduced"] = False
    lora_peak = t3_lora.get("ram_used_MB_max")
    qlora_peak = t3_qlora.get("ram_used_MB_max")
    memrift_peak = t3_memrift.get("ram_used_MB_max")
    qlora_reduction = reduction_percent(lora_peak, qlora_peak)
    memrift_reduction = reduction_percent(lora_peak, memrift_peak)
    checkpoint_bytes = file_bytes(args.checkpoint)
    table3 = {
        "model_logical_id": args.model_id, "sequence_length_tokens": context, "batch_size": points[0][0],
        "paper_activation_share_percent": EXPECTED["table3_activation_share_percent"],
        "activation_share_measured": False, "lora_status": t3_lora.get("status"),
        "qlora_status": t3_qlora.get("status"), "memrift_status": t3_memrift.get("status"),
        "lora_peak_system_memory_mb": lora_peak, "lora_peak_system_memory_gb_decimal": lora_peak / 1000 if lora_peak else None,
        "qlora_peak_system_memory_mb": qlora_peak, "qlora_peak_system_memory_gb_decimal": qlora_peak / 1000 if qlora_peak else None,
        "memrift_peak_system_memory_mb": memrift_peak, "memrift_peak_system_memory_gb_decimal": memrift_peak / 1000 if memrift_peak else None,
        "qlora_reduction_vs_lora_percent": qlora_reduction,
        "memrift_reduction_vs_lora_percent": memrift_reduction,
        "lora_qlora_storage_bytes": model["expected_bytes"], "lora_qlora_storage_gb_decimal": model["expected_bytes"] / 1e9,
        "memrift_storage_bytes": checkpoint_bytes, "memrift_storage_gb_decimal": checkpoint_bytes / 1e9,
        "expected_lora_peak_gb": EXPECTED["table3_lora_peak_gb"],
        "expected_qlora_reduction_percent": EXPECTED["table3_qlora_reduction_percent"],
        "expected_memrift_reduction_percent": EXPECTED["table3_memrift_reduction_percent"],
        "expected_memrift_storage_gb": EXPECTED["table3_memrift_storage_gb"],
        "peak_relative_error": relative_error(lora_peak / 1000 if lora_peak else None, EXPECTED["table3_lora_peak_gb"]),
        "memrift_reduction_error_points": abs(memrift_reduction - EXPECTED["table3_memrift_reduction_percent"]) if memrift_reduction is not None else None,
        "storage_relative_error": relative_error(checkpoint_bytes / 1e9, EXPECTED["table3_memrift_storage_gb"]),
        "memory_within_acceptance": bool(lora_peak and relative_error(lora_peak / 1000, EXPECTED["table3_lora_peak_gb"]) <= 0.05),
        "reduction_within_acceptance": bool(memrift_reduction is not None and abs(memrift_reduction - EXPECTED["table3_memrift_reduction_percent"]) <= 3.0),
        "attention_implementation": t3_memrift.get("attention_implementation"),
        "reportable_ae_configuration": protocol_reportable,
    }
    table3["qlora_reduction_within_acceptance"] = bool(
        qlora_reduction is not None and abs(qlora_reduction - EXPECTED["table3_qlora_reduction_percent"]) <= 3.0
    )
    table3["numeric_acceptance_met"] = bool(
        table3["memory_within_acceptance"] and table3["reduction_within_acceptance"]
        and table3["qlora_reduction_within_acceptance"]
    )
    table3["full_paper_row_reproduced"] = False
    write_csv(args.results_dir / "table2.csv", [table2])
    write_csv(args.results_dir / "table3.csv", [table3])
    required_execution_ok = (
        t2_memrift.get("status") == "ok" and all(run.get("status") == "ok" for run in (t3_lora, t3_qlora, t3_memrift))
        and loading_status in (("skipped",) if args.smoke else ("ok",))
    )
    manifest = {
        "schema_version": "1.0", "status": "complete" if required_execution_ok else "complete_with_failures",
        "protocol_reportable": protocol_reportable,
        "reported_numeric_acceptance_met": table2["numeric_acceptance_met"] and table3["numeric_acceptance_met"],
        "full_paper_tables_reproduced": False,
        "model": model, "dataset": dataset,
        "hardware": snapshot,
        "training_checkpoint": {"bytes": checkpoint_bytes, "directory_sha256": directory_sha256(args.checkpoint)},
        "loading_prepared": {
            name: {"bytes": file_bytes(args.loading_prepared / name),
                   "directory_sha256": directory_sha256(args.loading_prepared / name)}
            for name in ("nf4", "memrift")
        },
        "protocol": {"rounds": args.rounds, "warmup_rounds": args.warmup_rounds, "seed": args.seed,
                     "tegrastats_interval_ms": 500, "attention": "runtime default; historical FlashAttention path unavailable"},
        "limitations": [
            "The pinned Alpaca revision is an AE snapshot, not a recovered paper revision.",
            "Activation-share percentages are paper references and are not measured by this runtime.",
            "The historical FlashAttention selection is unavailable; timing is labeled with the runtime default.",
            "A safety-stop is not reported as an observed OOM.",
        ],
        "table2": table2, "table3": table3,
    }
    (args.results_dir / "tables23_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    marker.unlink()
    print(json.dumps({"table2": table2, "table3": table3}, indent=2))
    return 0 if required_execution_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
