#!/usr/bin/env python3
"""Run and summarize a balanced LoRA/QLoRA/MemRift memory comparison."""

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path


METHODS = ("lora", "qlora", "memrift")
ROOT = Path(__file__).resolve().parents[1]
METHOD_CONTRACTS = {
    "lora": {"finetune_type": "lora", "hook": False, "weight": False, "weight_async": False, "activation": False, "act_async": False},
    "qlora": {"finetune_type": "qlora", "hook": False, "weight": False, "weight_async": False, "activation": False, "act_async": False},
    "memrift": {"finetune_type": "lora", "hook": True, "weight": True, "weight_async": True, "activation": True, "act_async": True},
}
METRIC_FIELDS = {
    "peak_process_rss_bytes": ("peak_process_rss_bytes", "process_rss_bytes_max"),
    "peak_torch_allocated_bytes": ("peak_torch_allocated_bytes", "round_peak_gpu_alloc_bytes_max"),
    "peak_torch_reserved_bytes": ("peak_torch_reserved_bytes", "round_peak_gpu_reserved_bytes_max"),
    "minimum_system_available_bytes": ("minimum_system_available_bytes", "system_available_bytes_min"),
    "round_time_mean_sec": ("round_time_mean_sec",),
    "cpu_util_mean": ("cpu_util_mean",),
    "gpu_util_mean": ("gpu_util_mean",),
}


def method_order(repetition):
    offset = repetition % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def first_number(row, fields):
    for field in fields:
        value = row.get(field)
        if isinstance(value, (int, float)):
            return value
    return None


def peak_system_used_bytes(row):
    value = first_number(row, ("peak_system_used_bytes", "ram_used_bytes_max"))
    if value is not None:
        return int(value)
    value = row.get("ram_used_MB_max")
    return int(float(value) * 1024**2) if isinstance(value, (int, float)) else None


def row_matches_profile(row, profile, method):
    return (
        row.get("context") == profile["context"]
        and row.get("batch_size") == profile["batch_size"]
        and row.get("rounds") == profile["rounds"]
        and row.get("warmup_rounds") == profile["warmup_rounds"]
        and row.get("dataset") == profile["dataset"]
        and row.get("dataset_revision") == profile["dataset_revision"]
        and row.get("synthetic_data") is False
        and row.get("gradient_checkpointing") is False
        and row.get("seed") == 42
        and row.get("activation_backend") == "ebc-zstd"
        and row.get("activation_compression_level") == 1
        and row.get("weight_lookahead") == 1
        and row.get("activation_lookahead") == 0
        and all(row.get(key) == value for key, value in METHOD_CONTRACTS[method].items())
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_payload_sha256(root, index):
    digest = hashlib.sha256()
    names = ["index.json"] + sorted(item["file"] for item in index)
    for name in names:
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def validate_inputs(args):
    manifest = json.loads((ROOT / "manifests/models.json").read_text(encoding="utf-8"))
    matches = [item for item in manifest["models"] if item.get("logical_name") == args.name]
    if len(matches) != 1:
        raise SystemExit(f"model manifest does not contain exactly one {args.name!r} entry")
    model = matches[0]
    missing = [name for name in model["expected_files"] if not (args.model / name).is_file()]
    if missing:
        raise SystemExit(f"model directory is missing manifest files: {missing}")
    actual_size = sum(
        path.stat().st_size for path in args.model.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(args.model).parts
    )
    if actual_size != model["expected_bytes"]:
        raise SystemExit(f"model directory size is {actual_size}, expected {model['expected_bytes']}")
    for relative, expected in model.get("sha256", {}).items():
        if sha256(args.model / relative) != expected:
            raise SystemExit(f"model SHA-256 mismatch: {relative}")

    try:
        metadata = json.loads((args.checkpoint / "metadata.json").read_text(encoding="utf-8"))
        index = json.loads((args.checkpoint / "index.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid checkpoint metadata: {exc}") from exc
    if (args.checkpoint / ".incomplete").exists() or not isinstance(index, list) or not index:
        raise SystemExit("checkpoint is incomplete or has an invalid index")
    expected_metadata = {
        "schema_version": "1.0",
        "format": model["checkpoint_format"],
        "model_logical_id": args.name,
        "source_revision": model["revision"],
        "zstd_level": model["zstd_level"],
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise SystemExit("checkpoint metadata does not match the pinned model")
    if metadata.get("tensor_count") != len(index):
        raise SystemExit("checkpoint tensor count does not match its index")
    if set(metadata) != {
        "schema_version", "format", "model_logical_id", "source_revision",
        "zstd_level", "tensor_count", "compression_seconds",
    } or not isinstance(metadata.get("compression_seconds"), (int, float)) or metadata["compression_seconds"] < 0:
        raise SystemExit("checkpoint metadata does not match the checkpoint schema")
    names = set()
    files = set()
    for item in index:
        relative = Path(item.get("file", "")) if isinstance(item, dict) else Path()
        required = {"name", "file", "shape", "dtype", "scheme"}
        allowed = required | {"stride", "storage_offset"}
        valid_shape = isinstance(item, dict) and isinstance(item.get("shape"), list) and all(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in item["shape"]
        )
        valid = (
            isinstance(item, dict) and required.issubset(item) and set(item).issubset(allowed)
            and isinstance(item.get("name"), str) and item["name"] and item["name"] not in names
            and relative.name and str(relative) not in files and not relative.is_absolute() and ".." not in relative.parts
            and (args.checkpoint / relative).is_file() and valid_shape
            and isinstance(item.get("dtype"), str) and item["dtype"]
            and item.get("scheme") in {"split_zstd", "raw_torch"}
            and ("stride" not in item or isinstance(item["stride"], list) and len(item["stride"]) == len(item["shape"]) and all(isinstance(value, int) and not isinstance(value, bool) for value in item["stride"]))
            and ("storage_offset" not in item or isinstance(item["storage_offset"], int) and not isinstance(item["storage_offset"], bool) and item["storage_offset"] >= 0)
        )
        if not valid:
            raise SystemExit("checkpoint index does not match the checkpoint schema")
        names.add(item["name"])
        files.add(str(relative))
    checkpoint_digest = checkpoint_payload_sha256(args.checkpoint, index)
    if checkpoint_digest != model["checkpoint_payload_sha256"]:
        raise SystemExit("checkpoint payload digest does not match the model manifest")

    try:
        receipt = json.loads((args.cache / "memrift-dataset-receipt.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid dataset receipt: {exc}") from exc
    datasets = receipt.get("datasets", []) if isinstance(receipt, dict) else []
    matches = [item for item in datasets if item.get("huggingface_id") == args.dataset and item.get("revision") == args.dataset_revision]
    if len(matches) != 1 or not matches[0].get("fingerprint") or matches[0].get("num_rows", 0) < 1:
        raise SystemExit("dataset receipt does not bind the requested dataset revision")
    if not any(args.cache.rglob("*.arrow")):
        raise SystemExit("dataset cache contains no Arrow data")
    return {
        "model_revision": model["revision"],
        "checkpoint_model_logical_id": metadata["model_logical_id"],
        "checkpoint_source_revision": metadata["source_revision"],
        "checkpoint_format": metadata["format"],
        "checkpoint_zstd_level": metadata["zstd_level"],
        "checkpoint_payload_sha256": checkpoint_digest,
        "dataset_fingerprint": matches[0]["fingerprint"],
        "dataset_rows": matches[0]["num_rows"],
    }


def summarize(rows, profile, image_digest, source_revision, runtime_source_revision, orders, input_validation=None):
    normalized = []
    methods = {}
    for row in rows:
        item = dict(row)
        item["peak_system_used_bytes"] = peak_system_used_bytes(item)
        for output_name, fields in METRIC_FIELDS.items():
            item[output_name] = first_number(item, fields)
        normalized.append(item)

    for method in METHODS:
        selected = [row for row in normalized if row.get("variant") == method]
        successful = []
        for repetition in range(1, profile["repetitions"] + 1):
            candidates = [row for row in selected if row.get("repetition") == repetition]
            if len(candidates) != 1:
                continue
            row = candidates[0]
            if (
                row.get("status") == "ok"
                and row.get("peak_system_used_bytes") is not None
                and row.get("tegrastats_samples", 0) > 0
                and row_matches_profile(row, profile, method)
            ):
                successful.append(row)
        summary = {
            "requested_runs": profile["repetitions"],
            "successful_runs": len(successful),
            "peak_system_used_bytes": [row["peak_system_used_bytes"] for row in successful],
            "peak_system_used_bytes_median": None,
        }
        if successful:
            summary["peak_system_used_bytes_median"] = statistics.median(
                row["peak_system_used_bytes"] for row in successful
            )
        for output_name, fields in METRIC_FIELDS.items():
            values = [first_number(row, fields) for row in successful]
            values = [value for value in values if value is not None]
            summary[f"{output_name}_median"] = statistics.median(values) if values else None
        methods[method] = summary

    memrift_peak = methods["memrift"]["peak_system_used_bytes_median"]
    reductions = {}
    for baseline in ("lora", "qlora"):
        baseline_peak = methods[baseline]["peak_system_used_bytes_median"]
        reductions[baseline] = (
            (baseline_peak - memrift_peak) / baseline_peak * 100
            if baseline_peak and memrift_peak is not None else None
        )
    observed_orders = []
    for repetition in range(1, profile["repetitions"] + 1):
        current = sorted(
            (row for row in normalized if row.get("repetition") == repetition),
            key=lambda row: row.get("position", 0),
        )
        observed_orders.append(tuple(row.get("variant") for row in current))
    order_valid = observed_orders == list(orders)
    complete = order_valid and all(methods[method]["successful_runs"] == profile["repetitions"] for method in METHODS)
    minimum = profile["minimum_reduction_percent"]
    claim_supported = complete and all(
        value is not None and value > minimum for value in reductions.values()
    )
    return {
        "schema_version": "1.0",
        "status": "complete" if complete else "complete_with_failures",
        "profile": profile,
        "container_image_digest": image_digest,
        "orchestrator_source_revision": source_revision,
        "runtime_source_revision": runtime_source_revision,
        "input_validation": input_validation,
        "primary_metric": "peak whole-system used RAM sampled by tegrastats",
        "method_orders": [list(order) for order in orders],
        "method_order_valid": order_valid,
        "methods": methods,
        "memrift_reduction_percent": reductions,
        "claim_supported": claim_supported,
        "claim_scope": (
            "MemRift has lower median peak whole-system RAM than successful matched LoRA and "
            "online-QLoRA runs for this exact reviewer configuration."
            if claim_supported else
            "The completed evidence does not support a lower-memory claim for this exact configuration."
        ),
        "runs": normalized,
    }


def build_container_command(args, comparison_name, repetition, order, runtime_source_revision):
    container_results = f"/results/{comparison_name}/rep-{repetition + 1:02d}"
    return [
        args.docker, "run", "--rm", "--runtime=nvidia", "--network=none", "--ipc=host",
        "--mount", "type=bind,src=/usr/bin/tegrastats,dst=/usr/bin/tegrastats,readonly",
        "--mount", f"type=bind,src={args.model},dst=/models/model,readonly",
        "--mount", f"type=bind,src={args.checkpoint},dst=/checkpoints/model,readonly",
        "--mount", f"type=bind,src={args.cache},dst=/cache/huggingface",
        "--mount", f"type=bind,src={args.results_root},dst=/results",
        "-e", f"MEMRIFT_IMAGE_DIGEST={args.image_digest}",
        "-e", f"MEMRIFT_GIT_REVISION={runtime_source_revision}",
        args.image, "training", "/workspace/experiments/gradient_checkpointing/run.py",
        "--model", "/models/model", "--checkpoint", "/checkpoints/model",
        "--dataset", args.dataset, "--dataset-revision", args.dataset_revision,
        "--dataset-cache", "/cache/huggingface", "--results-dir", container_results,
        "--matched-context", str(args.context), "--batch-size", str(args.batch_size),
        "--rounds", str(args.rounds), "--warmup-rounds", str(args.warmup_rounds),
        "--variants", *order,
        "--activation-compaction-concurrency", "16", "--activation-decode-concurrency", "4",
        "--weight-materialization-concurrency", "4", "--weight-lookahead", "1",
        "--activation-lookahead", "0", "--activation-backend", "ebc-zstd",
        "--compression-level", "1", "--timeout-sec", str(args.timeout_seconds),
        "--min-available-mb", str(args.min_available_mb),
    ]


def write_runs_csv(path, rows):
    fields = [
        "repetition", "position", "variant", "status", "context", "batch_size", "rounds",
        "warmup_rounds", "peak_system_used_bytes", "minimum_system_available_bytes",
        "peak_process_rss_bytes", "peak_torch_allocated_bytes", "round_time_mean_sec",
        "cpu_util_mean", "gpu_util_mean", "tegrastats_samples", "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--name", default="tinyllama-1.1b-chat-v1.0")
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--dataset-revision", default="dce01c9b08f87459cf36a430d809084718273017")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--minimum-reduction-percent", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--min-available-mb", type=int, default=4096)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("context", "batch_size", "rounds", "repetitions", "timeout_seconds", "min_available_mb"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmup_rounds < 0 or args.warmup_rounds >= args.rounds:
        parser.error("--warmup-rounds must be non-negative and less than --rounds")
    if args.minimum_reduction_percent < 0:
        parser.error("--minimum-reduction-percent must be non-negative")
    if args.repetitions != 3:
        parser.error("--repetitions must be exactly 3 for the balanced reviewer profile")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", args.name):
        parser.error("--name must be a lowercase filename-safe identifier")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_digest):
        parser.error("--image-digest must be sha256 followed by 64 lowercase hexadecimal characters")
    if not args.image.endswith(f"@{args.image_digest}"):
        parser.error("--image must be pinned to the exact value supplied by --image-digest")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("--source-revision must be a 40-character lowercase Git revision")
    for path in (args.model, args.checkpoint, args.cache):
        if not path.is_dir():
            parser.error(f"input directory does not exist: {path}")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.results_root = args.results_root.resolve()
    comparison_name = f"memory-comparison-{args.name}"
    output_dir = args.results_root / comparison_name
    if output_dir.exists() and any(output_dir.iterdir()) and not args.dry_run:
        raise SystemExit(f"refusing to overwrite nonempty result directory: {output_dir}")
    orders = [method_order(index) for index in range(args.repetitions)]
    if args.dry_run:
        commands = [
            build_container_command(args, comparison_name, index, order, "runtime-source-revision-from-image-label")
            for index, order in enumerate(orders)
        ]
        print(json.dumps(commands, indent=2))
        return 0

    input_validation = validate_inputs(args)
    inspected = subprocess.run(
        [args.docker, "image", "inspect", args.image, "--format", "{{index .Config.Labels \"org.opencontainers.image.revision\"}}"],
        check=False, capture_output=True, text=True,
    )
    runtime_source_revision = inspected.stdout.strip()
    if inspected.returncode != 0:
        raise SystemExit(f"could not inspect pinned runtime image: {inspected.stderr.strip()}")
    if not re.fullmatch(r"[0-9a-f]{40}", runtime_source_revision):
        raise SystemExit("runtime image has no valid org.opencontainers.image.revision label")
    commands = [
        build_container_command(args, comparison_name, index, order, runtime_source_revision)
        for index, order in enumerate(orders)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    returncodes = [subprocess.run(command, check=False).returncode for command in commands]
    rows = []
    for repetition, order in enumerate(orders):
        result_file = output_dir / f"rep-{repetition + 1:02d}" / "gc_experiments.json"
        if not result_file.is_file():
            continue
        current = json.loads(result_file.read_text(encoding="utf-8"))
        for position, row in enumerate(current):
            row["repetition"] = repetition + 1
            row["position"] = position + 1
            row["run_dir"] = str(Path(f"rep-{repetition + 1:02d}") / row["run_dir"])
            rows.append(row)

    profile = {
        "name": "review",
        "model_logical_id": args.name,
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "context": args.context,
        "batch_size": args.batch_size,
        "rounds": args.rounds,
        "warmup_rounds": args.warmup_rounds,
        "repetitions": args.repetitions,
        "gradient_checkpointing": False,
        "minimum_reduction_percent": args.minimum_reduction_percent,
    }
    summary = summarize(
        rows, profile, args.image_digest, args.source_revision, runtime_source_revision,
        orders, input_validation,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_runs_csv(output_dir / "runs.csv", summary["runs"])
    print(json.dumps({key: summary[key] for key in ("status", "methods", "memrift_reduction_percent", "claim_supported", "claim_scope")}, indent=2))
    return 0 if all(code == 0 for code in returncodes) and summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
