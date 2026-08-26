#!/usr/bin/env python3
"""Check reviewer-facing experiment outputs against documented acceptance rules."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


LOADING_EXPECTED_SECONDS = {
    "lora": 2.15,
    "qlora-online": 4.26,
    "qlora-prequant": 2.07,
    "memrift": 2.66,
}


def record(checks, name, ok, observed, expected):
    checks.append({"name": name, "ok": bool(ok), "observed": observed, "expected": expected})


def check_smoke(path, checks):
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data.get("result") or {}
    record(checks, "status", data.get("status") == "success", data.get("status"), "success")
    record(checks, "exit_code", data.get("exit_code") == 0, data.get("exit_code"), 0)
    record(checks, "synthetic_data", result.get("synthetic_data") is True, result.get("synthetic_data"), True)
    record(checks, "measured_round", result.get("rounds", 0) > result.get("warmup_rounds", 0), result.get("rounds"), "greater than warmup_rounds")
    record(checks, "compression_ratio", result.get("activation_compression_ratio", 0) > 1, result.get("activation_compression_ratio"), "> 1")


def check_fidelity(path, checks):
    data = json.loads(path.read_text(encoding="utf-8"))
    record(checks, "steps_completed", data.get("steps_completed") == data.get("steps_requested"), data.get("steps_completed"), data.get("steps_requested"))
    record(checks, "tensor_mismatches", data.get("tensor_mismatches") == 0, data.get("tensor_mismatches"), 0)
    for scope in ("weights", "activations"):
        values = data.get(scope) or {}
        record(checks, f"{scope}_observed", values.get("tensors", 0) > 0, values.get("tensors"), "> 0")
        record(checks, f"{scope}_mismatches", values.get("mismatches") == 0, values.get("mismatches"), 0)


def check_entropy(path, checks):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    record(checks, "scopes", {row.get("scope") for row in rows} == {"W", "A"}, [row.get("scope") for row in rows], ["W", "A"])
    for row in rows:
        value = float(row["exponent_per_8"])
        record(checks, f"{row['scope']}_exponent_entropy", 2.61 <= value <= 2.93, value, "2.61 to 2.93 bits")


def check_loading(path, checks):
    data = json.loads(path.read_text(encoding="utf-8"))
    record(checks, "methods", set(data) == set(LOADING_EXPECTED_SECONDS), sorted(data), sorted(LOADING_EXPECTED_SECONDS))
    for method, expected in LOADING_EXPECTED_SECONDS.items():
        values = data.get(method) or {}
        observed = values.get("load_to_ready_seconds_median")
        within = isinstance(observed, (int, float)) and math.isfinite(observed) and abs(observed - expected) / expected <= 0.15
        record(checks, f"{method}_runs", values.get("runs", 0) >= 5, values.get("runs"), ">= 5")
        record(checks, f"{method}_cache", values.get("cache_state") == "warm" and values.get("cache_dropped") is False, {"state": values.get("cache_state"), "dropped": values.get("cache_dropped")}, {"state": "warm", "dropped": False})
        record(checks, f"{method}_median_seconds", within, observed, f"{expected} +/- 15%")
    prequant = data.get("qlora-prequant") or {}
    online = data.get("qlora-online") or {}
    record(checks, "serialized_nf4_path", prequant.get("online_quantized_tensor_calls") == 0 and prequant.get("prequantized_tensor_calls", 0) > 0, {"online": prequant.get("online_quantized_tensor_calls"), "prequantized": prequant.get("prequantized_tensor_calls")}, "online=0 and prequantized>0")
    record(checks, "online_nf4_path", online.get("online_quantized_tensor_calls", 0) > 0 and online.get("prequantized_tensor_calls") == 0, {"online": online.get("online_quantized_tensor_calls"), "prequantized": online.get("prequantized_tensor_calls")}, "online>0 and prequantized=0")


def check_backends(path, checks):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {"lz4", "zstd", "ebc-lz4", "ebc-zstd"}
    record(checks, "backends", {row.get("backend") for row in rows} == expected, sorted(row.get("backend") for row in rows), sorted(expected))
    for row in rows:
        record(checks, f"{row.get('backend')}_status", row.get("status") == "ok", row.get("status"), "ok")
        ratio = float(row.get("compression_ratio") or 0)
        record(checks, f"{row.get('backend')}_compression_ratio", ratio > 1, ratio, "> 1")


def check_tables23(path, checks):
    data = json.loads(path.read_text(encoding="utf-8"))
    record(checks, "status", data.get("status") == "complete", data.get("status"), "complete")
    record(checks, "protocol_reportable", data.get("protocol_reportable") is True, data.get("protocol_reportable"), True)
    record(checks, "numeric_acceptance", data.get("reported_numeric_acceptance_met") is True, data.get("reported_numeric_acceptance_met"), True)


MEMORY_METHOD_CONTRACTS = {
    "lora": {"finetune_type": "lora", "hook": False, "weight": False, "weight_async": False, "activation": False, "act_async": False},
    "qlora": {"finetune_type": "qlora", "hook": False, "weight": False, "weight_async": False, "activation": False, "act_async": False},
    "memrift": {"finetune_type": "lora", "hook": True, "weight": True, "weight_async": True, "activation": True, "act_async": True},
}


def command_option(command, option):
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        return None
    return command[positions[0] + 1]


def valid_memory_command(command, method):
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        return False
    if len(command) < 3 or command[1:3] != ["-u", "/workspace/src/train_memrift.py"]:
        return False
    expected = {
        "--model": "/models/model", "--dataset-cache": "/cache/huggingface",
        "--dataset": "tatsu-lab/alpaca",
        "--dataset-revision": "dce01c9b08f87459cf36a430d809084718273017",
        "--seed": "42", "--max_length": "2048", "--batch_size": "3",
        "--round": "7", "--warmup_rounds": "1",
        "--act_compact_concurrency": "16", "--act_decode_concurrency": "4",
        "--weight_async_concurrency": "4", "--weight_lookahead": "1",
        "--activation_lookahead": "0", "--activation-backend": "ebc-zstd",
        "--level": "1", "--tegra-csv": "tegrastats.csv",
    }
    if any(command_option(command, option) != value for option, value in expected.items()):
        return False
    if "--gradient_checkpointing" in command or "--synthetic-data" in command:
        return False
    compression_flags = {"--hook", "--weight", "--weight_async", "--activation", "--act_async"}
    if method == "lora":
        return not compression_flags.intersection(command) and "--finetune_type" not in command and "--checkpoint" not in command
    if method == "qlora":
        return (
            not compression_flags.intersection(command) and command_option(command, "--finetune_type") == "qlora"
            and "--autocast_context" in command and "--checkpoint" not in command
        )
    return (
        compression_flags.issubset(command) and "--finetune_type" not in command
        and command_option(command, "--checkpoint") == "/checkpoints/model"
    )


def read_tegrastats(path):
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        values = [int(row["ram_used_MB"]) for row in rows]
    except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError):
        return None, None
    return len(values), max(values) * 2**20 if values else None


def check_memory(path, checks):
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = data.get("profile") or {}
    methods = data.get("methods") or {}
    repetitions = profile.get("repetitions", 0)
    minimum = profile.get("minimum_reduction_percent", 0)
    record(checks, "status", data.get("status") == "complete", data.get("status"), "complete")
    record(checks, "methods", set(methods) == {"lora", "qlora", "memrift"}, sorted(methods), ["lora", "memrift", "qlora"])
    record(checks, "repetitions", repetitions == 3, repetitions, 3)
    observed_profile = {key: profile.get(key) for key in (
        "model_logical_id", "dataset", "dataset_revision", "context",
        "batch_size", "rounds", "warmup_rounds",
    )}
    expected_profile = {
        "model_logical_id": "tinyllama-1.1b-chat-v1.0",
        "dataset": "tatsu-lab/alpaca",
        "dataset_revision": "dce01c9b08f87459cf36a430d809084718273017",
        "context": 2048,
        "batch_size": 3,
        "rounds": 7,
        "warmup_rounds": 1,
    }
    record(checks, "review_profile", observed_profile == expected_profile, observed_profile, expected_profile)
    record(checks, "gradient_checkpointing", profile.get("gradient_checkpointing") is False, profile.get("gradient_checkpointing"), False)
    record(checks, "minimum_reduction", isinstance(minimum, (int, float)) and minimum >= 0, minimum, ">= 0%")
    record(checks, "method_order", data.get("method_order_valid") is True, data.get("method_order_valid"), True)
    for field in ("orchestrator_source_revision", "runtime_source_revision"):
        value = data.get(field)
        valid = isinstance(value, str) and len(value) == 40 and all(character in "0123456789abcdef" for character in value)
        record(checks, field, valid, value, "40-character Git revision")
    digest = data.get("container_image_digest")
    digest_valid = (
        isinstance(digest, str) and digest.startswith("sha256:") and len(digest) == 71
        and all(character in "0123456789abcdef" for character in digest[7:])
    )
    record(checks, "container_image_digest", digest_valid, digest, "sha256:<64 hex characters>")
    validation = data.get("input_validation") or {}
    model_revision = "de253fa9783f8bd558c9ed398c8ffbe3c55cedb3"
    validation_ok = (
        validation.get("model_revision") == model_revision
        and validation.get("checkpoint_model_logical_id") == "tinyllama-1.1b-chat-v1.0"
        and validation.get("checkpoint_source_revision") == model_revision
        and validation.get("checkpoint_format") == "memrift-float-split-stride-v1"
        and validation.get("checkpoint_zstd_level") == 18
        and validation.get("checkpoint_payload_sha256") == "f4dff6bb6c0017a8668e0263a24c311bc2891a2ed467ca58520aff30a5c9cdaa"
        and isinstance(validation.get("dataset_fingerprint"), str) and validation.get("dataset_fingerprint")
        and isinstance(validation.get("dataset_rows"), int) and validation.get("dataset_rows") > 0
    )
    record(checks, "input_validation", validation_ok, validation, "manifest-matched model, checkpoint, and dataset")

    runs = data.get("runs") or []
    expected_orders = [
        ("lora", "qlora", "memrift"),
        ("qlora", "memrift", "lora"),
        ("memrift", "lora", "qlora"),
    ]
    raw_valid = isinstance(runs, list) and len(runs) == 9
    run_peaks = {method: [] for method in ("lora", "qlora", "memrift")}
    if raw_valid:
        for repetition, order in enumerate(expected_orders, 1):
            try:
                environment = json.loads((path.parent / f"rep-{repetition:02d}" / "environment.json").read_text(encoding="utf-8"))
                resolved = json.loads((path.parent / f"rep-{repetition:02d}" / "resolved_config.json").read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                environment = {}
                resolved = {}
            raw_valid = raw_valid and (
                environment.get("git_revision") == data.get("runtime_source_revision")
                and environment.get("container_image_digest") == data.get("container_image_digest")
                and resolved.get("min_available_mb") == 4096
                and resolved.get("matched_context") == 2048 and resolved.get("batch_size") == 3
                and resolved.get("rounds") == 7 and resolved.get("warmup_rounds") == 1
                and resolved.get("variants") == list(order)
                and resolved.get("activation_compaction_concurrency") == 16
                and resolved.get("activation_decode_concurrency") == 4
                and resolved.get("weight_materialization_concurrency") == 4
                and resolved.get("weight_lookahead") == 1 and resolved.get("activation_lookahead") == 0
                and resolved.get("activation_backend") == "ebc-zstd" and resolved.get("compression_level") == 1
            )
            for position, method in enumerate(order, 1):
                candidates = [
                    row for row in runs
                    if row.get("repetition") == repetition and row.get("position") == position and row.get("variant") == method
                ]
                if len(candidates) != 1:
                    raw_valid = False
                    continue
                row = candidates[0]
                contract = MEMORY_METHOD_CONTRACTS[method]
                row_ok = (
                    row.get("status") == "ok" and row.get("context") == 2048 and row.get("batch_size") == 3
                    and row.get("rounds") == 7 and row.get("warmup_rounds") == 1
                    and row.get("dataset") == "tatsu-lab/alpaca"
                    and row.get("dataset_revision") == "dce01c9b08f87459cf36a430d809084718273017"
                    and row.get("synthetic_data") is False and row.get("gradient_checkpointing") is False
                    and row.get("seed") == 42 and row.get("activation_backend") == "ebc-zstd"
                    and row.get("activation_compression_level") == 1 and row.get("weight_lookahead") == 1
                    and row.get("activation_lookahead") == 0 and row.get("tegrastats_samples", 0) > 0
                    and all(row.get(key) == value for key, value in contract.items())
                )
                peak = row.get("peak_system_used_bytes")
                row_ok = row_ok and isinstance(peak, (int, float)) and peak > 0
                run_dir = Path(row.get("run_dir", ""))
                safe_dir = bool(run_dir.parts) and not run_dir.is_absolute() and ".." not in run_dir.parts
                artifact_dir = path.parent / run_dir if safe_dir else path.parent / ".invalid"
                try:
                    command = json.loads((artifact_dir / "command.json").read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    command = None
                samples, telemetry_peak = read_tegrastats(artifact_dir / "tegrastats.csv")
                row_ok = (
                    row_ok and safe_dir and valid_memory_command(command, method)
                    and samples == row.get("tegrastats_samples") and telemetry_peak == peak
                )
                raw_valid = raw_valid and row_ok
                if row_ok:
                    run_peaks[method].append(peak)
    record(checks, "raw_matched_runs", raw_valid, len(runs) if isinstance(runs, list) else None, "9 matched runs with commands and telemetry")
    medians = {}
    for method in ("lora", "qlora", "memrift"):
        values = methods.get(method) or {}
        peaks = values.get("peak_system_used_bytes") or []
        record(checks, f"{method}_successful_runs", values.get("successful_runs") == repetitions, values.get("successful_runs"), repetitions)
        record(checks, f"{method}_system_peaks", len(peaks) == repetitions and all(isinstance(value, (int, float)) and value > 0 for value in peaks), peaks, f"{repetitions} positive whole-system peaks")
        median = statistics.median(peaks) if len(peaks) == repetitions else None
        medians[method] = median
        recorded_median = values.get("peak_system_used_bytes_median")
        record(checks, f"{method}_median", median is not None and recorded_median == median, recorded_median, median)
        record(checks, f"{method}_raw_peaks", raw_valid and peaks == run_peaks[method], peaks, run_peaks[method])
    reductions = data.get("memrift_reduction_percent") or {}
    for baseline in ("lora", "qlora"):
        observed = reductions.get(baseline)
        recomputed = (
            (medians[baseline] - medians["memrift"]) / medians[baseline] * 100
            if medians.get(baseline) and medians.get("memrift") is not None else None
        )
        consistent = isinstance(observed, (int, float)) and recomputed is not None and math.isclose(observed, recomputed, rel_tol=1e-12)
        record(checks, f"memrift_vs_{baseline}_consistent", consistent, observed, recomputed)
        record(checks, f"memrift_vs_{baseline}", consistent and observed > minimum, observed, f"> {minimum}%")
    recomputed_claim = all(
        medians.get(baseline) is not None and medians.get("memrift") is not None and medians["memrift"] < medians[baseline]
        for baseline in ("lora", "qlora")
    )
    record(checks, "claim_supported", data.get("claim_supported") is True and recomputed_claim, data.get("claim_supported"), True)


CHECKERS = {
    "smoke": check_smoke,
    "fidelity": check_fidelity,
    "entropy": check_entropy,
    "loading": check_loading,
    "backends": check_backends,
    "memory": check_memory,
    "tables23": check_tables23,
}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(CHECKERS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")

    checks = []
    CHECKERS[args.experiment](args.input, checks)
    report = {"schema_version": "1.0", "experiment": args.experiment, "input": str(args.input), "ok": all(item["ok"] for item in checks), "checks": checks}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
