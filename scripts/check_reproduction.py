#!/usr/bin/env python3
"""Check reviewer-facing experiment outputs against documented acceptance rules."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


LOADING_METHODS = {"lora", "qlora-online", "qlora-prequant", "memrift"}


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
    record(checks, "methods", set(data) == LOADING_METHODS, sorted(data), sorted(LOADING_METHODS))
    medians = {}
    for method in sorted(LOADING_METHODS):
        values = data.get(method) or {}
        observed = values.get("load_to_ready_seconds_median")
        valid_median = isinstance(observed, (int, float)) and math.isfinite(observed) and observed > 0
        medians[method] = observed if valid_median else None
        record(checks, f"{method}_runs", values.get("runs", 0) >= 5, values.get("runs"), ">= 5")
        record(checks, f"{method}_cache", values.get("cache_state") == "warm" and values.get("cache_dropped") is False, {"state": values.get("cache_state"), "dropped": values.get("cache_dropped")}, {"state": "warm", "dropped": False})
        record(checks, f"{method}_median_seconds", valid_median, observed, "positive finite seconds")
    prequant = data.get("qlora-prequant") or {}
    online = data.get("qlora-online") or {}
    record(checks, "serialized_nf4_path", prequant.get("online_quantized_tensor_calls") == 0 and prequant.get("prequantized_tensor_calls", 0) > 0, {"online": prequant.get("online_quantized_tensor_calls"), "prequantized": prequant.get("prequantized_tensor_calls")}, "online=0 and prequantized>0")
    record(checks, "online_nf4_path", online.get("online_quantized_tensor_calls", 0) > 0 and online.get("prequantized_tensor_calls") == 0, {"online": online.get("online_quantized_tensor_calls"), "prequantized": online.get("prequantized_tensor_calls")}, "online>0 and prequantized=0")
    improvement = (
        (medians["qlora-online"] - medians["memrift"]) / medians["qlora-online"] * 100
        if medians["qlora-online"] and medians["memrift"] is not None else None
    )
    record(checks, "memrift_vs_qlora_online", improvement is not None and improvement > 0, improvement, "> 0%")


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
        "--act_compact_concurrency": "1", "--act_decode_concurrency": "1",
        "--weight_async_concurrency": "1", "--weight_lookahead": "1",
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
        "batch_size", "rounds", "warmup_rounds", "activation_compaction_concurrency",
        "activation_decode_concurrency", "weight_materialization_concurrency",
    )}
    expected_profile = {
        "model_logical_id": "tinyllama-1.1b-chat-v1.0",
        "dataset": "tatsu-lab/alpaca",
        "dataset_revision": "dce01c9b08f87459cf36a430d809084718273017",
        "context": 2048,
        "batch_size": 3,
        "rounds": 7,
        "warmup_rounds": 1,
        "activation_compaction_concurrency": 1,
        "activation_decode_concurrency": 1,
        "weight_materialization_concurrency": 1,
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
    exact_inputs = {
        "model_revision": model_revision,
        "checkpoint_model_logical_id": "tinyllama-1.1b-chat-v1.0",
        "checkpoint_source_revision": model_revision,
        "checkpoint_format": "memrift-float-split-stride-v1",
        "checkpoint_zstd_level": 18,
        "checkpoint_payload_sha256": "f4dff6bb6c0017a8668e0263a24c311bc2891a2ed467ca58520aff30a5c9cdaa",
    }
    for name, expected in exact_inputs.items():
        record(checks, name, validation.get(name) == expected, validation.get(name), expected)
    fingerprint = validation.get("dataset_fingerprint")
    rows = validation.get("dataset_rows")
    record(checks, "dataset_fingerprint", isinstance(fingerprint, str) and bool(fingerprint), fingerprint, "non-empty dataset fingerprint")
    record(checks, "dataset_rows", isinstance(rows, int) and not isinstance(rows, bool) and rows > 0, rows, "positive dataset row count")

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
            expected_environment = {
                "git_revision": data.get("runtime_source_revision"),
                "container_image_digest": data.get("container_image_digest"),
            }
            observed_environment = {key: environment.get(key) for key in expected_environment}
            record(checks, f"rep_{repetition:02d}_environment", observed_environment == expected_environment, observed_environment, expected_environment)
            expected_resolved = {
                "min_available_mb": 4096, "matched_context": 2048, "batch_size": 3,
                "rounds": 7, "warmup_rounds": 1, "variants": list(order),
                "activation_compaction_concurrency": 1, "activation_decode_concurrency": 1,
                "weight_materialization_concurrency": 1, "weight_lookahead": 1,
                "activation_lookahead": 0, "activation_backend": "ebc-zstd", "compression_level": 1,
            }
            observed_resolved = {key: resolved.get(key) for key in expected_resolved}
            record(checks, f"rep_{repetition:02d}_configuration", observed_resolved == expected_resolved, observed_resolved, expected_resolved)
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


def result_metrics(experiment, path):
    if experiment in {"fidelity", "loading", "memory", "smoke", "tables23"}:
        data = json.loads(path.read_text(encoding="utf-8"))
    if experiment == "fidelity":
        return {
            "steps": f"{data.get('steps_completed')}/{data.get('steps_requested')}",
            "tensor_mismatches": data.get("tensor_mismatches"),
        }
    if experiment == "loading":
        medians = {
            method: (data.get(method) or {}).get("load_to_ready_seconds_median")
            for method in sorted(LOADING_METHODS)
        }
        online = medians["qlora-online"]
        memrift = medians["memrift"]
        improvement = (
            (online - memrift) / online * 100
            if isinstance(online, (int, float)) and online > 0 and isinstance(memrift, (int, float)) else None
        )
        return {
            "median_loading_seconds": medians,
            "memrift_improvement_vs_qlora_online_percent": improvement,
        }
    if experiment == "memory":
        methods = data.get("methods") or {}
        return {
            "median_peak_system_memory_mib": {
                method: (values.get("peak_system_used_bytes_median") / 2**20
                         if isinstance(values.get("peak_system_used_bytes_median"), (int, float)) else None)
                for method, values in sorted(methods.items())
            },
            "memrift_reduction_percent": data.get("memrift_reduction_percent"),
        }
    if experiment == "smoke":
        result = data.get("result") or {}
        return {"rounds": result.get("rounds"), "activation_compression_ratio": result.get("activation_compression_ratio")}
    if experiment == "tables23":
        return {"protocol_reportable": data.get("protocol_reportable"), "numeric_acceptance_met": data.get("reported_numeric_acceptance_met")}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if experiment == "entropy":
        return {"exponent_entropy_bits": {row.get("scope"): float(row["exponent_per_8"]) for row in rows}}
    if experiment == "backends":
        return {
            "backends": {
                row.get("backend"): {"status": row.get("status"), "compression_ratio": float(row.get("compression_ratio") or 0)}
                for row in rows
            }
        }
    return {}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(CHECKERS), required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")

    checks = []
    CHECKERS[args.experiment](args.input, checks)
    ok = all(item["ok"] for item in checks)
    failed = [
        {"requirement": item["name"], "observed": item["observed"], "expected": item["expected"]}
        for item in checks if not item["ok"]
    ]
    claim_checks = {"memory": {"memrift_vs_lora", "memrift_vs_qlora", "claim_supported"},
                    "loading": {"memrift_vs_qlora_online"}}
    failed_names = {item["requirement"] for item in failed}
    claim_not_supported = bool(failed_names) and failed_names.issubset(claim_checks.get(args.experiment, set()))
    report = {
        "schema_version": "1.0", "experiment": args.experiment, "input": str(args.input),
        "ok": ok,
        "outcome": "passed" if ok else "claim_not_supported" if claim_not_supported else "requirements_not_met",
        "metrics": result_metrics(args.experiment, args.input),
        "unmet_requirements": failed,
        "checks": checks,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
