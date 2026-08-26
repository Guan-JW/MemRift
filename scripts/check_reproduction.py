#!/usr/bin/env python3
"""Check reviewer-facing experiment outputs against documented acceptance rules."""

import argparse
import csv
import json
import math
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


CHECKERS = {
    "smoke": check_smoke,
    "fidelity": check_fidelity,
    "entropy": check_entropy,
    "loading": check_loading,
    "backends": check_backends,
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
