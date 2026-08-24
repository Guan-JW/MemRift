#!/usr/bin/env python3
import argparse
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

try:
    from experiments.model_loading.driver_utils import (
        WorkerFailure,
        environment_record,
        prepare_outputs,
        run_supervised,
        utc_now,
        write_json,
    )
except ModuleNotFoundError:
    from driver_utils import (
        WorkerFailure,
        environment_record,
        prepare_outputs,
        run_supervised,
        utc_now,
        write_json,
    )


METHODS = ("lora", "qlora-online", "qlora-prequant", "memrift")
CACHE_STATE = "warm"


def build_schedule(runs, methods, seed):
    if runs < 1:
        raise ValueError("--runs must be at least 1")
    schedule = [(run, method) for run in range(runs) for method in methods]
    random.Random(seed).shuffle(schedule)
    return schedule


def checkpoint_for_method(method, prepared):
    if method not in ("qlora-prequant", "memrift"):
        return None
    if prepared is None:
        raise ValueError(f"--prepared is required for method {method}")
    directory = "nf4" if method == "qlora-prequant" else "memrift"
    return Path(prepared) / directory


def summarize_rows(rows, methods):
    summary = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            raise ValueError(f"no current-run results for method {method}")
        summary[method] = {
            "runs": len(method_rows),
            "cache_state": CACHE_STATE,
            "cache_dropped": False,
            "load_to_ready_seconds_median": statistics.median(
                row["load_to_ready_seconds"] for row in method_rows
            ),
            "peak_process_rss_bytes_median": statistics.median(
                row["peak_process_rss_bytes"] for row in method_rows
            ),
            "peak_process_rss_delta_bytes_median": statistics.median(
                row["peak_process_rss_delta_bytes"] for row in method_rows
            ),
            "peak_system_used_bytes_median": statistics.median(
                row["peak_system_used_bytes"] for row in method_rows
            ),
            "peak_system_used_delta_bytes_median": statistics.median(
                row["peak_system_used_delta_bytes"] for row in method_rows
            ),
            "peak_torch_allocated_bytes_median": statistics.median(
                row["peak_torch_allocated_bytes"] for row in method_rows
            ),
            "checkpoint_bytes": method_rows[0]["checkpoint_bytes"],
            "online_quantized_tensor_calls": method_rows[0]["online_quantized_tensor_calls"],
            "prequantized_tensor_calls": method_rows[0]["prequantized_tensor_calls"],
        }
    return summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-timeout-seconds", type=float, default=3600)
    parser.add_argument("--min-mem-available-bytes", type=int, default=1024**3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker_timeout_seconds < 0:
        parser.error("--worker-timeout-seconds must be nonnegative")
    if args.min_mem_available_bytes < 0:
        parser.error("--min-mem-available-bytes must be nonnegative")
    methods = tuple(dict.fromkeys(args.methods))
    try:
        schedule = build_schedule(args.runs, methods, args.seed)
        checkpoints = {
            method: checkpoint_for_method(method, args.prepared) for method in methods
        }
    except ValueError as error:
        parser.error(str(error))

    output_root = Path(args.output_root) / args.name
    output_root.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("loading_worker.py")
    current_outputs = [output_root / f"{method}-{run}.json" for run, method in schedule]
    summary_path = output_root / "summary.json"
    driver_path = output_root / "driver.json"
    try:
        prepare_outputs((*current_outputs, summary_path, driver_path), args.overwrite)
    except (FileExistsError, IsADirectoryError) as error:
        parser.error(str(error))
    driver = {
        "kind": "model_loading_benchmark",
        "status": "running",
        "cache_state": CACHE_STATE,
        "cache_dropped": False,
        "started_at": utc_now(),
        "command": [sys.executable, str(Path(__file__)), *(argv if argv is not None else sys.argv[1:])],
        "environment": environment_record(),
        "schedule": [{"run": run, "method": method} for run, method in schedule],
        "failures": [],
    }
    write_json(driver_path, driver)

    # Each schedule entry gets its own process; shuffling avoids method-order bias.
    for (run, method), output in zip(schedule, current_outputs):
        command = [
            args.python,
            str(worker),
            "--method",
            method,
            "--model",
            args.model,
            "--output",
            str(output),
            "--device",
            args.device,
            "--cache-state",
            CACHE_STATE,
        ]
        if checkpoints[method] is not None:
            command += ["--checkpoint", str(checkpoints[method])]
        try:
            run_supervised(
                command, args.worker_timeout_seconds, args.min_mem_available_bytes
            )
        except WorkerFailure as error:
            driver["status"] = "failed"
            driver["finished_at"] = utc_now()
            driver["failures"].append(
                {
                    "run": run,
                    "method": method,
                    "command": command,
                    "reason": error.reason,
                    "returncode": error.returncode,
                    "timestamp": utc_now(),
                }
            )
            write_json(driver_path, driver)
            raise

    # Read only files produced by this invocation, never stale neighboring JSON.
    try:
        rows = [json.loads(path.read_text()) for path in current_outputs]
        summary = summarize_rows(rows, methods)
    except Exception as error:
        driver["status"] = "failed"
        driver["finished_at"] = utc_now()
        driver["failures"].append(
            {"stage": "summary", "reason": repr(error), "timestamp": utc_now()}
        )
        write_json(driver_path, driver)
        raise
    write_json(summary_path, summary)
    driver["status"] = "complete"
    driver["finished_at"] = utc_now()
    driver["result_files"] = [path.name for path in current_outputs]
    driver["summary"] = summary_path.name
    write_json(driver_path, driver)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
