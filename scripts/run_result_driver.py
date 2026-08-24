#!/usr/bin/env python3
"""Run one experiment under a memory/timeout watchdog and record provenance."""

import argparse
import datetime as dt
import json
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def system_memory() -> tuple[int, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        name, value, *_ = line.replace(":", "").split()
        if name in ("MemTotal", "MemAvailable"):
            values[name] = int(value) * 1024
    if len(values) != 2:
        raise RuntimeError("MemTotal or MemAvailable missing from /proc/meminfo")
    return values["MemTotal"], values["MemAvailable"]


def process_rss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def memory_events() -> int:
    for candidate in (Path("/sys/fs/cgroup/memory.events"),):
        try:
            fields = dict(line.split()[:2] for line in candidate.read_text().splitlines())
            return int(fields.get("oom_kill", "0"))
        except (OSError, ValueError):
            pass
    return 0


def terminate_group(process: subprocess.Popen, grace: float = 10.0) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)


def extract_result(log_path: Path):
    result = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        markers = ("MEMRIFT_RESULT_JSON=", "MEMRIFT_RESULT_JSON ")
        marker = next((item for item in markers if item in line), None)
        if marker:
            value = line.split(marker, 1)[1].strip()
            try:
                result = json.loads(value)
            except json.JSONDecodeError:
                candidate = Path(value)
                if candidate.is_file():
                    result = json.loads(candidate.read_text(encoding="utf-8"))
    return result


SAFETY_OPTIONS = (
    "--max_length", "--round", "--warmup_rounds", "--batch_size",
    "--gradient_checkpointing", "--weight", "--gc_keep_recompute_weights",
    "--gc_no_recompute_prefetch",
)


def safety_occurrences(command) -> dict[str, list[tuple[int, str | None]]]:
    found = {option: [] for option in SAFETY_OPTIONS}
    for index, token in enumerate(command):
        if not token.startswith("--"):
            continue
        key, separator, inline = token.partition("=")
        matches = [option for option in SAFETY_OPTIONS if option.startswith(key)]
        if len(matches) == 1:
            found[matches[0]].append((index, inline if separator else None))
    return found


def command_integer(command, occurrences, option: str, default: int) -> int:
    matches = occurrences[option]
    if not matches:
        return default
    index, inline = matches[-1]
    if inline is not None:
        return int(inline)
    if index + 1 >= len(command):
        raise ValueError(f"{option} requires a value")
    return int(command[index + 1])


def validate_and_write_record(record: dict, run_dir: Path) -> None:
    import jsonschema

    schema_path = Path(__file__).parents[1] / "manifests" / "run-record.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(record, schema)
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--min-available-gib", type=float, default=4.0)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--allow-unsafe", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_name):
        parser.error("--run-name must be a portable filename component")
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be at least 1")
    if args.min_available_gib < 0:
        parser.error("--min-available-gib must be non-negative")
    occurrences = safety_occurrences(command)
    duplicates = [option for option, matches in occurrences.items() if len(matches) > 1]
    if duplicates:
        parser.error(f"duplicate safety-sensitive child options are prohibited: {duplicates}")
    try:
        context_tokens = command_integer(command, occurrences, "--max_length", 2048)
        rounds = command_integer(command, occurrences, "--round", 2)
        warmup_rounds = command_integer(command, occurrences, "--warmup_rounds", 1)
        batch_size = command_integer(command, occurrences, "--batch_size", 1)
    except ValueError as exc:
        parser.error(str(exc))
    if min(context_tokens, rounds, batch_size) < 1 or warmup_rounds < 0 or warmup_rounds >= rounds:
        parser.error("child dimensions and round counts are invalid")
    if context_tokens >= 4096 and not occurrences["--gradient_checkpointing"]:
        parser.error("non-GC runs at 4096 tokens or above are prohibited")
    corrected_gc = {"--gc_keep_recompute_weights", "--gc_no_recompute_prefetch"}
    present = {option for option, matches in occurrences.items() if matches}
    if context_tokens >= 4096 and "--weight" in present and not corrected_gc.issubset(present):
        parser.error("MemRift+GC at 4096 or above requires both recomputation correction flags")
    if context_tokens > 4096 and not args.allow_unsafe:
        parser.error("contexts above 4096 require --allow-unsafe")

    run_dir = args.results_dir / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to mix results in non-empty directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    minimum = int(args.min_available_gib * 1024**3)
    total_memory, initial_available = system_memory()
    estimate = {
        "context_tokens": context_tokens, "batch_size": batch_size, "rounds": rounds,
        "warmup_rounds": warmup_rounds, "minimum_available_bytes": minimum,
        "available_before_bytes": initial_available,
    }
    print(json.dumps({"resource_estimate": estimate, "command": command}, indent=2), flush=True)
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    (run_dir / "resolved-config.json").write_text(json.dumps(estimate, indent=2) + "\n")
    environment = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "source_revision": "bb138185d2bd0b88d924d7ea20fe61d72571a7b6",
        "benchmark_revision": "2fdc90fbcad7c20cc565480ddd7c8931af596531",
        "container_image_digest": os.environ.get("MEMRIFT_IMAGE_DIGEST", "unknown-unverified"),
        "wandb_mode": os.environ.get("WANDB_MODE", "disabled"),
        "offline": os.environ.get("HF_HUB_OFFLINE", "1") == "1",
        "dataset_revision": args.dataset_revision,
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    started = dt.datetime.now(dt.timezone.utc)
    if initial_available < minimum:
        log_path = run_dir / "raw.log"
        log_path.write_text(
            f"memory guard refused launch: {initial_available} bytes available, {minimum} required\n",
            encoding="utf-8",
        )
        ended = dt.datetime.now(dt.timezone.utc)
        record = {
            "schema_version": "1.0", "run_name": args.run_name, "status": "memory_guard",
            "exit_code": 125, "started_at": started.isoformat(), "ended_at": ended.isoformat(),
            "model_logical_id": args.model_id, "checkpoint_logical_id": args.checkpoint_id,
            "command_file": "command.txt", "raw_log": "raw.log", "result": None,
            "environment": environment, "validation_error": None,
        }
        validate_and_write_record(record, run_dir)
        print(json.dumps(record, indent=2))
        return 1
    watch_started = time.monotonic()
    before_oom = memory_events()
    reason = None
    minimum_available = initial_available
    peak_system_used = total_memory - initial_available
    peak_process_rss = 0
    log_path = run_dir / "raw.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "MEMRIFT_RUN_DIR": str(run_dir)}, text=True,
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - watch_started
                _, current_available = system_memory()
                minimum_available = min(minimum_available, current_available)
                peak_system_used = max(peak_system_used, total_memory - current_available)
                peak_process_rss = max(peak_process_rss, process_rss(process.pid))
                if elapsed > args.timeout_seconds:
                    reason = "timeout"
                    terminate_group(process)
                    break
                if current_available < minimum:
                    reason = "memory_guard"
                    terminate_group(process)
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            reason = "user_termination"
            terminate_group(process)
    returncode = process.wait()
    ended = dt.datetime.now(dt.timezone.utc)
    result = extract_result(log_path)
    validation_error = None
    if isinstance(result, dict):
        for key, logical_id in (("model", args.model_id), ("checkpoint", args.checkpoint_id)):
            if isinstance(result.get(key), str) and Path(result[key]).is_absolute():
                result[key] = logical_id
        if "round_peak_gpu_alloc_MB_max" in result:
            result.setdefault("peak_torch_allocated_bytes", int(result["round_peak_gpu_alloc_MB_max"] * 1024**2))
        if "round_peak_gpu_reserved_MB_max" in result:
            result.setdefault("peak_torch_reserved_bytes", int(result["round_peak_gpu_reserved_MB_max"] * 1024**2))
        result.setdefault("peak_process_rss_bytes", peak_process_rss)
        result.setdefault("peak_system_used_bytes", peak_system_used)
        result.setdefault("minimum_system_available_bytes", minimum_available)
        try:
            import jsonschema

            schema_path = Path(__file__).parents[1] / "manifests" / "result.schema.json"
            jsonschema.validate(result, json.loads(schema_path.read_text(encoding="utf-8")))
        except Exception as exc:
            validation_error = str(exc)
    if memory_events() > before_oom:
        reason = "oom"
    elif reason is None and returncode == 0 and (result is None or validation_error):
        reason = "validation_failure"
    elif reason is None and returncode == 0:
        reason = "success"
    elif reason is None:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        reason = "dependency_failure" if "ModuleNotFoundError" in text or "ImportError" in text else "software_failure"

    record = {
        "schema_version": "1.0", "run_name": args.run_name, "status": reason,
        "exit_code": returncode, "started_at": started.isoformat(), "ended_at": ended.isoformat(),
        "model_logical_id": args.model_id, "checkpoint_logical_id": args.checkpoint_id,
        "command_file": "command.txt", "raw_log": "raw.log", "result": result,
        "environment": environment, "validation_error": validation_error,
    }
    validate_and_write_record(record, run_dir)
    print(json.dumps(record, indent=2))
    return 0 if reason == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
