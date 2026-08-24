#!/usr/bin/env python3
"""Safely run the MemRift gradient-checkpointing experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
TRAIN = ARTIFACT_ROOT / "src" / "train_memrift.py"
RESULT_PREFIX = "MEMRIFT_RESULT_JSON "
MIB = 1024 * 1024

VARIANTS: dict[str, list[str]] = {
    "lora": [],
    "lora_gc": ["--gradient_checkpointing"],
    "memrift": ["--hook", "--weight", "--weight_async", "--activation", "--act_async"],
    "memrift_gc": [
        "--hook",
        "--weight",
        "--weight_async",
        "--activation",
        "--act_async",
        "--gradient_checkpointing",
        "--gc_keep_recompute_weights",
        "--gc_no_recompute_prefetch",
    ],
    "qlora_gc": ["--finetune_type", "qlora", "--autocast_context", "--gradient_checkpointing"],
}

USER_TERMINATED = threading.Event()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def available_memory_bytes() -> int:
    """Read Linux MemAvailable without requiring psutil."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def parse_memory_events(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    try:
        return {
            key: int(value)
            for key, value in (line.split() for line in (path / "memory.events").read_text().splitlines())
        }
    except (OSError, ValueError):
        return {}


class CgroupLimit:
    """Best-effort cgroup-v2 limit for one worker process group."""

    def __init__(self, limit_mb: int | None, root: Path = Path("/sys/fs/cgroup")) -> None:
        self.limit_mb = limit_mb
        self.root = root
        self.path: Path | None = None
        self.before: dict[str, int] = {}
        self.error: str | None = None

    def attach(self, pid: int) -> None:
        if self.limit_mb is None:
            return
        try:
            self.path = self.root / f"memrift-{os.getpid()}-{pid}"
            self.path.mkdir()
            (self.path / "memory.max").write_text(str(self.limit_mb * MIB))
            self.before = parse_memory_events(self.path)
            (self.path / "cgroup.procs").write_text(str(pid))
        except OSError as exc:
            self.error = str(exc)
            self.path = None

    def oom_killed(self) -> bool:
        after = parse_memory_events(self.path)
        return after.get("oom_kill", 0) > self.before.get("oom_kill", 0)

    def close(self) -> None:
        if self.path is not None:
            try:
                self.path.rmdir()
            except OSError:
                pass


def classify_exit(
    returncode: int,
    output: str,
    *,
    timed_out: bool = False,
    safety_stop: bool = False,
    user_terminated: bool = False,
    cgroup_oom: bool = False,
) -> str:
    """Classify worker exits independently for straightforward unit tests."""
    text = output.lower()
    if user_terminated:
        return "user_termination"
    if timed_out:
        return "timeout"
    if safety_stop:
        return "safety_stop"
    if cgroup_oom or "cuda out of memory" in text or "outofmemoryerror" in text or "killed process" in text:
        return "oom"
    if returncode == 0 and RESULT_PREFIX in output:
        return "ok"
    dependency_markers = (
        "modulenotfounderror",
        "importerror",
        "no such file or directory",
        "cannot open shared object file",
        "extension was not built",
        "dependency launch failure",
    )
    if any(marker in text for marker in dependency_markers):
        return "dependency_failure"
    validation_markers = ("error: argument", "compressed checkpoint index is missing", "checkpoint file listed")
    if returncode == 0 or any(marker in text for marker in validation_markers):
        return "validation_failure"
    return "software_failure"


def terminate_process_group(proc: subprocess.Popen[str], grace_sec: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, AttributeError):
        proc.terminate()
    try:
        proc.wait(timeout=grace_sec)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        proc.kill()


class WorkerRunner:
    """Subprocess runner with injectable process, clock, and memory providers."""

    def __init__(
        self,
        *,
        popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
        memory_available: Callable[[], int] = available_memory_bytes,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.popen = popen
        self.clock = clock
        self.memory_available = memory_available
        self.sleep = sleep

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout_sec: int,
        min_available_mb: int,
        cgroup_limit_mb: int | None,
    ) -> tuple[int, str, dict[str, object]]:
        started = self.clock()
        try:
            proc = self.popen(
                list(command),
                cwd=str(cwd),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            return 127, f"dependency launch failure: {exc}\n", {
                "timed_out": False,
                "safety_stop": False,
                "user_terminated": False,
                "cgroup_oom": False,
                "cgroup_error": None,
                "minimum_available_system_MB": None,
            }
        cgroup = CgroupLimit(cgroup_limit_mb)
        cgroup.attach(proc.pid)
        lines: queue.Queue[str] = queue.Queue()

        def read_output() -> None:
            if proc.stdout is not None:
                for line in proc.stdout:
                    lines.put(line)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        output: list[str] = []
        timed_out = safety_stop = user_terminated = False
        minimum_seen = self.memory_available()
        while proc.poll() is None:
            while True:
                try:
                    line = lines.get_nowait()
                except queue.Empty:
                    break
                output.append(line)
                print(line, end="", flush=True)
            available = self.memory_available()
            if available:
                minimum_seen = available if not minimum_seen else min(minimum_seen, available)
            if USER_TERMINATED.is_set():
                user_terminated = True
                terminate_process_group(proc)
                break
            if self.clock() - started > timeout_sec:
                timed_out = True
                terminate_process_group(proc)
                break
            if available and available < min_available_mb * MIB:
                safety_stop = True
                terminate_process_group(proc)
                break
            self.sleep(0.2)
        reader.join(timeout=2)
        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                break
            output.append(line)
            print(line, end="", flush=True)
        returncode = proc.poll()
        if returncode is None:
            returncode = 130 if user_terminated else 124
        cgroup_oom = cgroup.oom_killed()
        metadata: dict[str, object] = {
            "timed_out": timed_out,
            "safety_stop": safety_stop,
            "user_terminated": user_terminated,
            "cgroup_oom": cgroup_oom,
            "cgroup_error": cgroup.error,
            "minimum_available_system_MB": minimum_seen // MIB if minimum_seen else None,
        }
        cgroup.close()
        return returncode, "".join(output), metadata


def build_command(args: argparse.Namespace, variant: str, context: int, rounds: int, run_dir: Path) -> list[str]:
    command = [
        args.python,
        "-u",
        str(TRAIN),
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--dataset-cache",
        args.dataset_cache,
        "--results-dir",
        str(run_dir),
        "--device",
        args.device,
        "--wandb-mode",
        args.wandb_mode,
        "--max_length",
        str(context),
        "--batch_size",
        str(args.batch_size),
        "--round",
        str(rounds),
        "--warmup_rounds",
        str(min(args.warmup_rounds, rounds - 1)),
        "--act_compact_concurrency",
        str(args.activation_compaction_concurrency),
        "--act_decode_concurrency",
        str(args.activation_decode_concurrency),
        "--weight_async_concurrency",
        str(args.weight_materialization_concurrency),
        *VARIANTS[variant],
    ]
    if variant.startswith("memrift"):
        command.extend(["--checkpoint", args.checkpoint])
    if args.disable_tegrastats:
        command.append("--disable-tegrastats")
    else:
        command.extend(["--tegrastats-bin", args.tegrastats_bin])
    return command


def resource_estimate(variant: str, context: int, args: argparse.Namespace) -> dict[str, object]:
    baseline_mb = {"lora": 20000, "lora_gc": 14500, "memrift": 18000, "memrift_gc": 13500, "qlora_gc": 12500}[variant]
    return {
        "variant": variant,
        "context_tokens": context,
        "batch_size": args.batch_size,
        "estimated_system_memory_MB": int(baseline_mb + context * args.batch_size * 2.5),
        "minimum_available_watchdog_MB": args.min_available_mb,
        "timeout_sec": args.timeout_sec,
        "cgroup_limit_MB": args.cgroup_memory_limit_mb,
        "estimate_note": "Conservative planning estimate; actual usage is hardware/model dependent.",
    }


def extract_result(output: str) -> dict[str, object]:
    for line in output.splitlines():
        if line.startswith(RESULT_PREFIX):
            try:
                value = json.loads(line[len(RESULT_PREFIX):])
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def run_one(
    args: argparse.Namespace,
    runner: WorkerRunner,
    variant: str,
    context: int,
    rounds: int,
    tag: str,
) -> dict[str, object]:
    if context >= 4096 and variant in {"lora", "memrift"} and not args.allow_unsafe:
        raise ValueError(f"unsafe non-GC configuration blocked: {variant} at context {context}; use --allow-unsafe to override")
    run_name = f"{tag}_{variant}_ctx{context}"
    run_dir = Path(args.results_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args, variant, context, rounds, run_dir)
    estimate = resource_estimate(variant, context, args)
    (run_dir / "command.json").write_text(json.dumps(command, indent=2))
    (run_dir / "resource_estimate.json").write_text(json.dumps(estimate, indent=2, sort_keys=True))
    print("RESOURCE_ESTIMATE " + json.dumps(estimate, sort_keys=True), flush=True)
    print("COMMAND " + json.dumps(command), flush=True)
    started_at = utc_now()
    started = time.monotonic()
    env = os.environ.copy()
    env["WANDB_MODE"] = args.wandb_mode
    returncode, output, metadata = runner.run(
        command,
        cwd=ARTIFACT_ROOT,
        env=env,
        timeout_sec=args.timeout_sec,
        min_available_mb=args.min_available_mb,
        cgroup_limit_mb=args.cgroup_memory_limit_mb,
    )
    ended_at = utc_now()
    (run_dir / "raw.log").write_text(output)
    status = classify_exit(returncode, output, **{key: bool(metadata[key]) for key in ("timed_out", "safety_stop", "user_terminated", "cgroup_oom")})
    row: dict[str, object] = {
        "variant": variant,
        "context": context,
        "status": status,
        "returncode": returncode,
        "elapsed_sec": time.monotonic() - started,
        "started_at": started_at,
        "ended_at": ended_at,
        "run_dir": str(Path("runs") / run_name),
        **metadata,
        **extract_result(output),
    }
    (run_dir / "run.json").write_text(json.dumps(row, indent=2, sort_keys=True))
    return row


def max_context_search(args: argparse.Namespace, runner: WorkerRunner, variant: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    step = args.context_step
    low = (args.max_context_low // step) * step
    high = (args.max_context_high // step) * step
    best: int | None = None
    while low <= high and not USER_TERMINATED.is_set():
        mid = max(step, ((low + high) // (2 * step)) * step)
        row = run_one(args, runner, variant, mid, max(2, args.warmup_rounds + 1), "maxctx")
        rows.append(row)
        if row["status"] == "ok":
            best = mid
            low = mid + step
        elif row["status"] == "oom":
            high = mid - step
        else:
            rows.append({"variant": variant, "status": "search_aborted", "context": mid, "reason": row["status"]})
            break
    if best is not None:
        rows.append({"variant": variant, "status": "max_context", "context": best})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--checkpoint", default="/checkpoints/TinyLlama-1.1B-Chat-v1.0/memrift")
    parser.add_argument("--dataset", default="timdettmers/openassistant-guanaco")
    parser.add_argument("--dataset-cache", default="/cache/huggingface")
    parser.add_argument("--results-dir", default="/results/gradient_checkpointing")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tegrastats-bin", default="/usr/bin/tegrastats")
    parser.add_argument("--disable-tegrastats", action="store_true")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--matched-context", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=2400)
    parser.add_argument("--min-available-mb", type=int, default=4096)
    parser.add_argument("--cgroup-memory-limit-mb", type=int)
    parser.add_argument("--activation-compaction-concurrency", type=int, default=1)
    parser.add_argument("--activation-decode-concurrency", type=int, default=1)
    parser.add_argument("--weight-materialization-concurrency", type=int, default=1)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["lora", "lora_gc", "memrift", "memrift_gc"])
    parser.add_argument("--run-max-context", action="store_true", help="Opt in to the maximum-context search")
    parser.add_argument("--max-context-low", type=int, default=2048)
    parser.add_argument("--max-context-high", type=int, default=4096)
    parser.add_argument("--context-step", type=int, default=256)
    parser.add_argument("--allow-unsafe", action="store_true")
    args = parser.parse_args(argv)
    positive = (
        "matched_context", "batch_size", "rounds", "timeout_sec", "min_available_mb",
        "activation_compaction_concurrency", "activation_decode_concurrency",
        "weight_materialization_concurrency", "context_step",
    )
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmup_rounds < 0 or args.warmup_rounds >= args.rounds:
        parser.error("--warmup-rounds must be non-negative and less than --rounds")
    if args.run_max_context and args.max_context_low > args.max_context_high:
        parser.error("--max-context-low cannot exceed --max-context-high")
    return args


def environment_manifest() -> dict[str, object]:
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "artifact_root": ".",
        "git_revision": os.environ.get("MEMRIFT_GIT_REVISION", "unknown"),
        "container_image_digest": os.environ.get("MEMRIFT_IMAGE_DIGEST", "unknown"),
    }


def main(argv: Sequence[str] | None = None, runner: WorkerRunner | None = None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    (results_dir / "environment.json").write_text(json.dumps(environment_manifest(), indent=2, sort_keys=True))
    runner = runner or WorkerRunner()
    rows: list[dict[str, object]] = []
    for variant in args.variants:
        if USER_TERMINATED.is_set():
            break
        rows.append(run_one(args, runner, variant, args.matched_context, args.rounds, "matched"))
    if args.run_max_context and not USER_TERMINATED.is_set():
        for variant in ("lora_gc", "memrift_gc"):
            rows.extend(max_context_search(args, runner, variant))
            if USER_TERMINATED.is_set():
                break
    (results_dir / "gc_experiments.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    write_csv(results_dir / "gc_experiments.csv", rows)
    return 130 if USER_TERMINATED.is_set() else (0 if all(row.get("status") == "ok" for row in rows if row.get("status") != "max_context") else 1)


def _handle_signal(_signum: int, _frame: object) -> None:
    USER_TERMINATED.set()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    raise SystemExit(main())
