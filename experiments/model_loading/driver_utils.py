import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


PROVENANCE_PACKAGES = ("torch", "transformers", "peft", "bitsandbytes")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def environment_record():
    versions = {}
    for package in PROVENANCE_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    uname = os.uname()
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": {
            "sysname": uname.sysname,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "packages": versions,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
    }


def prepare_outputs(paths, overwrite=False):
    paths = tuple(Path(path) for path in paths)
    stale = [path for path in paths if path.exists()]
    if stale and not overwrite:
        names = ", ".join(str(path) for path in stale)
        raise FileExistsError(f"invocation output already exists: {names}; use --overwrite")
    for path in stale:
        if path.is_dir():
            raise IsADirectoryError(f"expected an output file, found directory: {path}")
        path.unlink()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def enter_process_group():
    """Give a worker and any descendants a group the driver can terminate."""
    try:
        os.setsid()
    except PermissionError:
        # Already a process-group leader, which still provides the required boundary.
        pass


def mem_available_bytes():
    with open("/proc/meminfo") as source:
        for line in source:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _descendants(parent_pid):
    descendants = set()
    while True:
        added = False
        for status in Path("/proc").glob("[0-9]*/status"):
            try:
                fields = dict(
                    line.split(":", 1) for line in status.read_text().splitlines() if ":" in line
                )
                pid = int(status.parent.name)
                ppid = int(fields["PPid"].strip())
            except (FileNotFoundError, KeyError, PermissionError, ValueError):
                continue
            if ppid == parent_pid or ppid in descendants:
                if pid not in descendants:
                    descendants.add(pid)
                    added = True
        if not added:
            return descendants


def _terminate_descendants(parent_pid):
    parent_group = os.getpgrp()
    for pid in _descendants(parent_pid):
        try:
            group = os.getpgid(pid)
            if group != parent_group:
                os.killpg(group, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(0.5)
    for pid in _descendants(parent_pid):
        try:
            group = os.getpgid(pid)
            if group != parent_group:
                os.killpg(group, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class WorkerFailure(RuntimeError):
    def __init__(self, reason, returncode=None):
        super().__init__(reason)
        self.reason = reason
        self.returncode = returncode


def run_supervised(command, timeout_seconds, min_mem_available_bytes, poll_seconds=0.25):
    """Run through subprocess.run for testability while a parent watchdog supervises it."""
    stopped = threading.Event()
    failure = []
    started = time.monotonic()

    def watch():
        low_samples = 0
        while not stopped.wait(poll_seconds):
            reason = None
            if timeout_seconds and time.monotonic() - started >= timeout_seconds:
                reason = f"worker exceeded {timeout_seconds:g} second timeout"
            if min_mem_available_bytes:
                try:
                    available = mem_available_bytes()
                except (OSError, RuntimeError):
                    available = None
                low_samples = low_samples + 1 if available is not None and available < min_mem_available_bytes else 0
                if low_samples >= 3:
                    reason = (
                        f"MemAvailable remained below {min_mem_available_bytes} bytes "
                        f"(last sample {available} bytes)"
                    )
            if reason:
                failure.append(reason)
                _terminate_descendants(os.getpid())
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    returncode = None
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        returncode = error.returncode
        if not failure:
            failure.append(f"worker exited with status {error.returncode}")
    finally:
        stopped.set()
        watcher.join()
    if failure:
        raise WorkerFailure(failure[0], returncode)
