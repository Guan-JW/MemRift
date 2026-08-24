import json
import signal
import subprocess
import sys

import pytest

from conftest import load_module


VALID_RESULT = {
    "batch_size": 1,
    "max_length": 16,
    "rounds": 2,
    "warmup_rounds": 1,
    "round_time_mean_sec": 0.1,
}


class FakeProcess:
    next_pid = 4000

    def __init__(self, command, stdout, returncode=0, output="", running=False, **kwargs):
        self.command = command
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode = None if running else returncode
        self.final_returncode = returncode
        stdout.write(output)
        stdout.flush()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            if timeout is not None:
                raise subprocess.TimeoutExpired(self.command, timeout)
            self.returncode = self.final_returncode
        return self.returncode


def invoke(monkeypatch, tmp_path, *, output="", returncode=0, running=False, memory=None, timeout=10):
    module = load_module("scripts/run_result_driver.py", f"result_driver_{FakeProcess.next_pid}")
    process_holder = {}

    def popen(*args, **kwargs):
        process = FakeProcess(*args, **kwargs, returncode=returncode, output=output, running=running)
        process_holder["process"] = process
        return process

    values = iter(memory or [(16 * 1024**3, 12 * 1024**3)])
    last = [16 * 1024**3, 12 * 1024**3]

    def system_memory():
        try:
            last[:] = next(values)
        except StopIteration:
            pass
        return tuple(last)

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.platform, "platform", lambda: "mock-platform")
    monkeypatch.setattr(module, "system_memory", system_memory)
    monkeypatch.setattr(module, "memory_events", lambda: 0)
    monkeypatch.setattr(module, "process_rss", lambda pid: 1234)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    ticks = iter([0.0, 20.0] if timeout == 1 else [0.0, 0.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks, 20.0))

    def terminate(process, grace=10):
        process.returncode = -signal.SIGTERM

    monkeypatch.setattr(module, "terminate_group", terminate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_result_driver.py",
            "--run-name", "case",
            "--results-dir", str(tmp_path),
            "--timeout-seconds", str(timeout),
            "--min-available-gib", "4",
            "--model-id", "model-id",
            "--checkpoint-id", "checkpoint-id",
            "--", "fake-worker",
        ],
    )
    exit_code = module.main()
    record = json.loads((tmp_path / "case" / "run.json").read_text())
    return exit_code, record, process_holder["process"]


def test_success_normalizes_paths_and_adds_memory(monkeypatch, tmp_path):
    result = {**VALID_RESULT, "model": "/models/private", "checkpoint": "/checkpoints/private"}
    code, record, _ = invoke(monkeypatch, tmp_path, output="MEMRIFT_RESULT_JSON=" + json.dumps(result) + "\n")
    assert code == 0
    assert record["status"] == "success"
    assert record["result"]["model"] == "model-id"
    assert record["result"]["checkpoint"] == "checkpoint-id"
    assert record["result"]["peak_process_rss_bytes"] == 0
    assert record["command_file"] == "command.txt"


@pytest.mark.parametrize(
    ("output", "returncode", "status"),
    [
        ("ModuleNotFoundError: missing\n", 1, "dependency_failure"),
        ("unexpected traceback\n", 1, "software_failure"),
        ("ordinary output\n", 0, "validation_failure"),
        ("MEMRIFT_RESULT_JSON={bad json}\n", 0, "validation_failure"),
    ],
)
def test_exit_classification(monkeypatch, tmp_path, output, returncode, status):
    code, record, _ = invoke(monkeypatch, tmp_path, output=output, returncode=returncode)
    assert code == 1
    assert record["status"] == status


def test_watchdog_classifies_timeout_and_terminates(monkeypatch, tmp_path):
    code, record, process = invoke(monkeypatch, tmp_path, running=True, timeout=1)
    assert code == 1
    assert record["status"] == "timeout"
    assert process.returncode == -signal.SIGTERM


def test_watchdog_classifies_low_available_memory_as_memory_guard(monkeypatch, tmp_path):
    gib = 1024**3
    code, record, process = invoke(
        monkeypatch,
        tmp_path,
        running=True,
        memory=[(16 * gib, 12 * gib), (16 * gib, 3 * gib)],
    )
    assert code == 1
    assert record["status"] == "memory_guard"
    assert process.returncode == -signal.SIGTERM


def test_terminate_group_uses_term_then_kill(monkeypatch):
    module = load_module("scripts/run_result_driver.py", "result_driver_groups")
    process = FakeProcess(["worker"], stdout=type("Sink", (), {"write": lambda self, text: None, "flush": lambda self: None})(), running=True)
    signals = []
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    module.terminate_group(process, grace=0.01)
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]


def test_result_driver_rejects_unsafe_command_before_launch(monkeypatch, tmp_path):
    module = load_module("scripts/run_result_driver.py", "result_driver_validation")
    monkeypatch.setattr(
        sys,
        "argv",
        ["driver", "--run-name", "x", "--results-dir", str(tmp_path), "--model-id", "m", "--checkpoint-id", "c", "--", "worker", "--max_length", "4096"],
    )
    with pytest.raises(SystemExit):
        module.main()
