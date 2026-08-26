import argparse
import signal
import subprocess
import sys

import pytest

from conftest import load_module


@pytest.fixture
def gradient():
    module = load_module("experiments/gradient_checkpointing/run.py", "gradient_driver")
    module.USER_TERMINATED.clear()
    return module


def arguments(gradient, tmp_path):
    return gradient.parse_args([
        "--results-dir", str(tmp_path), "--disable-tegrastats", "--dataset-revision", "a" * 40
    ])


def test_memrift_gc_command_has_all_fixed_flags(gradient, tmp_path):
    args = arguments(gradient, tmp_path)
    command = gradient.build_command(args, "memrift_gc", 4096, 2, tmp_path / "run")
    assert command[0] == sys.executable
    for flag in ("--gradient_checkpointing", "--gc_keep_recompute_weights", "--gc_no_recompute_prefetch"):
        assert command.count(flag) == 1
    assert "--checkpoint" in command
    assert command[command.index("--dataset-revision") + 1] == "a" * 40


@pytest.mark.parametrize(
    ("returncode", "output", "kwargs", "expected"),
    [
        (0, 'MEMRIFT_RESULT_JSON {"rounds": 2}\n', {}, "ok"),
        (1, "CUDA out of memory", {}, "oom"),
        (1, "traceback", {"timed_out": True}, "timeout"),
        (1, "ModuleNotFoundError: x", {}, "dependency_failure"),
        (0, "no result", {}, "validation_failure"),
        (2, "run.py: error: LZ4 compression level must be between 0 and 16", {}, "validation_failure"),
        (2, "traceback", {}, "software_failure"),
        (1, "anything", {"user_terminated": True}, "user_termination"),
    ],
)
def test_classify_exit(gradient, returncode, output, kwargs, expected):
    assert gradient.classify_exit(returncode, output, **kwargs) == expected


def test_max_context_search_adjusts_bounds_for_success_and_oom(gradient, monkeypatch, tmp_path):
    args = arguments(gradient, tmp_path)
    args.max_context_low = 2048
    args.max_context_high = 3072
    args.context_step = 256
    seen = []
    statuses = iter(["ok", "oom"])

    def run_one(args, runner, variant, context, rounds, tag):
        seen.append(context)
        return {"variant": variant, "context": context, "status": next(statuses)}

    monkeypatch.setattr(gradient, "run_one", run_one)
    rows = gradient.max_context_search(args, object(), "lora_gc")
    assert seen == [2560, 2816]
    assert rows[-1]["status"] == "max_context"
    assert rows[-1]["context"] == 2560


def test_max_context_search_aborts_without_treating_software_failure_as_oom(gradient, monkeypatch, tmp_path):
    args = arguments(gradient, tmp_path)
    monkeypatch.setattr(gradient, "run_one", lambda *a, **k: {"variant": "lora_gc", "context": a[3], "status": "software_failure"})
    rows = gradient.max_context_search(args, object(), "lora_gc")
    assert [row["status"] for row in rows] == ["software_failure", "search_aborted"]


class WaitingProcess:
    pid = 5151

    def poll(self):
        return None

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired("worker", timeout)

    def terminate(self):
        raise AssertionError("PID fallback should not be used")

    def kill(self):
        raise AssertionError("PID fallback should not be used")


def test_gradient_process_group_termination(gradient, monkeypatch):
    sent = []
    monkeypatch.setattr(gradient.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    gradient.terminate_process_group(WaitingProcess(), grace_sec=0)
    assert sent == [(5151, signal.SIGTERM), (5151, signal.SIGKILL)]


def test_every_variant_has_a_resource_estimate(gradient, tmp_path):
    args = arguments(gradient, tmp_path)
    for variant in gradient.VARIANTS:
        assert gradient.resource_estimate(variant, 64, args)["variant"] == variant
