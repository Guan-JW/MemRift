import subprocess
import sys

import pytest

from conftest import ROOT, load_module


@pytest.mark.parametrize(
    "script",
    [
        "src/train_memrift.py",
        "scripts/run_result_driver.py",
        "scripts/prepare_weights.py",
        "scripts/validate_environment.py",
        "experiments/gradient_checkpointing/run.py",
        "experiments/model_loading/run_benchmarks.py",
        "experiments/model_loading/run_validation.py",
        "experiments/model_loading/loading_worker.py",
    ],
)
def test_help_does_not_load_models(script):
    completed = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--weight"], "--weight requires --hook"),
        (["--hook", "--weight"], "--checkpoint is required"),
        (["--weight_async"], "--weight_async requires --weight"),
        (["--activation"], "--activation requires --hook"),
        (["--act_async"], "--act_async requires --activation"),
        (["--device", "cpu"], "--device must name a CUDA device"),
        (["--round", "1", "--warmup_rounds", "1"], "less than --round"),
        (["--tegra-csv", "/tmp/raw.csv"], "must be a filename relative"),
    ],
)
def test_training_validation_precedes_heavy_imports(tmp_path, extra, message):
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/train_memrift.py"),
            "--model",
            "unused",
            "--results-dir",
            str(tmp_path),
            *extra,
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert message in completed.stderr


def test_training_rejects_missing_checkpoint_index_without_importing_model(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/train_memrift.py"),
            "--model",
            "unused",
            "--hook",
            "--weight",
            "--checkpoint",
            str(checkpoint),
        ],
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert "compressed checkpoint index is missing" in completed.stderr


def test_gradient_cli_validation_and_default_python():
    module = load_module("experiments/gradient_checkpointing/run.py", "gradient_cli")
    assert module.parse_args([]).python == sys.executable
    with pytest.raises(SystemExit):
        module.parse_args(["--rounds", "1", "--warmup-rounds", "1"])
    with pytest.raises(SystemExit):
        module.parse_args(["--run-max-context", "--max-context-low", "4", "--max-context-high", "2"])


def test_loading_parsers_default_to_current_interpreter():
    benchmark = load_module("experiments/model_loading/run_benchmarks.py", "loading_cli")
    validation = load_module("experiments/model_loading/run_validation.py", "validation_cli")
    benchmark_args = benchmark.build_parser().parse_args(
        ["--name", "n", "--model", "m", "--output-root", "o"]
    )
    validation_args = validation.build_parser().parse_args(
        ["--name", "n", "--model", "m", "--prepared", "p", "--output-root", "o"]
    )
    assert benchmark_args.python == sys.executable
    assert validation_args.python == sys.executable
