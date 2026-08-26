import argparse
import json
import sys
from pathlib import Path

import pytest

from conftest import load_module


def loading_row(method, value=1):
    return {
        "method": method,
        "load_to_ready_seconds": value,
        "peak_process_rss_bytes": value,
        "peak_process_rss_delta_bytes": value,
        "peak_system_used_bytes": value,
        "peak_system_used_delta_bytes": value,
        "peak_torch_allocated_bytes": value,
        "checkpoint_bytes": value,
        "online_quantized_tensor_calls": 0,
        "prequantized_tensor_calls": 0,
    }


def test_partial_method_summary_contains_only_selected_methods():
    module = load_module("experiments/model_loading/run_benchmarks.py", "loading_summary")
    rows = [loading_row("lora", 1), loading_row("memrift", 99)]
    summary = module.summarize_rows(rows, ("lora",))
    assert list(summary) == ["lora"]
    assert summary["lora"]["cache_state"] == "warm"
    assert summary["lora"]["cache_dropped"] is False


def test_prequantized_summary_enforces_serialized_nf4_path():
    module = load_module("experiments/model_loading/run_benchmarks.py", "loading_semantics")
    valid = loading_row("qlora-prequant")
    valid["prequantized_tensor_calls"] = 1
    module.summarize_rows([valid], ("qlora-prequant",))

    invalid = loading_row("qlora-prequant")
    with pytest.raises(ValueError, match="serialized NF4"):
        module.summarize_rows([invalid], ("qlora-prequant",))


def test_memrift_summary_requires_post_timing_forward_validation():
    module = load_module("experiments/model_loading/run_benchmarks.py", "loading_memrift_validation")
    valid = loading_row("memrift")
    valid["post_timing_forward_validated"] = True
    module.summarize_rows([valid], ("memrift",))
    with pytest.raises(ValueError, match="forward validation"):
        module.summarize_rows([loading_row("memrift")], ("memrift",))


def test_main_reads_only_current_outputs_and_uses_current_python(monkeypatch, tmp_path):
    module = load_module("experiments/model_loading/run_benchmarks.py", "loading_current")
    output_dir = tmp_path / "benchmark"
    output_dir.mkdir()
    (output_dir / "memrift-99.json").write_text(json.dumps(loading_row("memrift", 999)))
    commands = []

    def run(command, check):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        method = command[command.index("--method") + 1]
        output.write_text(json.dumps(loading_row(method, 3)))

    monkeypatch.setattr(module.subprocess, "run", run)
    module.main(["--name", "benchmark", "--model", "model", "--output-root", str(tmp_path), "--runs", "1", "--methods", "lora"])
    summary = json.loads((output_dir / "summary.json").read_text())
    assert list(summary) == ["lora"]
    assert summary["lora"]["runs"] == 1
    assert commands[0][0] == sys.executable


def test_validation_commands_use_current_python_and_separate_outputs(tmp_path):
    module = load_module("experiments/model_loading/run_validation.py", "loading_validation_commands")
    args = module.build_parser().parse_args(["--name", "case", "--model", "m", "--prepared", "p", "--output-root", str(tmp_path)])
    root, commands = module.build_commands(args, tmp_path / "scripts")
    assert all(command[0] == sys.executable for command in commands)
    assert root == tmp_path / "case"
    assert str(root / "online.json") in commands[0]
    assert str(root / "prequant.json") in commands[1]


def test_relative_logits_are_resolved_against_each_record(monkeypatch, tmp_path):
    module = load_module("experiments/model_loading/compare_validation.py", "relative_logits")
    online_file = tmp_path / "online" / "record.json"
    prequant_file = tmp_path / "prequant" / "record.json"
    loaded = []

    class FakeTensor:
        def __sub__(self, other): return self
        def abs(self): return self
        def max(self): return self
        def mean(self): return self
        def item(self): return 0.0

    class FakeTorch:
        @staticmethod
        def load(path, **kwargs):
            loaded.append(Path(path))
            return FakeTensor()

        @staticmethod
        def allclose(*args, **kwargs):
            return True

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    record = {"logits": "artifacts/logits.pt", "training_step_seconds": 1, "training_peak_torch_bytes": 1, "loss": 0}
    module.compare(record, record, online_file, prequant_file)
    assert loaded == [online_file.parent / "artifacts/logits.pt", prequant_file.parent / "artifacts/logits.pt"]


def test_validation_comparison_reports_failed_tolerance(monkeypatch, tmp_path):
    module = load_module("experiments/model_loading/compare_validation.py", "failed_tolerance")

    class FakeTensor:
        def __sub__(self, other): return self
        def abs(self): return self
        def max(self): return self
        def mean(self): return self
        def item(self): return 1.0

    class FakeTorch:
        @staticmethod
        def load(*args, **kwargs): return FakeTensor()

        @staticmethod
        def allclose(*args, **kwargs): return False

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    record = {"logits": "logits.pt", "training_step_seconds": 1, "training_peak_torch_bytes": 1, "loss": 0}
    result = module.compare(record, record, tmp_path / "online.json", tmp_path / "prequant.json")
    assert result["logits_allclose_atol_1e-5"] is False


def test_stale_preparation_output_is_rejected_and_overwrite_is_scoped(tmp_path):
    module = load_module("experiments/model_loading/prepare_checkpoints.py", "stale_outputs")
    stale = tmp_path / "memrift"
    stale.mkdir()
    (stale / "old.bin").write_bytes(b"old")
    untouched = tmp_path / "nf4"
    untouched.mkdir()
    (untouched / "keep.bin").write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="--overwrite"):
        module.prepare_output_directories(tmp_path, ("memrift",))
    paths = module.prepare_output_directories(tmp_path, ("memrift",), overwrite=True)
    assert list(paths["memrift"].iterdir()) == []
    assert (untouched / "keep.bin").read_bytes() == b"keep"


def test_checkpoint_method_contract_and_version_summary():
    module = load_module("experiments/model_loading/loading_worker.py", "checkpoint_contract")
    with pytest.raises(ValueError, match="required"):
        module.validate_checkpoint("memrift", None)
    with pytest.raises(ValueError, match="not used"):
        module.validate_checkpoint("lora", "checkpoint")
    module.validate_checkpoint("qlora-prequant", "checkpoint")
    with pytest.raises(RuntimeError) as error:
        module.require_exact_versions(lambda package: "0.0")
    assert "transformers==4.49.0 required" in str(error.value)
    assert "peft==0.14.0 required" in str(error.value)
