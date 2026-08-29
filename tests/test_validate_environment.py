import json
import sys
import types

from conftest import load_module


def fake_torch(cuda_available=True, capability=(8, 7)):
    cuda = types.SimpleNamespace(
        is_available=lambda: cuda_available,
        get_device_capability=lambda index: capability,
        get_device_name=lambda index: "Mock Orin",
    )
    return types.SimpleNamespace(__version__="2.6.0-mock", version=types.SimpleNamespace(cuda="12.6"), cuda=cuda)


def run_validation(module, monkeypatch, capsys, tmp_path, torch):
    extension = types.SimpleNamespace(__file__="/mock/float_split_stride/_ext.so")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: extension)
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(sys, "argv", ["validate_environment.py", "--allow-non-jetson", "--results", str(tmp_path)])
    code = module.main()
    return code, json.loads(capsys.readouterr().out)


def test_allow_non_jetson_validation_succeeds_with_mock_cuda(monkeypatch, capsys, tmp_path):
    module = load_module("scripts/validate_environment.py", "environment_success")
    code, report = run_validation(module, monkeypatch, capsys, tmp_path, fake_torch())
    assert code == 0
    assert report["ok"] is True
    assert report["architecture"] == "x86_64"
    assert report["compute_capability"] == [8, 7]
    assert report["extension"] == "_ext.so"
    assert any("tegrastats" in warning for warning in report["warnings"])
    assert not (tmp_path / ".memrift-write-test").exists()


def test_mock_validation_reports_cuda_and_mount_failures(monkeypatch, capsys, tmp_path):
    module = load_module("scripts/validate_environment.py", "environment_failure")
    missing_model = tmp_path / "model"
    missing_checkpoint = tmp_path / "checkpoint"
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_available=False))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: types.SimpleNamespace(__file__="extension.so"))
    monkeypatch.setattr(module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_environment.py", "--allow-non-jetson", "--results", str(tmp_path / "results"), "--require-model", str(missing_model), "--require-checkpoint", str(missing_checkpoint)],
    )
    code = module.main()
    report = json.loads(capsys.readouterr().out)
    assert code == 2
    assert report["ok"] is False
    assert any("CUDA support is unavailable" in error for error in report["errors"])
    assert any("lacks config.json" in error for error in report["errors"])
    assert any("lacks index.json" in error for error in report["errors"])


def test_dataset_receipt_reports_fingerprint_and_row_count_separately(tmp_path):
    module = load_module("scripts/validate_environment.py", "dataset_receipt_details")
    (tmp_path / "data.arrow").write_bytes(b"arrow")
    (tmp_path / "memrift-dataset-receipt.json").write_text(json.dumps({
        "datasets": [{
            "huggingface_id": "tatsu-lab/alpaca", "revision": "a" * 40,
            "fingerprint": "", "num_rows": 0,
        }]
    }))
    errors = module.validate_dataset_cache(tmp_path, "tatsu-lab/alpaca", "a" * 40)
    assert errors == [
        "dataset receipt is missing a fingerprint",
        "dataset receipt row count must be a positive integer",
    ]
