from pathlib import Path

import pytest

from conftest import load_module


def module():
    return load_module("experiments/tables23/run.py", "tables23_driver")


def test_tables23_requires_pinned_inputs(tmp_path):
    args = module().parse_args([
        "--model", "/model", "--checkpoint", "/checkpoint", "--loading-prepared", "/loading",
        "--dataset-cache", "/cache", "--results-dir", str(tmp_path), "--skip-loading",
    ])
    assert args.model_id == "tinyllama-1.1b-chat-v1.0"
    assert args.dataset_revision == "dce01c9b08f87459cf36a430d809084718273017"
    assert args.rounds == 7


def test_tables23_rejects_nonstandard_loading_count(tmp_path):
    with pytest.raises(SystemExit):
        module().parse_args([
            "--model", "/model", "--checkpoint", "/checkpoint", "--loading-prepared", "/loading",
            "--dataset-cache", "/cache", "--results-dir", str(tmp_path), "--loading-runs", "4",
        ])


def test_smoke_requires_nonreportable_controls(tmp_path):
    with pytest.raises(SystemExit):
        module().parse_args([
            "--model", "/model", "--checkpoint", "/checkpoint", "--loading-prepared", "/loading",
            "--dataset-cache", "/cache", "--results-dir", str(tmp_path), "--smoke",
        ])


def test_results_directory_is_transactional(tmp_path):
    target = module()
    marker = target.prepare_results(tmp_path / "run", False)
    assert marker.name == ".incomplete"
    with pytest.raises(FileExistsError):
        target.prepare_results(tmp_path / "run", False)
    replacement = target.prepare_results(tmp_path / "run", True)
    assert replacement.is_file()


def test_table_metrics_use_decimal_gb_and_paper_formulas():
    target = module()
    assert target.reduction_percent(28280, 24050) == pytest.approx(14.957567)
    assert target.relative_error(19.32, 19.32) == 0
    assert target.file_bytes(Path(__file__).parent) > 0
    assert len(target.directory_sha256(Path(__file__).parent)) == 64
