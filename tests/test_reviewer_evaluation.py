import json
from argparse import Namespace

import pytest

from conftest import load_module


def reviewer_module():
    return load_module("scripts/run_reviewer_evaluation.py", "run_reviewer_evaluation")


def write_loading_receipts(root, module):
    common = {
        "schema_version": "1.0", "model_logical_id": "tinyllama-1.1b-chat-v1.0",
        "source_revision": "de253fa9783f8bd558c9ed398c8ffbe3c55cedb3",
        "source_weight_sha256": "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
    }
    for method, level in (("nf4", None), ("memrift", 3)):
        directory = root / method
        directory.mkdir(parents=True)
        (directory / "payload.bin").write_bytes(method.encode("ascii"))
        (directory / "preparation.json").write_text(json.dumps({
            **common, "method": method, "zstd_level": level,
            "prepared_directory_sha256": module.directory_sha256(directory),
        }))


def test_memory_valid_negative_is_valid_evidence():
    module = reviewer_module()
    checks = [{"name": name, "ok": True} for name in module.REQUIRED_MEMORY_CHECKS]
    checks.extend([
        {"name": "memrift_vs_lora", "ok": False},
        {"name": "memrift_vs_qlora", "ok": False},
        {"name": "claim_supported", "ok": False},
    ])
    report = {
        "ok": False,
        "checks": checks,
    }
    assert module.classify_check("memory", report) == (True, "valid_negative", "not_supported")


def test_memory_integrity_failure_is_not_valid_negative():
    module = reviewer_module()
    report = {
        "ok": False,
        "checks": [
            {"name": "raw_matched_runs", "ok": False},
            {"name": "claim_supported", "ok": False},
        ],
    }
    assert module.classify_check("memory", report) == (False, "invalid_evidence", "not_supported")


def test_memory_negative_requires_all_claim_checks():
    module = reviewer_module()
    checks = [{"name": name, "ok": True} for name in module.REQUIRED_MEMORY_CHECKS]
    checks.append({"name": "claim_supported", "ok": False})
    assert module.classify_check("memory", {"ok": False, "checks": checks}) == (
        False, "invalid_evidence", "not_supported",
    )


def test_stage_summary_only_reports_claim_for_memory():
    module = reviewer_module()
    assert module.stage_summary({"stage": "correctness", "outcome": "passed"}) == {
        "stage": "correctness", "outcome": "passed",
    }
    assert module.stage_summary({"stage": "memory", "outcome": "valid_negative", "claim": "not_supported"}) == {
        "stage": "memory", "outcome": "valid_negative", "claim": "not_supported",
    }


def test_completed_result_supports_resume(tmp_path):
    module = reviewer_module()
    first = tmp_path / "attempt-001"
    second = tmp_path / "attempt-002"
    first.mkdir()
    second.mkdir()
    (first / "outputs").mkdir()
    (second / "outputs").mkdir()
    (first / "result.json").write_text(json.dumps({"evidence_valid": False, "outcome": "execution_failure"}))
    expected = {
        "evidence_valid": True,
        "outcome": "valid_negative",
        "outputs_sha256": module.hash_tree(second / "outputs"),
    }
    (second / "result.json").write_text(json.dumps(expected))
    assert module.completed_result(tmp_path) == expected


def test_latest_failed_attempt_prevents_stale_resume(tmp_path):
    module = reviewer_module()
    first = tmp_path / "attempt-001"
    second = tmp_path / "attempt-002"
    first.mkdir()
    second.mkdir()
    (first / "outputs").mkdir()
    (second / "outputs").mkdir()
    (first / "result.json").write_text(json.dumps({
        "evidence_valid": True, "outcome": "passed",
        "outputs_sha256": module.hash_tree(first / "outputs"),
    }))
    (second / "result.json").write_text(json.dumps({"evidence_valid": False, "outcome": "execution_failure"}))
    assert module.completed_result(tmp_path) is None


def test_interrupted_latest_attempt_prevents_stale_resume(tmp_path):
    module = reviewer_module()
    first = tmp_path / "attempt-001"
    second = tmp_path / "attempt-002"
    first.mkdir()
    second.mkdir()
    (first / "outputs").mkdir()
    (second / "started.json").write_text("{}")
    (first / "result.json").write_text(json.dumps({
        "evidence_valid": True, "outcome": "passed",
        "outputs_sha256": module.hash_tree(first / "outputs"),
    }))
    assert module.completed_result(tmp_path) is None


def test_core_default_and_optional_stage_contracts(tmp_path):
    module = reviewer_module()
    assert module.DEFAULT_STAGES == ("validate", "correctness", "smoke", "memory")
    assert module.OPTIONAL_STAGES == ("loading", "entropy", "backends")
    args = Namespace(
        image="image", image_digest="sha256:test", model="/model", checkpoint="/checkpoint",
        cache="/cache", loading_checkpoint="/loading",
    )
    loading = module.stage_command(args, "loading", tmp_path)
    backends = module.stage_command(args, "backends", tmp_path)
    assert loading[3] == "model-loading"
    assert "LOADING_CHECKPOINT_DIR=/loading" in loading
    assert "LOADING_RUNS=5" in loading
    assert backends[3] == "backends"
    assert "TABLE6_CONTEXT=2048" in backends
    assert "TABLE6_WARMUP_ROUNDS=2" in backends
    assert module.primary_output("loading", tmp_path) == tmp_path / "model-loading/tinyllama-1.1b-chat-v1.0/summary.json"
    assert module.primary_output("entropy", tmp_path) == tmp_path / "table1-tinyllama-1.1b-chat-v1.0.csv"
    assert module.primary_output("backends", tmp_path) == tmp_path / "table6-tinyllama-1.1b-chat-v1.0/table6_backends.csv"


def test_full_profile_requires_valid_loading_checkpoint(tmp_path):
    module = reviewer_module()
    model = tmp_path / "model"
    checkpoint = tmp_path / "checkpoint"
    cache = tmp_path / "cache"
    loading = tmp_path / "loading"
    for path in (model, checkpoint, cache, loading):
        path.mkdir()
    base = [
        "--image", "image@sha256:" + "a" * 64,
        "--model", str(model), "--checkpoint", str(checkpoint), "--cache", str(cache),
        "--results-root", str(tmp_path / "results"), "--full",
    ]
    with pytest.raises(SystemExit):
        module.parse_args(base)
    write_loading_receipts(loading, module)
    args = module.parse_args([*base, "--loading-checkpoint", str(loading)])
    assert args.stage_names == module.ALL_STAGES
    (loading / "nf4/payload.bin").write_bytes(b"changed")
    with pytest.raises(SystemExit):
        module.parse_args([*base, "--loading-checkpoint", str(loading)])
