import json

from conftest import load_module


def reviewer_module():
    return load_module("scripts/run_reviewer_evaluation.py", "run_reviewer_evaluation")


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
