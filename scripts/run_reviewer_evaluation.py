#!/usr/bin/env python3
"""Run the core reviewer workflow with resumable, structured progress records."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGES = ("validate", "correctness", "memory")
OPTIONAL_STAGES = ("loading", "entropy", "backends")
ALL_STAGES = DEFAULT_STAGES + OPTIONAL_STAGES
STAGE_TARGETS = {
    "validate": "validate", "correctness": "correctness-quick", "smoke": "smoke",
    "memory": "memory-comparison", "loading": "model-loading", "entropy": "entropy",
    "backends": "backends",
}
STAGE_CHECKERS = {
    "correctness": "fidelity", "smoke": "smoke", "memory": "memory",
    "loading": "loading", "entropy": "entropy", "backends": "backends",
}
CLAIM_CHECKS = {"memrift_vs_lora", "memrift_vs_qlora", "claim_supported"}
LOADING_CLAIM_CHECKS = {"memrift_vs_qlora_online"}
LOADING_MEDIAN_CHECKS = {
    "lora_median_seconds", "qlora-online_median_seconds",
    "qlora-prequant_median_seconds", "memrift_median_seconds",
}
REQUIRED_LOADING_CHECKS = {
    "methods", "lora_runs", "lora_cache", "qlora-online_runs",
    "qlora-online_cache", "qlora-prequant_runs", "qlora-prequant_cache",
    "memrift_runs", "memrift_cache", "serialized_nf4_path", "online_nf4_path",
    *LOADING_MEDIAN_CHECKS,
}
REQUIRED_MEMORY_CHECKS = {
    "status", "methods", "repetitions", "review_profile", "gradient_checkpointing",
    "minimum_reduction", "method_order", "orchestrator_source_revision",
    "runtime_source_revision", "container_image_digest", "model_revision",
    "checkpoint_model_logical_id", "checkpoint_source_revision", "checkpoint_format",
    "checkpoint_zstd_level", "checkpoint_payload_sha256", "dataset_fingerprint", "dataset_rows",
    "rep_01_environment", "rep_01_configuration", "rep_02_environment", "rep_02_configuration",
    "rep_03_environment", "rep_03_configuration",
    "raw_matched_runs", "lora_successful_runs", "lora_system_peaks", "lora_median",
    "lora_raw_peaks", "qlora_successful_runs", "qlora_system_peaks", "qlora_median",
    "qlora_raw_peaks", "memrift_successful_runs", "memrift_system_peaks",
    "memrift_median", "memrift_raw_peaks", "memrift_vs_lora_consistent",
    "memrift_vs_qlora_consistent",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(path, stage, event, **details):
    record = {"time": utc_now(), "stage": stage, "event": event, **details}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def hash_tree(root):
    digest = hashlib.sha256()
    if not root.is_dir():
        return None
    for path in sorted(
        candidate for candidate in root.rglob("*")
        if candidate.is_file() and ".git" not in candidate.relative_to(root).parts
    ):
        relative = str(path.relative_to(root))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def file_sha256(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree_for_pattern(root, pattern):
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob(pattern) if path.is_file())
    if not paths:
        return None
    for path in paths:
        relative = str(path.relative_to(root))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def directory_sha256(root, excluded_names=()):
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in excluded_names)
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "little"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def next_attempt(stage_dir):
    attempts = [int(path.name.split("-")[-1]) for path in stage_dir.glob("attempt-[0-9][0-9][0-9]")]
    return max(attempts, default=0) + 1


def completed_result(stage_dir):
    attempts = sorted(stage_dir.glob("attempt-[0-9][0-9][0-9]"), reverse=True)
    if not attempts:
        return None
    path = attempts[0] / "result.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if result.get("evidence_valid") is True and result.get("outputs_sha256") == hash_tree(path.parent / "outputs"):
        return result
    return None


def validate_loading_checkpoint(root):
    expected = {
        "model_logical_id": "tinyllama-1.1b-chat-v1.0",
        "source_revision": "de253fa9783f8bd558c9ed398c8ffbe3c55cedb3",
        "source_weight_sha256": "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
    }
    for method, level in (("nf4", None), ("memrift", 3)):
        path = root / method / "preparation.json"
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid {method} loading checkpoint receipt: {exc}") from exc
        valid = (
            receipt.get("schema_version") == "1.0" and receipt.get("method") == method
            and receipt.get("zstd_level") == level
            and all(receipt.get(key) == value for key, value in expected.items())
            and isinstance(receipt.get("prepared_directory_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", receipt["prepared_directory_sha256"])
            and receipt.get("prepared_directory_sha256") == directory_sha256(root / method, {"preparation.json"})
        )
        if not valid:
            raise ValueError(f"{method} loading checkpoint receipt does not match the pinned model")


def stage_command(args, stage, outputs):
    common = [
        "make", "-s", "--no-print-directory", STAGE_TARGETS[stage],
        f"TAG={args.image}", f"MODEL_DIR={args.model}", f"CHECKPOINT_DIR={args.checkpoint}",
        f"CACHE_DIR={args.cache}", f"RESULTS_DIR={outputs}",
        f"MEMRIFT_IMAGE_DIGEST={args.image_digest}",
        "MODEL_NAME=tinyllama-1.1b-chat-v1.0", "DATASET_ID=tatsu-lab/alpaca",
        "DATASET_REVISION=dce01c9b08f87459cf36a430d809084718273017",
        "CONTEXT_TOKENS=2048", "BATCH_SIZE=1", "ROUNDS=7", "WARMUP_ROUNDS=1",
        "MEMORY_CONTEXT=2048", "MEMORY_BATCH_SIZE=3", "MEMORY_REPETITIONS=3",
        "MEMORY_MIN_REDUCTION_PERCENT=0", "MIN_AVAILABLE_MB=4096",
        "MIN_AVAILABLE_GIB=4", "TIMEOUT_SECONDS=2400",
        "MODEL_LOGICAL_ID=tinyllama-1.1b-chat-v1.0",
        "CHECKPOINT_LOGICAL_ID=tinyllama-1.1b-chat-v1.0-memrift", "DOCKER=docker",
        "LOADING_RUNS=5", "TABLE6_CONTEXT=2048", "TABLE6_WARMUP_ROUNDS=2",
        "TABLE6_LEVEL=1",
    ]
    if stage == "loading":
        common.append(f"LOADING_CHECKPOINT_DIR={args.loading_checkpoint}")
    return common


def run_streamed(command, log_path, capture=False):
    lines = [] if capture else None
    with log_path.open("w", encoding="utf-8") as log:
        environment = os.environ.copy()
        for name in ("MAKEFLAGS", "MFLAGS", "GNUMAKEFLAGS"):
            environment.pop(name, None)
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if lines is not None:
                lines.append(line)
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return process.wait(), "".join(lines) if lines is not None else None


def primary_output(stage, outputs):
    if stage == "correctness":
        return outputs / "correctness-quick-tinyllama-1.1b-chat-v1.0.json"
    if stage == "memory":
        return outputs / "memory-comparison-tinyllama-1.1b-chat-v1.0" / "summary.json"
    if stage == "loading":
        return outputs / "model-loading" / "tinyllama-1.1b-chat-v1.0" / "summary.json"
    if stage == "entropy":
        return outputs / "table1-tinyllama-1.1b-chat-v1.0.csv"
    if stage == "backends":
        return outputs / "table6-tinyllama-1.1b-chat-v1.0" / "table6_backends.csv"
    if stage == "smoke":
        candidates = sorted(outputs.glob("smoke-*/run.json"))
        return candidates[0] if len(candidates) == 1 else None
    return None


def classify_check(stage, report):
    checks = {item.get("name"): item.get("ok") for item in report.get("checks", [])}
    failed = {name for name, ok in checks.items() if not ok}
    if report.get("ok") is True:
        return True, "passed", "supported" if stage in {"memory", "loading"} else None
    integrity_complete = (
        (REQUIRED_MEMORY_CHECKS | CLAIM_CHECKS).issubset(checks)
        and all(checks[name] for name in REQUIRED_MEMORY_CHECKS)
    )
    if stage == "memory" and integrity_complete and failed and failed.issubset(CLAIM_CHECKS):
        return True, "valid_negative", "not_supported"
    loading_integrity_complete = (
        (REQUIRED_LOADING_CHECKS | LOADING_CLAIM_CHECKS).issubset(checks)
        and all(checks[name] for name in REQUIRED_LOADING_CHECKS)
    )
    if stage == "loading" and loading_integrity_complete and failed and failed.issubset(LOADING_CLAIM_CHECKS):
        return True, "valid_negative", "not_supported"
    return False, "requirements_not_met", "not_evaluated" if stage in {"memory", "loading"} else None


def report_issues(report):
    if isinstance(report.get("unmet_requirements"), list):
        return report["unmet_requirements"]
    if isinstance(report.get("errors"), list):
        return [{"requirement": "environment", "reason": error} for error in report["errors"]]
    return [
        {"requirement": item.get("name"), "observed": item.get("observed"), "expected": item.get("expected")}
        for item in report.get("checks", []) if item.get("ok") is not True
    ]


def check_stage(stage, output, attempt_dir, validate_text=None):
    if stage == "validate":
        try:
            report = json.loads(validate_text or "")
        except json.JSONDecodeError:
            report = {"ok": False, "error": "validation output was not one JSON document"}
        write_json(attempt_dir / "check.json", report)
        if report.get("ok") is True:
            metrics = {key: report.get(key) for key in ("device", "compute_capability", "cuda", "l4t_release")}
            return True, "passed", None, metrics, []
        issues = report_issues(report)
        if not issues:
            issues = [{"requirement": "validation_output", "reason": report.get("error", "validation results are missing")}]
        return False, "requirements_not_met", None, {}, issues
    experiment = STAGE_CHECKERS[stage]
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_reproduction.py"), "--experiment", experiment, "--input", str(output)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        report = {"ok": False, "error": completed.stderr or "checker returned invalid JSON", "checks": []}
    write_json(attempt_dir / "check.json", report)
    evidence_valid, outcome, claim = classify_check(stage, report)
    issues = [] if evidence_valid else report_issues(report)
    if not issues and not evidence_valid:
        issues = [{"requirement": "checker_output", "reason": report.get("error", "checker results are missing")}]
    return evidence_valid, outcome, claim, report.get("metrics") or {}, issues


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--loading-checkpoint", type=Path)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--rerun", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    match = re.search(r"@(sha256:[0-9a-f]{64})$", args.image)
    if not match:
        parser.error("--image must be pinned by sha256 digest")
    args.image_digest = match.group(1)
    args.stage_names = ALL_STAGES if args.full else tuple(name.strip() for name in args.stages.split(",") if name.strip())
    unknown = set(args.stage_names) - set(ALL_STAGES)
    if unknown or not args.stage_names:
        parser.error(f"--stages must use: {', '.join(ALL_STAGES)}")
    unknown_reruns = set(args.rerun) - set(args.stage_names)
    if unknown_reruns:
        parser.error(f"--rerun names stages not selected: {sorted(unknown_reruns)}")
    for path in (args.model, args.checkpoint, args.cache):
        if not path.is_dir():
            parser.error(f"input directory does not exist: {path}")
    if "loading" in args.stage_names:
        if args.loading_checkpoint is None or not args.loading_checkpoint.is_dir():
            parser.error("--loading-checkpoint must name an existing directory when loading is selected")
        try:
            validate_loading_checkpoint(args.loading_checkpoint)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def stage_summary(result):
    summary = {key: result.get(key) for key in ("stage", "outcome")}
    if result.get("stage") in {"memory", "loading"}:
        summary["claim"] = result.get("claim")
    if result.get("metrics"):
        summary["metrics"] = result["metrics"]
    if result.get("unmet_requirements"):
        summary["unmet_requirements"] = result["unmet_requirements"]
        summary["action"] = "correct the listed requirement and rerun"
    return summary


def main(argv=None):
    args = parse_args(argv)
    git_top_level = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if git_top_level.returncode == 0 and Path(git_top_level.stdout.strip()).resolve() == ROOT:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout.strip()
        git_checkout = True
    else:
        marker = ROOT / ".memrift-source-revision"
        revision = marker.read_text(encoding="ascii").strip() if marker.is_file() else ""
        git_checkout = False
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("cannot determine an exact source revision")
    if git_checkout and not args.dry_run:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=True,
        ).stdout
        if dirty:
            raise SystemExit("reviewer evaluation requires a clean source tree")
    input_fingerprints = {
        "model_sha256": hash_tree(args.model),
        "checkpoint_sha256": hash_tree(args.checkpoint),
        "dataset_arrow_sha256": hash_tree_for_pattern(args.cache, "*.arrow"),
        "dataset_receipt_sha256": file_sha256(args.cache / "memrift-dataset-receipt.json"),
    }
    if "loading" in args.stage_names:
        input_fingerprints["loading_checkpoint_sha256"] = hash_tree(args.loading_checkpoint)
    configuration_values = {
        "model": str(args.model.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "cache": str(args.cache.resolve()), "stages": args.stage_names,
        "input_fingerprints": input_fingerprints,
    }
    if "loading" in args.stage_names:
        configuration_values["loading_checkpoint"] = str(args.loading_checkpoint.resolve())
    configuration = json.dumps(configuration_values, sort_keys=True).encode("utf-8")
    configuration_id = hashlib.sha256(configuration).hexdigest()[:12]
    evaluation_id = f"reviewer-{revision[:12]}-{args.image_digest[7:19]}-{configuration_id}"
    evaluation_dir = args.results_root.resolve() / evaluation_id
    if git_checkout and not args.dry_run:
        try:
            evaluation_dir.relative_to(ROOT)
        except ValueError:
            pass
        else:
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", str(evaluation_dir)], cwd=ROOT, check=False,
            ).returncode == 0
            if not ignored:
                raise SystemExit("an in-repository RESULTS_DIR must be ignored by Git")
    commands = {
        stage: stage_command(args, stage, evaluation_dir / "stages" / stage / "attempt-001" / "outputs")
        for stage in args.stage_names
    }
    if args.dry_run:
        print(json.dumps({"evaluation_id": evaluation_id, "commands": commands}, indent=2))
        return 0

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    events = evaluation_dir / "events.jsonl"
    provenance = {
        "schema_version": "1.0", "created_at": utc_now(), "source_revision": revision,
        "container_image": args.image, "container_image_digest": args.image_digest,
        "model": str(args.model.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "cache": str(args.cache.resolve()), "stages": list(args.stage_names),
        "input_fingerprints": input_fingerprints,
    }
    if "loading" in args.stage_names:
        provenance["loading_checkpoint"] = str(args.loading_checkpoint.resolve())
    write_json(evaluation_dir / "provenance.json", provenance)

    stage_results = []
    failed = False
    total = len(args.stage_names)
    for index, stage in enumerate(args.stage_names, 1):
        stage_dir = evaluation_dir / "stages" / stage
        previous = completed_result(stage_dir)
        if previous and stage not in args.rerun:
            print(f"[{index}/{total}] {stage}: SKIPPED ({previous['outcome']})", flush=True)
            append_event(events, stage, "skipped", outcome=previous["outcome"])
            stage_results.append(previous)
            continue
        attempt = next_attempt(stage_dir)
        attempt_dir = stage_dir / f"attempt-{attempt:03d}"
        outputs = attempt_dir / "outputs"
        outputs.mkdir(parents=True)
        command = stage_command(args, stage, outputs)
        started = utc_now()
        print(f"[{index}/{total}] {stage}: START", flush=True)
        append_event(events, stage, "started", attempt=attempt, command=command)
        write_json(attempt_dir / "started.json", {"stage": stage, "attempt": attempt, "started_at": started, "command": command})
        exit_code, output_text = run_streamed(command, attempt_dir / "stdout.log", capture=stage == "validate")
        primary = primary_output(stage, outputs)
        if stage == "validate":
            evidence_valid, outcome, claim, metrics, issues = check_stage(stage, primary, attempt_dir, output_text)
        elif exit_code != 0:
            evidence_valid, outcome = False, "incomplete_run"
            claim = "not_evaluated" if stage in {"memory", "loading"} else None
            metrics = {}
            issues = [{
                "requirement": "stage_execution", "observed": f"exit code {exit_code}",
                "expected": "completed execution", "reason": "the run ended before complete results were produced",
            }]
        elif stage != "validate" and (primary is None or not primary.is_file()):
            evidence_valid, outcome = False, "missing_results"
            claim = "not_evaluated" if stage in {"memory", "loading"} else None
            metrics = {}
            issues = [{
                "requirement": "primary_output", "observed": str(primary) if primary else None,
                "expected": "completed result file", "reason": "the expected result was not produced",
            }]
        else:
            evidence_valid, outcome, claim, metrics, issues = check_stage(stage, primary, attempt_dir, output_text)
        result = {
            "schema_version": "1.0", "stage": stage, "attempt": attempt,
            "started_at": started, "ended_at": utc_now(), "command": command,
            "exit_code": exit_code, "execution": "success" if exit_code == 0 else "failure",
            "evidence_valid": evidence_valid, "outcome": outcome,
            "primary_output": str(primary.relative_to(evaluation_dir)) if primary else None,
            "outputs_sha256": hash_tree(outputs),
        }
        if metrics:
            result["metrics"] = metrics
        if issues:
            result["unmet_requirements"] = issues
        if stage in {"memory", "loading"}:
            result["claim"] = claim
        write_json(attempt_dir / "result.json", result)
        append_event(events, stage, "finished", attempt=attempt, outcome=outcome, evidence_valid=evidence_valid)
        print(f"[{index}/{total}] {stage}: {outcome.upper()}", flush=True)
        stage_results.append(result)
        if not evidence_valid:
            failed = True
            break

    evaluation = {
        "schema_version": "1.0", "evaluation_id": evaluation_id,
        "status": "incomplete" if failed else "complete", "updated_at": utc_now(),
        "stages": stage_results,
    }
    if failed:
        evaluation["action"] = "correct the listed requirement and rerun; completed stages will be reused"
    write_json(evaluation_dir / "evaluation.json", evaluation)
    print(json.dumps({"evaluation": str(evaluation_dir), "status": evaluation["status"], "stages": [
        stage_summary(item) for item in stage_results
    ]}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
