import copy
import json

import pytest

from conftest import ROOT


jsonschema = pytest.importorskip("jsonschema")


def schema(name):
    return json.loads((ROOT / "manifests" / name).read_text())


def test_runtime_result_schema_accepts_current_memory_fields():
    result = {
        "batch_size": 1,
        "max_length": 2048,
        "rounds": 2,
        "warmup_rounds": 1,
        "round_time_mean_sec": 1.25,
        "peak_torch_allocated_bytes": 100,
        "peak_torch_reserved_bytes": 120,
        "peak_process_rss_bytes": 200,
        "peak_system_used_bytes": 300,
        "minimum_system_available_bytes": 400,
    }
    jsonschema.validate(result, schema("result.schema.json"))


@pytest.mark.parametrize("field", ["batch_size", "max_length", "rounds", "round_time_mean_sec"])
def test_runtime_result_schema_rejects_invalid_required_values(field):
    result = {
        "batch_size": 1,
        "max_length": 1,
        "rounds": 1,
        "warmup_rounds": 0,
        "round_time_mean_sec": 0,
    }
    result[field] = -1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(result, schema("result.schema.json"))


def test_run_record_schema_accepts_portable_paths_and_all_statuses():
    base = {
        "schema_version": "1.0",
        "run_name": "test",
        "status": "success",
        "exit_code": 0,
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:01+00:00",
        "model_logical_id": "model",
        "checkpoint_logical_id": "checkpoint",
        "command_file": "command.txt",
        "raw_log": "raw.log",
        "result": None,
        "environment": {},
    }
    statuses = ("success", "oom", "timeout", "dependency_failure", "validation_failure", "software_failure", "user_termination")
    for status in statuses:
        record = copy.copy(base)
        record["status"] = status
        jsonschema.validate(record, schema("run-record.schema.json"))
    invalid = copy.copy(base)
    invalid["raw_log"] = "/host/raw.log"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema("run-record.schema.json"))


def test_example_model_manifest_matches_schema():
    document = json.loads((ROOT / "configs/models.example.json").read_text())
    jsonschema.validate(document, schema("model-manifest.schema.json"))
