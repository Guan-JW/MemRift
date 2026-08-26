from conftest import load_module


def test_summary_preserves_scalar_metrics_and_environment():
    module = load_module("scripts/summarize_results.py", "result_summary")
    record = {
        "run_name": "run",
        "status": "success",
        "exit_code": 0,
        "model_logical_id": "model",
        "checkpoint_logical_id": "checkpoint",
        "started_at": "start",
        "ended_at": "end",
        "environment": {"container_image_digest": "sha256:test", "nested": {"ignored": True}},
        "result": {
            "round_time_mean_sec": 1.25,
            "peak_system_used_bytes": 42,
            "nested": {"ignored": True},
        },
    }

    row = module.summarize_record(record)

    assert row["result.round_time_mean_sec"] == 1.25
    assert row["result.peak_system_used_bytes"] == 42
    assert row["environment.container_image_digest"] == "sha256:test"
    assert "result.nested" not in row
