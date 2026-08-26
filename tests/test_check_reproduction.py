import json

from conftest import load_module


def checker():
    return load_module("scripts/check_reproduction.py", "check_reproduction")


def test_fidelity_accepts_exact_completed_roundtrip(tmp_path):
    path = tmp_path / "fidelity.json"
    path.write_text(json.dumps({
        "steps_requested": 100,
        "steps_completed": 100,
        "tensor_mismatches": 0,
        "weights": {"tensors": 201, "mismatches": 0},
        "activations": {"tensors": 497, "mismatches": 0},
    }))
    assert checker().main(["--experiment", "fidelity", "--input", str(path)]) == 0


def test_loading_requires_five_runs_and_tolerance(tmp_path):
    data = {}
    for method, expected in checker().LOADING_EXPECTED_SECONDS.items():
        data[method] = {
            "runs": 5,
            "cache_state": "warm",
            "cache_dropped": False,
            "load_to_ready_seconds_median": expected,
            "online_quantized_tensor_calls": 154 if method == "qlora-online" else 0,
            "prequantized_tensor_calls": 154 if method == "qlora-prequant" else 0,
        }
    path = tmp_path / "loading.json"
    path.write_text(json.dumps(data))
    assert checker().main(["--experiment", "loading", "--input", str(path)]) == 0
    data["memrift"]["runs"] = 1
    path.write_text(json.dumps(data))
    assert checker().main(["--experiment", "loading", "--input", str(path)]) == 1


def test_tables23_rejects_nonreportable_result(tmp_path):
    path = tmp_path / "tables23.json"
    path.write_text(json.dumps({
        "status": "complete_with_failures",
        "protocol_reportable": False,
        "reported_numeric_acceptance_met": False,
    }))
    assert checker().main(["--experiment", "tables23", "--input", str(path)]) == 1
