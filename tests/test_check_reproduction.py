import json

from conftest import load_module


def checker():
    return load_module("scripts/check_reproduction.py", "check_reproduction")


def add_memory_evidence(tmp_path, data):
    data["input_validation"] = {
        "model_revision": "de253fa9783f8bd558c9ed398c8ffbe3c55cedb3",
        "checkpoint_model_logical_id": "tinyllama-1.1b-chat-v1.0",
        "checkpoint_source_revision": "de253fa9783f8bd558c9ed398c8ffbe3c55cedb3",
        "checkpoint_format": "memrift-float-split-stride-v1",
        "checkpoint_zstd_level": 18,
        "checkpoint_payload_sha256": "f4dff6bb6c0017a8668e0263a24c311bc2891a2ed467ca58520aff30a5c9cdaa",
        "dataset_fingerprint": "fingerprint",
        "dataset_rows": 52002,
    }
    orders = [
        ("lora", "qlora", "memrift"),
        ("qlora", "memrift", "lora"),
        ("memrift", "lora", "qlora"),
    ]
    runs = []
    for repetition, order in enumerate(orders, 1):
        repetition_dir = tmp_path / f"rep-{repetition:02d}"
        repetition_dir.mkdir(parents=True)
        (repetition_dir / "environment.json").write_text(json.dumps({
            "git_revision": data["runtime_source_revision"],
            "container_image_digest": data["container_image_digest"],
        }))
        (repetition_dir / "resolved_config.json").write_text(json.dumps({
            "min_available_mb": 4096, "matched_context": 2048, "batch_size": 3,
            "rounds": 7, "warmup_rounds": 1, "variants": list(order),
            "activation_compaction_concurrency": 1, "activation_decode_concurrency": 1,
            "weight_materialization_concurrency": 1, "weight_lookahead": 1,
            "activation_lookahead": 0, "activation_backend": "ebc-zstd", "compression_level": 1,
        }))
        for position, method in enumerate(order, 1):
            run_dir = tmp_path / f"rep-{repetition:02d}" / method
            run_dir.mkdir(parents=True)
            command = [
                "python", "-u", "/workspace/src/train_memrift.py", "--model", "/models/model",
                "--dataset", "tatsu-lab/alpaca", "--dataset-revision", "dce01c9b08f87459cf36a430d809084718273017",
                "--dataset-cache", "/cache/huggingface",
                "--seed", "42", "--max_length", "2048", "--batch_size", "3", "--round", "7", "--warmup_rounds", "1",
                "--act_compact_concurrency", "1", "--act_decode_concurrency", "1", "--weight_async_concurrency", "1",
                "--weight_lookahead", "1", "--activation_lookahead", "0", "--activation-backend", "ebc-zstd",
                "--level", "1", "--tegra-csv", "tegrastats.csv",
            ]
            if method == "qlora":
                command.extend(["--finetune_type", "qlora", "--autocast_context"])
            elif method == "memrift":
                command.extend(["--hook", "--weight", "--weight_async", "--activation", "--act_async", "--checkpoint", "/checkpoints/model"])
            (run_dir / "command.json").write_text(json.dumps(command))
            peak = data["methods"][method]["peak_system_used_bytes"][repetition - 1]
            (run_dir / "tegrastats.csv").write_text(f"timestamp_ms,ram_used_MB\n0,{peak // 2**20}\n")
            contract = checker().MEMORY_METHOD_CONTRACTS[method]
            runs.append({
                "repetition": repetition, "position": position, "variant": method, "status": "ok",
                "context": 2048, "batch_size": 3, "rounds": 7, "warmup_rounds": 1,
                "dataset": "tatsu-lab/alpaca", "dataset_revision": "dce01c9b08f87459cf36a430d809084718273017",
                "synthetic_data": False, "gradient_checkpointing": False, "seed": 42,
                "activation_backend": "ebc-zstd", "activation_compression_level": 1,
                "weight_lookahead": 1, "activation_lookahead": 0, "tegrastats_samples": 1,
                "peak_system_used_bytes": peak,
                "run_dir": str(run_dir.relative_to(tmp_path)), **contract,
            })
    data["runs"] = runs


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


def test_memory_requires_completed_lower_memrift_medians(tmp_path):
    path = tmp_path / "memory.json"
    data = {
        "status": "complete",
        "profile": {
            "repetitions": 3,
            "gradient_checkpointing": False,
            "minimum_reduction_percent": 0,
            "model_logical_id": "tinyllama-1.1b-chat-v1.0",
            "dataset": "tatsu-lab/alpaca",
            "dataset_revision": "dce01c9b08f87459cf36a430d809084718273017",
            "context": 2048,
            "batch_size": 3,
            "rounds": 7,
            "warmup_rounds": 1,
            "activation_compaction_concurrency": 1,
            "activation_decode_concurrency": 1,
            "weight_materialization_concurrency": 1,
        },
        "methods": {
            "lora": {"successful_runs": 3, "peak_system_used_bytes": [100 * 2**20, 101 * 2**20, 99 * 2**20], "peak_system_used_bytes_median": 100 * 2**20},
            "qlora": {"successful_runs": 3, "peak_system_used_bytes": [90 * 2**20, 91 * 2**20, 89 * 2**20], "peak_system_used_bytes_median": 90 * 2**20},
            "memrift": {"successful_runs": 3, "peak_system_used_bytes": [80 * 2**20, 81 * 2**20, 79 * 2**20], "peak_system_used_bytes_median": 80 * 2**20},
        },
        "memrift_reduction_percent": {"lora": 20.0, "qlora": (90 - 80) / 90 * 100},
        "claim_supported": True,
        "method_order_valid": True,
        "orchestrator_source_revision": "a" * 40,
        "runtime_source_revision": "b" * 40,
        "container_image_digest": "sha256:" + "c" * 64,
    }
    add_memory_evidence(tmp_path, data)
    path.write_text(json.dumps(data))
    assert checker().main(["--experiment", "memory", "--input", str(path)]) == 0
    telemetry = tmp_path / data["runs"][0]["run_dir"] / "tegrastats.csv"
    original_telemetry = telemetry.read_text()
    telemetry.write_text("timestamp_ms,ram_used_MB\n0,999\n")
    assert checker().main(["--experiment", "memory", "--input", str(path)]) == 1
    telemetry.write_text(original_telemetry)
    data["methods"]["qlora"]["successful_runs"] = 2
    data["claim_supported"] = False
    path.write_text(json.dumps(data))
    assert checker().main(["--experiment", "memory", "--input", str(path)]) == 1


def test_memory_recomputes_reductions_from_peaks(tmp_path):
    path = tmp_path / "memory.json"
    data = {
        "status": "complete",
        "profile": {
            "repetitions": 3, "gradient_checkpointing": False, "minimum_reduction_percent": 0,
            "model_logical_id": "tinyllama-1.1b-chat-v1.0", "dataset": "tatsu-lab/alpaca",
            "dataset_revision": "dce01c9b08f87459cf36a430d809084718273017",
            "context": 2048, "batch_size": 3, "rounds": 7, "warmup_rounds": 1,
            "activation_compaction_concurrency": 1, "activation_decode_concurrency": 1,
            "weight_materialization_concurrency": 1,
        },
        "methods": {
            "lora": {"successful_runs": 3, "peak_system_used_bytes": [80 * 2**20, 81 * 2**20, 79 * 2**20], "peak_system_used_bytes_median": 80 * 2**20},
            "qlora": {"successful_runs": 3, "peak_system_used_bytes": [75 * 2**20, 76 * 2**20, 74 * 2**20], "peak_system_used_bytes_median": 75 * 2**20},
            "memrift": {"successful_runs": 3, "peak_system_used_bytes": [90 * 2**20, 91 * 2**20, 89 * 2**20], "peak_system_used_bytes_median": 90 * 2**20},
        },
        "memrift_reduction_percent": {"lora": 20, "qlora": 20},
        "claim_supported": True,
        "method_order_valid": True,
        "orchestrator_source_revision": "a" * 40,
        "runtime_source_revision": "b" * 40,
        "container_image_digest": "sha256:" + "c" * 64,
    }
    add_memory_evidence(tmp_path, data)
    path.write_text(json.dumps(data))
    assert checker().main(["--experiment", "memory", "--input", str(path)]) == 1
