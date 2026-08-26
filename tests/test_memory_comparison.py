from argparse import Namespace

from conftest import load_module


def memory_module():
    return load_module("scripts/run_memory_comparison.py", "run_memory_comparison")


def rows(peaks):
    output = []
    for repetition in range(3):
        order = memory_module().method_order(repetition)
        for position, method in enumerate(order):
            output.append({
                "repetition": repetition + 1,
                "position": position + 1,
                "variant": method,
                "status": "ok",
                "ram_used_bytes_max": peaks[method][repetition],
                "system_available_bytes_min": 5 * 1024**3,
                "round_peak_gpu_alloc_bytes_max": 1024,
                "process_rss_bytes_max": 2048,
                "round_time_mean_sec": 1.0,
                "cpu_util_mean": 20.0,
                "gpu_util_mean": 80.0,
                "tegrastats_samples": 10,
                "rounds": 7,
                "warmup_rounds": 1,
                "context": 2048,
                "batch_size": 3,
                "dataset": "tatsu-lab/alpaca",
                "dataset_revision": "dce01c9b08f87459cf36a430d809084718273017",
                "synthetic_data": False,
                "gradient_checkpointing": False,
                "seed": 42,
                "activation_backend": "ebc-zstd",
                "activation_compression_level": 1,
                "weight_lookahead": 1,
                "activation_lookahead": 0,
                **module_contract(method),
            })
    return output


def module_contract(method):
    return memory_module().METHOD_CONTRACTS[method]


def profile():
    return {
        "repetitions": 3,
        "minimum_reduction_percent": 0,
        "gradient_checkpointing": False,
        "dataset": "tatsu-lab/alpaca",
        "dataset_revision": "dce01c9b08f87459cf36a430d809084718273017",
        "context": 2048,
        "batch_size": 3,
        "rounds": 7,
        "warmup_rounds": 1,
    }


def test_method_order_rotates_balanced_sequence():
    module = memory_module()
    assert module.method_order(0) == ("lora", "qlora", "memrift")
    assert module.method_order(1) == ("qlora", "memrift", "lora")
    assert module.method_order(2) == ("memrift", "lora", "qlora")


def test_summary_supports_lower_memrift_claim():
    module = memory_module()
    data = rows({"lora": [100, 101, 99], "qlora": [90, 91, 89], "memrift": [80, 81, 79]})
    summary = module.summarize(data, profile(), "sha256:test", "a" * 40, "b" * 40, [module.method_order(i) for i in range(3)])
    assert summary["status"] == "complete"
    assert summary["claim_supported"] is True
    assert summary["memrift_reduction_percent"]["lora"] == 20
    assert summary["methods"]["memrift"]["minimum_system_available_bytes_median"] == 5 * 1024**3


def test_summary_preserves_negative_result():
    module = memory_module()
    data = rows({"lora": [80, 81, 79], "qlora": [75, 76, 74], "memrift": [90, 91, 89]})
    summary = module.summarize(data, profile(), "sha256:test", "a" * 40, "b" * 40, [module.method_order(i) for i in range(3)])
    assert summary["status"] == "complete"
    assert summary["claim_supported"] is False
    assert summary["memrift_reduction_percent"]["lora"] < 0


def test_summary_rejects_duplicate_repetition():
    module = memory_module()
    data = rows({"lora": [100, 101, 99], "qlora": [90, 91, 89], "memrift": [80, 81, 79]})
    next(row for row in data if row["variant"] == "qlora" and row["repetition"] == 2)["repetition"] = 1
    summary = module.summarize(data, profile(), "sha256:test", "a" * 40, "b" * 40, [module.method_order(i) for i in range(3)])
    assert summary["status"] == "complete_with_failures"
    assert summary["claim_supported"] is False


def test_worker_records_runtime_source_revision():
    module = memory_module()
    args = Namespace(
        docker="docker", model="/model", checkpoint="/checkpoint", cache="/cache",
        results_root="/results", image="registry/image@sha256:test", image_digest="sha256:test",
        source_revision="a" * 40, dataset="dataset", dataset_revision="revision", context=2048,
        batch_size=3, rounds=7, warmup_rounds=1, timeout_seconds=1, min_available_mb=4096,
    )
    command = module.build_container_command(args, "comparison", 0, module.method_order(0), "b" * 40)
    assert f"MEMRIFT_GIT_REVISION={'b' * 40}" in command
    assert f"MEMRIFT_GIT_REVISION={'a' * 40}" not in command
