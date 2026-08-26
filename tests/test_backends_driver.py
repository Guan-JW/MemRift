from conftest import load_module


def test_table6_defaults_to_pinned_alpaca(tmp_path):
    module = load_module("experiments/backends/run.py", "backends_driver")
    args = module.parse_args([
        "--model", "/model", "--model-id", "model", "--checkpoint", "/checkpoint",
        "--results-dir", str(tmp_path),
    ])
    assert args.dataset == "tatsu-lab/alpaca"
    assert args.dataset_revision == "dce01c9b08f87459cf36a430d809084718273017"
    assert args.compression_level == 1
    assert args.seed == 42


def test_table6_rejects_level_invalid_for_lz4(tmp_path):
    module = load_module("experiments/backends/run.py", "backends_level")
    try:
        module.parse_args([
            "--model", "/model", "--model-id", "model", "--checkpoint", "/checkpoint",
            "--results-dir", str(tmp_path), "--compression-level", "-1",
        ])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("invalid matrix compression level was accepted")


def test_table6_row_normalizes_time_and_memory():
    module = load_module("experiments/backends/run.py", "backends_rows")
    row = module.table_row("model", "ebc-zstd", {
        "status": "ok", "round_time_mean_sec": 10.2,
        "ram_used_bytes_max": 20 * 2**30,
        "activation_compression_ratio": 1.48,
        "activation_original_bytes": 200, "activation_stored_bytes": 135,
        "run_dir": "runs/example",
    }, 10.0)
    assert row["peak_system_memory_gib"] == 20
    assert row["normalized_step_time"] == 1.02
    assert row["weight_backend"] == "ebc-zstd"


def test_shared_driver_passes_backend_to_runtime(tmp_path):
    module = load_module("experiments/backends/run.py", "backends_command")
    driver = module.load_driver()
    args = driver.parse_args(["--results-dir", str(tmp_path), "--activation-backend", "lz4"])
    command = driver.build_command(args, "memrift", 2048, 2, tmp_path)
    index = command.index("--activation-backend")
    assert command[index + 1] == "lz4"
