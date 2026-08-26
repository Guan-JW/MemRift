from conftest import load_module


def test_ablation_requires_explicit_context(tmp_path):
    module = load_module("experiments/ablation/run.py", "ablation_driver")
    args = module.parse_args([
        "--model", "/model", "--model-id", "model", "--checkpoint", "/checkpoint",
        "--results-dir", str(tmp_path), "--context", "3000",
    ])
    assert args.context == 3000
    assert args.dataset == "tatsu-lab/alpaca"


def test_shared_driver_success_status_is_ok():
    module = load_module("experiments/ablation/run.py", "ablation_status")
    driver = module.load_driver()
    assert driver.classify_exit(0, 'MEMRIFT_RESULT_JSON {"rounds": 2}\n') == "ok"
