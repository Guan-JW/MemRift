import argparse

import pytest

from conftest import load_module


def test_lookahead_values_parse_and_reject_negative():
    module = load_module("experiments/lookahead/run.py", "lookahead_driver")
    assert module.parse_values("0,1,2,4,8") == [0, 1, 2, 4, 8]
    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_values("0,-1")


def test_lookahead_defaults_match_paper(tmp_path):
    module = load_module("experiments/lookahead/run.py", "lookahead_defaults")
    args = module.parse_args([
        "--model", "/model", "--model-id", "model", "--checkpoint", "/checkpoint",
        "--results-dir", str(tmp_path),
    ])
    assert args.lookaheads == [0, 1, 2, 4, 8]
    assert (args.context, args.batch_size) == (2048, 1)


def test_csv_accepts_normalized_metric(tmp_path):
    module = load_module("experiments/lookahead/run.py", "lookahead_csv")
    output = tmp_path / "out.csv"
    module.write_csv(output, [{"method": "memrift", "step_time_vs_qlora": 1.02}])
    assert "step_time_vs_qlora" in output.read_text()
