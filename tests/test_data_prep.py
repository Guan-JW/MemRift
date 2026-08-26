from conftest import load_module


def test_alpaca_format_and_revision_are_preserved(monkeypatch):
    module = load_module("src/data_prep.py", "alpaca_data")
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [
            {"instruction": "Short", "input": "", "output": "A"},
            {"instruction": "Explain", "input": "this input", "output": "A longer answer"},
        ]

    monkeypatch.setattr(module, "load_dataset", load_dataset)
    rows = module.data_prepare(
        "tatsu-lab/alpaca", 1, cache_dir="/cache", revision="a" * 40
    )

    assert rows == ["### Instruction:\nExplain\n\nInput:\nthis input\n\n### Response:\nA longer answer"]
    assert calls == [
        (("tatsu-lab/alpaca",), {"split": "train", "cache_dir": "/cache", "revision": "a" * 40})
    ]
