import pytest

from conftest import load_module


torch = pytest.importorskip("torch")


def test_bitwise_equality_handles_values_and_shape():
    module = load_module("experiments/fidelity/roundtrip.py", "fidelity_bits")
    source = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
    assert module.bitwise_equal(source, source.clone())
    changed = source.clone()
    changed[0] = 2
    assert not module.bitwise_equal(source, changed)
    assert not module.bitwise_equal(source, source.reshape(2, 1))


def test_fidelity_counts_report_bytes_and_mismatches():
    module = load_module("experiments/fidelity/roundtrip.py", "fidelity_counts")
    counts = module.FidelityCounts()
    source = torch.tensor([1.0], dtype=torch.bfloat16)
    counts.add(source, source.clone())
    counts.add(source, torch.tensor([2.0], dtype=torch.bfloat16))
    assert counts.as_dict() == {"tensors": 2, "bytes": 4, "mismatches": 1}
