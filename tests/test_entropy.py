import pytest

from conftest import load_module


torch = pytest.importorskip("torch")


def test_shannon_entropy_for_constant_and_balanced_symbols():
    module = load_module("experiments/entropy/collect.py", "entropy_collect")
    assert module.shannon_entropy(torch.zeros(8, dtype=torch.uint8), 2) == 0
    assert module.shannon_entropy(torch.tensor([0, 1] * 4), 2) == pytest.approx(1.0)


def test_bf16_field_entropy_is_finite():
    module = load_module("experiments/entropy/collect.py", "entropy_fields")
    values = module.tensor_entropies(torch.tensor([0.0, 1.0, -1.0], dtype=torch.bfloat16))
    assert set(values) == {"raw_per_8", "sign_per_1", "exponent_per_8", "mantissa_per_7"}
    assert all(0 <= value <= 8 for value in values.values())
