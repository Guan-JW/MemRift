import ast

from conftest import ROOT


SOURCE = ROOT / "src/train_memrift.py"


def calls_named(node, name):
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == name
    ]


def test_cleanup_does_not_reset_cuda_peak_state():
    tree = ast.parse(SOURCE.read_text())
    compressor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AsyncCompressor")
    reset = next(node for node in compressor.body if isinstance(node, ast.FunctionDef) and node.name == "_reset")
    assert not calls_named(reset, "reset_peak_memory_stats")


def test_measured_round_has_exactly_one_peak_reset():
    tree = ast.parse(SOURCE.read_text())
    measure = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "measure")
    assert len(calls_named(measure, "reset_peak_memory_stats")) == 1


def test_checkpoint_index_entries_are_checked_before_opening_payloads():
    source = SOURCE.read_text()
    assert "def _checkpoint_entries" in source
    assert source.count("_checkpoint_entries(") >= 3
    assert "checkpoint file listed by index is missing" in source
