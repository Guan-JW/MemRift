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


def test_tegrastats_is_stopped_before_summary_reads():
    tree = ast.parse(SOURCE.read_text())
    measure = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "measure")
    stop = next(
        node for node in ast.walk(measure)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "stop_tegrastats"
    )
    stats_read = next(
        node for node in ast.walk(measure)
        if isinstance(node, ast.With)
        and any(isinstance(item.context_expr, ast.Name) and item.context_expr.id == "tegra_stats_lock"
                for item in node.items)
        and node.lineno > stop.lineno
    )
    assert stop.lineno < stats_read.lineno


def test_tegrastats_shutdown_stops_reaps_and_joins():
    tree = ast.parse(SOURCE.read_text())
    shutdown = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "stop_tegrastats")
    calls = {call.func.attr for call in ast.walk(shutdown) if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)}
    assert {"set", "terminate", "join"}.issubset(calls)
