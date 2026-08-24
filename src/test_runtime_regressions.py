import ast
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("train_memrift.py")


def calls_named(node, attribute):
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == attribute
    ]


def function_named(node, name):
    return next(child for child in ast.walk(node) if isinstance(child, ast.FunctionDef) and child.name == name)


class RuntimeSourceTests(unittest.TestCase):
    def test_runtime_cleanup_does_not_reset_cuda_peaks(self):
        tree = ast.parse(SOURCE.read_text())
        compressor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AsyncCompressor")
        reset = next(node for node in compressor.body if isinstance(node, ast.FunctionDef) and node.name == "_reset")
        self.assertFalse(calls_named(reset, "reset_peak_memory_stats"))

    def test_peak_reset_occurs_once_in_measured_round(self):
        tree = ast.parse(SOURCE.read_text())
        measure = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "measure")
        self.assertEqual(len(calls_named(measure, "reset_peak_memory_stats")), 1)

    def test_async_workers_signal_readiness_and_reraise(self):
        tree = ast.parse(SOURCE.read_text())
        compressor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AsyncCompressor")
        for parent_name, worker_name, ready_name in (
            ("decompress_async", "_decode", "set"),
            ("materialize_async", "_materialize", "set"),
        ):
            worker = function_named(function_named(compressor, parent_name), worker_name)
            outer_try = next(node for node in worker.body if isinstance(node, ast.Try))
            self.assertTrue(any(isinstance(node, ast.Raise) for node in ast.walk(outer_try)))
            self.assertTrue(calls_named(outer_try.finalbody[-1], ready_name))

    def test_synthetic_data_does_not_call_dataset_loader(self):
        tree = ast.parse(SOURCE.read_text())
        synthetic_if = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and node.test.attr == "synthetic_data"
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "data_prepare"
                for statement in synthetic_if.body
                for node in ast.walk(statement)
            )
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "data_prepare"
                for statement in synthetic_if.orelse
                for node in ast.walk(statement)
            )
        )


if __name__ == "__main__":
    unittest.main()
