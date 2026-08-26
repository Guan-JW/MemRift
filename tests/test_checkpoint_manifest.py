import hashlib
import json

import pytest

from conftest import load_module


def model_entry(data, **updates):
    entry = {
        "logical_name": "tiny",
        "revision": "a" * 40,
        "expected_bytes": len(data),
        "expected_files": ["weights.bin"],
        "sha256": {"weights.bin": hashlib.sha256(data).hexdigest()},
    }
    entry.update(updates)
    return entry


def test_manifest_selects_exactly_one_entry(tmp_path):
    module = load_module("scripts/prepare_weights.py", "manifest_selection")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"models": [model_entry(b"x"), model_entry(b"x")]}))
    with pytest.raises(ValueError, match="exactly one"):
        module.select_model(manifest, "tiny")


def test_source_manifest_validates_files_size_and_sha256(tmp_path):
    module = load_module("scripts/prepare_weights.py", "manifest_validation")
    data = b"checkpoint source"
    (tmp_path / "weights.bin").write_bytes(data)
    module.validate_source(model_entry(data), tmp_path)
    with pytest.raises(ValueError, match="revision is unresolved"):
        module.validate_source(model_entry(data, revision=None), tmp_path)
    with pytest.raises(ValueError, match="model size"):
        module.validate_source(model_entry(data, expected_bytes=1), tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.validate_source(model_entry(data, sha256={"weights.bin": "0" * 64}), tmp_path)
    with pytest.raises(FileNotFoundError, match="missing expected files"):
        module.validate_source(model_entry(data, expected_files=["missing.bin"]), tmp_path)


def test_source_manifest_ignores_git_metadata(tmp_path):
    module = load_module("scripts/prepare_weights.py", "manifest_git_metadata")
    data = b"checkpoint source"
    (tmp_path / "weights.bin").write_bytes(data)
    git_dir = tmp_path / ".git" / "objects"
    git_dir.mkdir(parents=True)
    (git_dir / "clone-specific-data").write_bytes(b"not model content")

    module.validate_source(model_entry(data), tmp_path)


def test_clear_output_preserves_mount_root(tmp_path):
    module = load_module("scripts/prepare_weights.py", "manifest_clear_output")
    (tmp_path / "stale.bin").write_bytes(b"stale")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "stale.bin").write_bytes(b"stale")
    module.clear_output(tmp_path)
    assert tmp_path.is_dir()
    assert list(tmp_path.iterdir()) == []
