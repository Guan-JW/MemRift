import json

import pytest

from conftest import load_module


def test_checkpoint_format_is_selected_from_metadata_not_model_name(tmp_path):
    module = load_module("src/checkpoint_format.py", "checkpoint_format")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"format": "memrift-float-split-stride-v1"})
    )
    (tmp_path / "index.json").write_text("[]")

    assert module.uses_legacy_checkpoint(tmp_path) is False


def test_checkpoint_without_metadata_uses_index_scheme(tmp_path):
    module = load_module("src/checkpoint_format.py", "checkpoint_index_format")
    (tmp_path / "index.json").write_text(json.dumps([{"scheme": "split_zstd"}]))
    assert module.uses_legacy_checkpoint(tmp_path) is False

    (tmp_path / "index.json").write_text(json.dumps([{"name": "legacy"}]))
    assert module.uses_legacy_checkpoint(tmp_path) is True


def test_empty_checkpoint_index_is_rejected(tmp_path):
    module = load_module("src/checkpoint_format.py", "checkpoint_empty_format")
    (tmp_path / "index.json").write_text("[]")
    with pytest.raises(ValueError, match="non-empty"):
        module.uses_legacy_checkpoint(tmp_path)
