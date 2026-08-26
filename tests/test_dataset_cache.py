import json

from conftest import load_module


def test_dataset_cache_receipt_binds_id_revision(tmp_path):
    module = load_module("scripts/validate_environment.py", "dataset_cache_validation")
    revision = "a" * 40
    arrow = tmp_path / "data.arrow"
    arrow.write_bytes(b"arrow")
    receipt = {
        "datasets": [
            {
                "huggingface_id": "tatsu-lab/alpaca",
                "revision": revision,
                "fingerprint": "fingerprint",
                "num_rows": 1,
            }
        ]
    }
    (tmp_path / "memrift-dataset-receipt.json").write_text(json.dumps(receipt))

    assert module.validate_dataset_cache(tmp_path, "tatsu-lab/alpaca", revision) == []
    errors = module.validate_dataset_cache(tmp_path, "wrong/dataset", revision)
    assert any("does not bind" in error for error in errors)
