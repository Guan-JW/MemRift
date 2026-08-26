#!/usr/bin/env python3
"""Populate the review dataset cache and record resolved fingerprints."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("/workspace/manifests/datasets.json"))
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/cache/huggingface"))
    args = parser.parse_args()

    from datasets import load_dataset

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = {entry["logical_name"]: entry for entry in manifest["datasets"]}
    unknown = sorted(set(args.name) - entries.keys())
    if unknown:
        parser.error(f"unknown dataset names: {unknown}")

    receipt = {"schema_version": "1.0", "datasets": []}
    for name in args.name:
        entry = entries[name]
        dataset = load_dataset(
            entry["huggingface_id"],
            split=entry["split"],
            revision=entry["revision"],
            cache_dir=str(args.cache_dir),
        )
        receipt["datasets"].append(
            {
                "logical_name": name,
                "huggingface_id": entry["huggingface_id"],
                "revision": entry["revision"],
                "split": entry["split"],
                "fingerprint": dataset._fingerprint,
                "num_rows": len(dataset),
                "column_names": list(dataset.column_names),
            }
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    output = args.cache_dir / "memrift-dataset-receipt.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
