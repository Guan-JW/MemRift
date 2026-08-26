import json
from pathlib import Path


def uses_legacy_checkpoint(checkpoint: str | Path) -> bool:
    checkpoint = Path(checkpoint)
    metadata_path = checkpoint / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata.get("format") != "memrift-float-split-stride-v1"

    index = json.loads((checkpoint / "index.json").read_text(encoding="utf-8"))
    if not isinstance(index, list) or not index:
        raise ValueError("checkpoint index must be a non-empty list")
    return all("scheme" not in entry for entry in index)
