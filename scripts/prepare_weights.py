#!/usr/bin/env python3
"""Prepare a manifest-checked MemRift split/Zstd checkpoint offline."""

import argparse
import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_model(manifest_path: Path, logical_name: str) -> dict:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [item for item in document["models"] if item["logical_name"] == logical_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one manifest entry for {logical_name!r}")
    return matches[0]


def validate_source(model: dict, source: Path) -> None:
    if not model.get("revision"):
        raise ValueError("manifest revision is unresolved; record an exact Hugging Face commit")
    if not model.get("expected_bytes"):
        raise ValueError("manifest expected_bytes is unresolved; record the exact directory size")
    missing = [name for name in model["expected_files"] if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"model directory is missing expected files: {missing}")
    actual_size = sum(
        path.stat().st_size
        for path in source.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(source).parts
    )
    if actual_size != model["expected_bytes"]:
        raise ValueError(f"model size is {actual_size}, manifest expects {model['expected_bytes']}")
    for relative, expected in model.get("sha256", {}).items():
        actual = sha256(source / relative)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative}: {actual}")


def prepare(source: Path, target: Path, logical_name: str, source_revision: str, level: int) -> None:
    import torch
    import zstandard as zstd
    from transformers import AutoModelForCausalLM
    import float_split_stride

    model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    compressor = zstd.ZstdCompressor(level=level, write_checksum=True)
    index = []
    compression_seconds = 0.0
    for number, (name, parameter) in enumerate(model.named_parameters()):
        filename = f"{number:06d}.bin"
        output_path = target / filename
        if parameter.dtype in (torch.bfloat16, torch.float32):
            stream = torch.cuda.current_stream()
            exponent, sign_mantissa = float_split_stride.split(parameter, stream.cuda_stream)
            stream.synchronize()
            sign_mantissa = sign_mantissa.cpu().contiguous()
            raw_exponent = exponent.numpy().tobytes()
            started = time.perf_counter()
            compressed = compressor.compress(raw_exponent)
            compression_seconds += time.perf_counter() - started
            with output_path.open("wb") as output:
                output.write(struct.pack("<Q", parameter.numel()))
                output.write(sign_mantissa.numpy().tobytes())
                output.write(compressed)
            scheme = "split_zstd"
        else:
            torch.save(parameter.detach().cpu(), output_path)
            scheme = "raw_torch"
        index.append({
            "name": name, "file": filename, "shape": list(parameter.shape),
            "stride": list(parameter.stride()), "storage_offset": parameter.storage_offset(),
            "dtype": str(parameter.dtype).removeprefix("torch."), "scheme": scheme,
        })
    (target / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": "1.0",
        "format": "memrift-float-split-stride-v1",
        "model_logical_id": logical_name,
        "source_revision": source_revision,
        "zstd_level": level,
        "tensor_count": len(index),
        "compression_seconds": compression_seconds,
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def clear_output(target: Path) -> None:
    if target.resolve() in (Path("/"), Path.home().resolve()):
        raise ValueError("refusing unsafe overwrite target")
    for child in target.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/models.json"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--zstd-level", type=int, default=18, choices=range(1, 23), metavar="1..22")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected = select_model(args.manifest, args.name)
    validate_source(selected, args.model)
    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"refusing stale non-empty output directory: {args.output}")
        clear_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    incomplete = args.output / ".incomplete"
    incomplete.write_text("checkpoint preparation in progress\n", encoding="ascii")
    try:
        prepare(args.model, args.output, args.name, selected["revision"], args.zstd_level)
    except BaseException:
        incomplete.write_text("checkpoint preparation failed\n", encoding="ascii")
        raise
    incomplete.unlink()
    print(json.dumps({"checkpoint": args.output.name, "model_logical_id": args.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
