#!/usr/bin/env python3
"""Validate the Jetson runtime without downloading or loading a model."""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
from pathlib import Path

CHECKPOINT_FORMAT = "memrift-float-split-stride-v1"
KNOWN_SCHEMES = {"split_zstd", "raw_torch"}
EXPECTED_PACKAGES = {
    "bitsandbytes": "0.45.4",
    "peft": "0.15.2",
    "transformers": "4.52.4",
}


def validate_checkpoint(root: Path) -> list[str]:
    errors = []
    if (root / ".incomplete").exists():
        errors.append(f"checkpoint is marked incomplete: {root}")
    documents = {}
    for name in ("index.json", "metadata.json"):
        path = root / name
        if not path.is_file():
            errors.append(f"checkpoint mount lacks {name}: {root}")
            continue
        try:
            documents[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid checkpoint {name}: {exc}")

    index = documents.get("index.json")
    metadata = documents.get("metadata.json")
    if index is not None and (not isinstance(index, list) or not index):
        errors.append("checkpoint index must be a non-empty array")
    names = set()
    files = set()
    if isinstance(index, list):
        for number, item in enumerate(index):
            prefix = f"checkpoint index item {number}"
            if not isinstance(item, dict):
                errors.append(f"{prefix} is not an object")
                continue
            required = {"name", "file", "shape", "dtype", "scheme"}
            missing = sorted(required - item.keys())
            if missing:
                errors.append(f"{prefix} lacks fields: {missing}")
                continue
            unexpected = sorted(item.keys() - (required | {"stride", "storage_offset"}))
            if unexpected:
                errors.append(f"{prefix} has unknown fields: {unexpected}")
            if not isinstance(item["name"], str) or not item["name"] or item["name"] in names:
                errors.append(f"{prefix} has an invalid or duplicate name")
            names.add(item["name"])
            relative = item["file"]
            if not isinstance(relative, str) or not relative or relative in files:
                errors.append(f"{prefix} has an invalid or duplicate file")
                continue
            files.add(relative)
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                errors.append(f"{prefix} file must be a safe relative path: {relative!r}")
            else:
                referenced = root / candidate
                try:
                    referenced.resolve().relative_to(root.resolve())
                except ValueError:
                    errors.append(f"{prefix} file escapes checkpoint root: {relative!r}")
                if not referenced.is_file():
                    errors.append(f"checkpoint referenced file is missing: {relative}")
            if item["scheme"] not in KNOWN_SCHEMES:
                errors.append(f"{prefix} has unknown scheme: {item['scheme']!r}")
            valid_shape = isinstance(item["shape"], list) and all(
                isinstance(size, int) and not isinstance(size, bool) and size >= 0 for size in item["shape"]
            )
            if not valid_shape:
                errors.append(f"{prefix} has an invalid shape")
            if not isinstance(item["dtype"], str) or not item["dtype"]:
                errors.append(f"{prefix} has an invalid dtype")
            if item["scheme"] == "split_zstd" and item["dtype"] not in {"bfloat16", "float32"}:
                errors.append(f"{prefix} split_zstd dtype is unsupported: {item['dtype']!r}")
            if "stride" in item and (
                not isinstance(item["stride"], list)
                or not valid_shape
                or len(item["stride"]) != len(item["shape"])
                or not all(isinstance(step, int) and not isinstance(step, bool) for step in item["stride"])
            ):
                errors.append(f"{prefix} has an invalid stride")
            if "storage_offset" in item and (
                not isinstance(item["storage_offset"], int)
                or isinstance(item["storage_offset"], bool)
                or item["storage_offset"] < 0
            ):
                errors.append(f"{prefix} has an invalid storage_offset")

    if metadata is not None:
        if not isinstance(metadata, dict):
            errors.append("checkpoint metadata must be an object")
        else:
            metadata_fields = {
                "schema_version", "format", "model_logical_id", "source_revision",
                "zstd_level", "tensor_count", "compression_seconds",
            }
            unexpected = sorted(metadata.keys() - metadata_fields)
            if unexpected:
                errors.append(f"checkpoint metadata has unknown fields: {unexpected}")
            expected = {
                "schema_version": "1.0",
                "format": CHECKPOINT_FORMAT,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    errors.append(f"checkpoint metadata {key} must be {value!r}")
            if not isinstance(metadata.get("model_logical_id"), str) or not metadata.get("model_logical_id"):
                errors.append("checkpoint metadata lacks model_logical_id")
            if not re.fullmatch(r"[0-9a-f]{40}", str(metadata.get("source_revision", ""))):
                errors.append("checkpoint metadata source_revision must be an exact 40-hex commit")
            if metadata.get("tensor_count") != (len(index) if isinstance(index, list) else None):
                errors.append("checkpoint metadata tensor_count does not match index")
            level = metadata.get("zstd_level")
            if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 22:
                errors.append("checkpoint metadata zstd_level must be between 1 and 22")
            seconds = metadata.get("compression_seconds")
            if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds < 0:
                errors.append("checkpoint metadata compression_seconds must be non-negative")
    return errors


def extension_roundtrip(torch, extension) -> None:
    for dtype, integer_dtype in ((torch.bfloat16, torch.int16), (torch.float32, torch.int32)):
        source = torch.randn((3, 11), device="cuda", dtype=dtype)[:, 1:10:2]
        stream = torch.cuda.current_stream()
        exponent, sign_mantissa = extension.split(source, stream.cuda_stream)
        restored = extension.merge(
            exponent, sign_mantissa, source.shape, source.stride(), source.storage_offset(),
            source.dtype, stream.cuda_stream,
        )
        stream.synchronize()
        if restored.stride() != source.stride() or restored.storage_offset() != source.storage_offset():
            raise RuntimeError(f"{dtype} shape metadata changed during odd non-contiguous roundtrip")
        if not torch.equal(restored.contiguous().view(integer_dtype), source.contiguous().view(integer_dtype)):
            raise RuntimeError(f"odd non-contiguous {dtype} roundtrip was not bit exact")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-model", type=Path)
    parser.add_argument("--require-checkpoint", type=Path)
    parser.add_argument("--require-dataset-cache", type=Path)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--results", type=Path, default=Path("/results"))
    parser.add_argument("--allow-non-jetson", action="store_true")
    args = parser.parse_args()

    errors = []
    warnings = []
    cuda_available = False
    architecture = platform.machine()
    if architecture != "aarch64" and not args.allow_non_jetson:
        errors.append(f"architecture is {architecture}, expected aarch64")

    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        capability = list(torch.cuda.get_device_capability(0)) if cuda_available else None
        device = torch.cuda.get_device_name(0) if cuda_available else None
        if not cuda_available:
            errors.append("PyTorch CUDA support is unavailable")
        elif capability != [8, 7]:
            errors.append(f"GPU capability is {capability}, expected [8, 7]")
    except Exception as exc:  # validation must report rather than traceback
        torch_version = cuda_version = device = None
        capability = None
        errors.append(f"cannot initialize PyTorch/CUDA: {exc}")

    try:
        extension = importlib.import_module("float_split_stride")
        extension_path = str(Path(extension.__file__).name)
        if cuda_available and not args.allow_non_jetson:
            extension_roundtrip(torch, extension)
    except Exception as exc:
        extension_path = None
        errors.append(f"cannot import float_split_stride: {exc}")

    if args.require_model:
        config = args.require_model / "config.json"
        if not config.is_file():
            errors.append(f"model mount lacks config.json: {args.require_model}")
        else:
            try:
                if not isinstance(json.loads(config.read_text(encoding="utf-8")), dict):
                    errors.append(f"model config.json is not an object: {args.require_model}")
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"model config.json is invalid: {exc}")
    if args.require_checkpoint:
        errors.extend(validate_checkpoint(args.require_checkpoint))
    if args.require_dataset_cache:
        if not re.fullmatch(r"[0-9a-f]{40}", args.dataset_revision or ""):
            errors.append("dataset revision must be an exact 40-hex commit")
        if not args.require_dataset_cache.is_dir() or not any(args.require_dataset_cache.rglob("*.arrow")):
            errors.append(f"dataset cache has no pre-populated Arrow data: {args.require_dataset_cache}")

    try:
        args.results.mkdir(parents=True, exist_ok=True)
        probe = args.results / ".memrift-write-test"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
    except OSError as exc:
        errors.append(f"results directory is not writable: {exc}")

    l4t_path = Path("/etc/nv_tegra_release")
    l4t_release = None
    if not l4t_path.exists():
        message = "/etc/nv_tegra_release is absent; cannot verify L4T R36.4.0"
        (warnings if args.allow_non_jetson else errors).append(message)
    else:
        release_text = l4t_path.read_text(encoding="ascii", errors="replace")
        match = re.search(r"# R(\d+) \(release\), REVISION: ([0-9.]+)", release_text)
        l4t_release = f"R{match.group(1)}.{match.group(2)}" if match else None
        if l4t_release != "R36.4.0":
            errors.append(f"L4T release is {l4t_release or 'unparseable'}, expected R36.4.0")
    package_versions = {}
    for package, expected in EXPECTED_PACKAGES.items():
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
        if not args.allow_non_jetson and package_versions[package] != expected:
            errors.append(f"{package} is {package_versions[package]!r}, expected {expected}")
    if shutil.which("tegrastats") is None:
        warnings.append("tegrastats unavailable; system telemetry will be omitted")

    report = {
        "ok": not errors,
        "architecture": architecture,
        "python": sys.version.split()[0],
        "torch": torch_version,
        "cuda": cuda_version,
        "cuda_available": "cuda_available" in locals() and cuda_available,
        "device": device,
        "compute_capability": capability,
        "l4t_release": l4t_release,
        "package_versions": package_versions,
        "extension": extension_path,
        "wandb_mode": os.environ.get("WANDB_MODE", "disabled"),
        "offline": os.environ.get("HF_HUB_OFFLINE", "1") == "1",
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
