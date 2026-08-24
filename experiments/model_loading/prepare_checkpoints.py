#!/usr/bin/env python3
import argparse
import gc
import json
import shutil
import struct
import time
from pathlib import Path


METHODS = ("bf16", "nf4", "memrift")


def directory_size(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def selected_methods(method):
    return METHODS if method == "all" else (method,)


def prepare_output_directories(output_root, methods, overwrite=False):
    root = Path(output_root)
    paths = {method: root / method for method in methods}
    conflicts = [path for path in paths.values() if path.exists() and not path.is_dir()]
    if conflicts:
        names = ", ".join(str(path) for path in conflicts)
        raise NotADirectoryError(f"prepared checkpoint paths are not directories: {names}")
    stale = [path for path in paths.values() if path.exists() and any(path.iterdir())]
    if stale and not overwrite:
        names = ", ".join(str(path) for path in stale)
        raise FileExistsError(f"prepared checkpoint directories are not empty: {names}; use --overwrite")

    root.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        if overwrite and path.exists():
            shutil.rmtree(path)
        path.mkdir(exist_ok=True)
    return paths


def prepare_nf4(model_path, output_path, device="cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=config,
        device_map={"": device},
    )
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
    del model
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def prepare_bf16(model_path, output_path, device="cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
    del model
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def prepare_memrift(model_path, output_path, level, device="cuda:0"):
    import torch
    import zstandard as zstd
    import float_split_stride as split_kernel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    compressor = zstd.ZstdCompressor(level=level, write_checksum=False)
    index = []
    compression_seconds = 0.0

    for number, (name, parameter) in enumerate(model.named_parameters()):
        filename = f"{number:06d}.bin"
        output_file = Path(output_path) / filename
        if parameter.dtype in (torch.bfloat16, torch.float32):
            stream = torch.cuda.current_stream(device=device)
            cpu_exponent, sign_mantissa = split_kernel.split(parameter, stream.cuda_stream)
            stream.synchronize()
            sign_mantissa = sign_mantissa.cpu().contiguous()
            raw_exponent = cpu_exponent.numpy().tobytes()
            started = time.perf_counter()
            compressed_exponent = compressor.compress(raw_exponent)
            compression_seconds += time.perf_counter() - started
            with output_file.open("wb") as output:
                output.write(struct.pack("<Q", parameter.numel()))
                output.write(sign_mantissa.numpy().tobytes())
                output.write(compressed_exponent)
            scheme = "split_zstd"
        else:
            torch.save(parameter.detach().cpu(), output_file)
            scheme = "raw_torch"
        index.append(
            {
                "name": name,
                "file": filename,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype).removeprefix("torch."),
                "scheme": scheme,
            }
        )

    with (Path(output_path) / "index.json").open("w") as output:
        json.dump(index, output)
    with (Path(output_path) / "metadata.json").open("w") as output:
        json.dump(
            {
                "source": str(model_path),
                "zstd_level": level,
                "compression_seconds": compression_seconds,
            },
            output,
            indent=2,
        )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument("--zstd-level", type=int, default=21)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    methods = selected_methods(args.method)
    try:
        paths = prepare_output_directories(args.output_root, methods, args.overwrite)
    except (FileExistsError, NotADirectoryError) as error:
        parser.error(str(error))

    preparers = {
        "bf16": lambda path: prepare_bf16(args.model, path, args.device),
        "nf4": lambda path: prepare_nf4(args.model, path, args.device),
        "memrift": lambda path: prepare_memrift(args.model, path, args.zstd_level, args.device),
    }
    for method in methods:
        preparers[method](paths[method])
        print(json.dumps({"method": method, "bytes": directory_size(paths[method])}))


if __name__ == "__main__":
    main()
