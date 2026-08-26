#!/usr/bin/env python3
import argparse
import importlib.metadata
import json
import math
import sys
import struct
import threading
import time
from pathlib import Path

try:
    from experiments.model_loading.driver_utils import enter_process_group, environment_record, utc_now
except ModuleNotFoundError:
    from driver_utils import enter_process_group, environment_record, utc_now


REQUIRED_VERSIONS = {"transformers": "4.49.0", "peft": "0.14.0"}


def require_exact_versions(version_getter=None):
    version_getter = version_getter or importlib.metadata.version
    mismatches = []
    for package, required in REQUIRED_VERSIONS.items():
        try:
            installed = version_getter(package)
        except importlib.metadata.PackageNotFoundError:
            installed = "not installed"
        if installed != required:
            mismatches.append(f"{package}=={required} required, found {installed}")
    if mismatches:
        raise RuntimeError("; ".join(mismatches))


def validate_checkpoint(method, checkpoint):
    required = method in ("qlora-prequant", "memrift")
    if required and not checkpoint:
        raise ValueError(f"--checkpoint is required for method {method}")
    if not required and checkpoint:
        raise ValueError(f"--checkpoint is not used for method {method}")


def contiguous_strides(shape):
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return tuple(strides)


def checkpoint_size(path):
    path = Path(path)
    if (path / "index.json").exists():
        return sum(file.stat().st_size for file in path.iterdir() if file.is_file())
    checkpoint_files = [path / "config.json"]
    for pattern in ("model*.safetensors", "pytorch_model*.bin"):
        checkpoint_files.extend(path.glob(pattern))
    return sum(file.stat().st_size for file in set(checkpoint_files) if file.exists())


class PeakSampler:
    def __init__(self):
        import psutil

        self.psutil = psutil
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.peak_rss = self.process.memory_info().rss
        self.peak_system_used = psutil.virtual_memory().used

    def run(self):
        while not self.stop_event.wait(0.01):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self.peak_system_used = max(
                self.peak_system_used, self.psutil.virtual_memory().used
            )


def lora_config():
    from peft import LoraConfig

    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.0,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )


def materialize(parameter, split_kernel, device):
    import torch
    import zstandard as zstd

    count = math.prod(parameter.original_shape)
    exponent = torch.empty(count, dtype=torch.uint8, pin_memory=True)
    with zstd.ZstdDecompressor().stream_reader(parameter.exponent) as reader:
        if reader.readinto(memoryview(exponent.numpy())) != count:
            raise ValueError("compressed exponent data has an unexpected size")
    stream = torch.cuda.current_stream(device=device)
    value = split_kernel.merge(
        exponent,
        parameter.sign_mantissa,
        parameter.original_shape,
        contiguous_strides(parameter.original_shape),
        0,
        parameter.dtype,
        stream.cuda_stream,
    )
    stream.synchronize()
    return value


def load_memrift(model_path, checkpoint_path, device="cuda:0"):
    import numpy as np
    import torch
    import float_split_stride as split_kernel
    from accelerate import init_empty_weights
    from peft import get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    class CompressedParameter(torch.nn.Parameter):
        def __new__(cls, shape, dtype, sign_mantissa, exponent):
            data = torch.empty(0, dtype=dtype, device=sign_mantissa.device)
            return super().__new__(cls, data, requires_grad=False)

        def __init__(self, shape, dtype, sign_mantissa, exponent):
            self.original_shape = tuple(shape)
            self.sign_mantissa = sign_mantissa
            self.exponent = exponent

    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    modules = dict(model.named_modules())
    with (Path(checkpoint_path) / "index.json").open() as source:
        index = json.load(source)

    compressed_count = 0
    layer_parameters = {}
    for item in index:
        module_name, _, attribute = item["name"].rpartition(".")
        module = modules[module_name]
        file_path = Path(checkpoint_path) / item["file"]
        if item["scheme"] == "raw_torch":
            value = torch.load(file_path, map_location=device, weights_only=True)
            module._parameters[attribute] = torch.nn.Parameter(value, requires_grad=False)
            continue
        with file_path.open("rb") as source:
            count = struct.unpack("<Q", source.read(8))[0]
            bytes_per_element = 1 if item["dtype"] == "bfloat16" else 3
            sign_mantissa = np.frombuffer(
                source.read(count * bytes_per_element), dtype=np.uint8
            ).copy()
            exponent = source.read()
        sign_mantissa = torch.from_numpy(sign_mantissa).to(device)
        dtype = torch.bfloat16 if item["dtype"] == "bfloat16" else torch.float32
        parameter = CompressedParameter(item["shape"], dtype, sign_mantissa, exponent)
        if ".layers." not in item["name"]:
            parameter.data = materialize(parameter, split_kernel, device)
            del parameter.sign_mantissa, parameter.exponent
        module._parameters[attribute] = parameter
        parts = module_name.split(".")
        if "layers" in parts:
            layer_position = parts.index("layers")
            layer_name = ".".join(parts[: layer_position + 2])
            layer_parameters.setdefault(modules[layer_name], []).append(parameter)
        compressed_count += 1

    if model.config.tie_word_embeddings:
        model.tie_weights()
    for module in model.modules():
        for name, buffer in module._buffers.items():
            if buffer is not None:
                module._buffers[name] = buffer.to(device)
    model = get_peft_model(model, lora_config(), autocast_adapter_dtype=True)

    # Hook installation is included in ready state; values remain lazy by design.
    for layer, parameters in layer_parameters.items():
        def materialize_layer(_, __, parameters=parameters):
            for parameter in parameters:
                if parameter.numel() == 0:
                    parameter.data = materialize(parameter, split_kernel, device)

        layer.register_forward_pre_hook(materialize_layer)
    return model, {
        "compressed_parameter_count": compressed_count,
        "materialization_hook_count": len(layer_parameters),
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("lora", "qlora-online", "qlora-prequant", "memrift"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-state", choices=("warm",), default="warm")
    return parser


def main(argv=None):
    enter_process_group()
    started_at = utc_now()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_checkpoint(args.method, args.checkpoint)
        require_exact_versions()
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))

    import psutil
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.quantizers.quantizer_bnb_4bit import Bnb4BitHfQuantizer

    online_calls = 0
    prequant_calls = 0
    original_create = Bnb4BitHfQuantizer.create_quantized_param

    def counted_create(self, *create_args, **create_kwargs):
        nonlocal online_calls, prequant_calls
        if self.pre_quantized:
            prequant_calls += 1
        else:
            online_calls += 1
        return original_create(self, *create_args, **create_kwargs)

    Bnb4BitHfQuantizer.create_quantized_param = counted_create
    sampler = PeakSampler()
    sampler_thread = threading.Thread(target=sampler.run, daemon=True)
    sampler_thread.start()
    baseline_rss = sampler.process.memory_info().rss
    baseline_system_used = psutil.virtual_memory().used
    started = time.perf_counter()
    details = {}
    try:
        if args.method == "lora":
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map={"": args.device}
            )
            model = get_peft_model(model, lora_config(), autocast_adapter_dtype=True)
            measured_checkpoint = args.model
        elif args.method == "qlora-online":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
                device_map={"": args.device},
            )
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
            model = get_peft_model(model, lora_config(), autocast_adapter_dtype=True)
            measured_checkpoint = args.model
        elif args.method == "qlora-prequant":
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint, torch_dtype=torch.bfloat16, device_map={"": args.device}
            )
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
            model = get_peft_model(model, lora_config(), autocast_adapter_dtype=True)
            measured_checkpoint = args.checkpoint
        else:
            model, details = load_memrift(args.model, args.checkpoint, args.device)
            measured_checkpoint = args.checkpoint

        torch.cuda.synchronize(device=args.device)
        elapsed = time.perf_counter() - started
    finally:
        Bnb4BitHfQuantizer.create_quantized_param = original_create
        sampler.stop_event.set()
        sampler_thread.join()

    if args.method == "memrift":
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, 2]], dtype=torch.long, device=args.device))
        torch.cuda.synchronize(device=args.device)
        details["post_timing_forward_validated"] = True

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    result = {
        "method": args.method,
        "model": str(args.model),
        "checkpoint": str(measured_checkpoint),
        "device": args.device,
        "cache_state": args.cache_state,
        "cache_dropped": False,
        "started_at": started_at,
        "finished_at": utc_now(),
        "command": [sys.executable, str(Path(__file__)), *(argv if argv is not None else sys.argv[1:])],
        "environment": environment_record(),
        "load_to_ready_seconds": elapsed,
        "checkpoint_bytes": checkpoint_size(measured_checkpoint),
        "baseline_process_rss_bytes": baseline_rss,
        "peak_process_rss_bytes": sampler.peak_rss,
        "peak_process_rss_delta_bytes": sampler.peak_rss - baseline_rss,
        "baseline_system_used_bytes": baseline_system_used,
        "peak_system_used_bytes": sampler.peak_system_used,
        "peak_system_used_delta_bytes": sampler.peak_system_used - baseline_system_used,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(device=args.device),
        "online_quantized_tensor_calls": online_calls,
        "prequantized_tensor_calls": prequant_calls,
        "trainable_parameters": trainable,
        **details,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
