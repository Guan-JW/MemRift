# Model-loading results

The values below are imported reported results from the original experiment
records. They were not independently rerun during artifact assembly; the tables
should be interpreted together with the raw JSON artifacts named below.

## Protocol

- Hardware: NVIDIA Jetson Orin, 29 GiB unified memory, 8 online Cortex-A78AE cores.
- Storage: all source and prepared checkpoints were on the same 1.8 TB USB `My Passport 264F` device.
- Software: PyTorch 2.6.0, Transformers 4.49.0, bitsandbytes 0.45.4.dev0, PEFT 0.14.0, CUDA 12.8.
- LoRA: rank 16, alpha 16, dropout 0, MLP projections (`gate_proj`, `up_proj`, and `down_proj`).
- QLoRA: NF4, BF16 compute, double quantization.
- MemRift: the repository's split sign/mantissa plus Zstd exponent representation, Zstd level 3.
- Each measurement used a fresh process. Timing began immediately before the first config/checkpoint read and ended after model construction, device placement, LoRA installation, and CUDA synchronization.
- Five runs per path were executed and results are medians. The main matrix used randomized method order; after the final audit, MemRift alone was rerun to include materialization-hook installation in its ready-state boundary.
- The container could not clear `/proc/sys/vm/drop_caches`, so these are explicitly **warm-cache results**. The power-mode utility was unavailable inside the container.
- Mistral's original `.bin` checkpoint was normalized offline to BF16 safetensors with a 5 GB maximum shard size. Prepared NF4 checkpoints also used safetensors and the same maximum shard size. MemRift necessarily uses its native per-tensor format.

## Loading

Times are seconds. Checkpoint sizes are decimal GB. LoRA and online QLoRA read the same BF16 checkpoint.

| Model | LoRA | QLoRA online | QLoRA prequant | MemRift | BF16 size | NF4 size | MemRift size |
|---|---:|---:|---:|---:|---:|---:|---:|
| TinyLlama-1.1B | 2.15 | 4.26 | **2.07** | 2.66 | 2.20 | 0.76 | 1.55 |
| Llama-3.2-3B-Instruct | 4.41 | 8.94 | **3.07** | 8.47 | 6.43 | 2.24 | 4.53 |
| Mistral-7B | 20.57 | 14.36 | **12.81** | 15.99 | 14.48 | 4.13 | 10.21 |
| Llama-3.1-8B | 35.61 | 41.47 | **18.28** | 24.65 | 16.06 | 5.70 | 11.32 |

Compared with online QLoRA, direct NF4 loading reduced median load-to-ready time by 51.5%, 65.7%, 10.8%, and 55.9%, respectively. Warm-cache variance was substantial for the larger checkpoints; the individual run JSON files should be retained with any reported medians.

## Loading RAM

Peak process RSS is shown in GiB. On Jetson, CUDA uses unified memory, so `peak_torch_allocated_bytes` and host-wide RAM samples are also retained in every raw result JSON; process RSS alone does not include all CUDA allocations.

| Model | LoRA | QLoRA online | QLoRA prequant | MemRift |
|---|---:|---:|---:|---:|
| TinyLlama-1.1B | 2.54 | 2.61 | **1.20** | 1.07 |
| Llama-3.2-3B-Instruct | 5.12 | 5.18 | **2.58** | 2.03 |
| Mistral-7B | 5.14 | 5.23 | **4.34** | 3.63 |
| Llama-3.1-8B | 5.14 | 5.22 | **4.82** | 4.69 |

The online path had higher process RSS than the prequantized path for every model. The difference is largest for TinyLlama and Llama-3.2, where full-precision source tensors visibly coexist with quantized tensors during conversion.

For completeness, median host-wide peak RAM used is below in GiB. These absolute values include the process, unified CUDA allocations, filesystem cache, and unrelated system use; baseline and delta samples are available in the raw JSON.

| Model | LoRA | QLoRA online | QLoRA prequant | MemRift |
|---|---:|---:|---:|---:|
| TinyLlama-1.1B | 6.23 | 5.98 | 5.92 | 5.49 |
| Llama-3.2-3B-Instruct | 8.39 | 6.30 | 5.93 | 7.36 |
| Mistral-7B | 15.94 | 7.47 | 8.52 | 12.76 |
| Llama-3.1-8B | 17.33 | 11.60 | 11.45 | 14.48 |

## Validation

The Transformers NF4 loader was instrumented at `Bnb4BitHfQuantizer.create_quantized_param`. Every online run used only the online branch; every prepared checkpoint run used only `Params4bit.from_prequantized` (154, 196, 224, and 224 quantized tensors by model). Thus the prepared checkpoints did not silently load BF16 and requantize.

For validation, each path used the same deterministic 64-token input. Two rows of first-forward logits were compared, followed by one rank-16 LoRA training step.

| Model | Max logit difference | Loss difference | Step time prequant / online | Peak training memory prequant / online |
|---|---:|---:|---:|---:|
| TinyLlama-1.1B | 0 | 0 | 0.873 | 1.000 |
| Llama-3.2-3B-Instruct | 0 | 0 | 0.994 | 0.999 |
| Mistral-7B | 0 | 0 | 0.999 | 0.997 |
| Llama-3.1-8B | 0 | 0 | 0.988 | 1.000 |

The imported results reported that serialization preserved the online NF4 model and did not change steady-state QLoRA training behavior under this protocol.

## Artifacts

- Raw loading runs and summaries: `/results/model_loading/`
- Forward/training validation: `/results/model_loading-validation/`
- Prepared checkpoints: `/checkpoints/`
- Benchmark and preparation commands: `/workspace/experiments/model_loading/README.md`

These are logical artifact mount paths, not claims about the original benchmark host's filesystem layout.
