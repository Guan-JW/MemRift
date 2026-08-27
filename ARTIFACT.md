# MemRift Artifact Evaluation Appendix

## Artifact Abstract

This artifact packages MemRift, a lossless weight-and-activation compression
runtime for LLM fine-tuning on unified-memory systems. It provides the MemRift
runtime, CUDA split/merge extension, checkpoint preparation, LoRA/QLoRA and
gradient-checkpointing baselines, model-loading benchmarks, safety watchdogs,
and structured result collection. The target is a 32 GB NVIDIA Jetson AGX Orin
running JetPack 6.1/L4T R36.4.0. Reviewers can validate the implementation and
run a deterministic TinyLlama smoke experiment first. Full paper reproduction
requires separately licensed model snapshots and pre-populated dataset caches.

The key paper claims are reduced peak system memory, longer trainable contexts,
35-40% smaller lossless checkpoints, competitive step and loading times, and
bit-exact reconstruction. `manifests/paper_claims.json` records every numeric
target and its current automation status.

## Artifact Checklist

- **Algorithm:** lossless exponent byte-stream coding integrated with separate
  static-weight and dynamic-activation runtime pipelines.
- **Program:** PyTorch/Transformers/PEFT fine-tuning, model-loading benchmarks,
  CUDA split/merge kernels, and Python experiment drivers.
- **Compilation:** the CUDA extension and bitsandbytes are built inside the
  pinned NVIDIA container for SM 8.7. No host compiler installation is used.
- **Transformations:** no compiler transformation pass is required.
- **Binary:** the release is an aarch64 Docker image; it is not portable to x86.
- **Models:** TinyLlama-1.1B, Llama-3.2-3B-Instruct, Mistral-7B, and
  Llama-3.1-8B. Weights are not redistributed. Only TinyLlama currently has a
  completed content manifest.
- **Datasets:** Alpaca is primary; LongForm and OASST1 are additional corpora.
  They are not redistributed. Review-time revisions and licenses are recorded
  in `manifests/datasets.json`.
- **Runtime:** Linux/aarch64, Docker Engine, NVIDIA Container Runtime, JetPack
  6.1/L4T R36.4.0, and an offline evaluation container.
- **Hardware:** 32 GB Jetson AGX Orin, Ampere GPU compute capability 8.7. The
  paper's Figure 10 uses 12 CPU cores. The locally validated smoke host was in
  `MODE_30W` with eight online cores, so that smoke is functional evidence, not
  a paper-performance reproduction.
- **Runtime state:** training results use 500-ms `tegrastats` sampling. Loading
  results are five-run warm-cache medians in fresh processes; cache dropping is
  not claimed.
- **Execution:** use an otherwise idle and thermally stable Jetson. Do not run
  unsafe non-GC 4096+ token configurations without the documented opt-in. A
  4-GiB `MemAvailable` watchdog is enabled by default.
- **Metrics:** peak total system RAM, CUDA allocator peaks, process RSS, step
  time, loading time, checkpoint bytes, maximum context, compression ratio,
  accuracy, and exact tensor mismatches.
- **Output:** each standard run writes the exact command, environment, raw log,
  resolved configuration, telemetry, and `run.json`. Aggregation produces JSON
  and CSV while preserving scalar measurements.
- **Workflow:** Docker, Make, shell wrappers, and Python experiment drivers.
- **Public availability:** the reviewer image is publicly pullable from GHCR by
  the immutable digest below.
- **License:** Apache-2.0 for the artifact; datasets and models retain their own
  licenses.
- **Archive:** the published Zenodo record is available at
  [doi:10.5281/zenodo.22119678](https://doi.org/10.5281/zenodo.22119678).

## Access

Pull the public ARM64 reviewer image by immutable digest:

```bash
export MEMRIFT_IMAGE=ghcr.io/guan-jw/memrift-artifact@sha256:45a7d409586dea875c504da09fd3e2215b2491476c151297b1c5d710d02b9979
docker pull "$MEMRIFT_IMAGE"
```

The registry image and the DOI-backed Zenodo archive are public. The archive
records the final source revision, evidence, provenance receipts, and release
checksums without redistributing the NVIDIA-derived container image.

## Hardware Dependencies

Required for reportable paper comparisons:

- NVIDIA Jetson AGX Orin with 32 GB unified LPDDR5.
- Ampere GPU with compute capability 8.7.
- JetPack 6.1 generation and L4T R36.4.0.
- Adequate active cooling and stable power.
- Twelve online CPU cores for Figure 10 and its worker-sensitivity comparison.
- Storage large enough for the container, model snapshots, prepared
  checkpoints, dataset caches, and result logs.

Record `nvpmodel -q`, `nproc`, thermal state, clocks, kernel, storage device, and
free memory with each paper run. Do not compare timings collected under
different power modes as if they used the same setup.

## Software Dependencies

The image is based on the digest-pinned NVIDIA PyTorch 24.12 Jetson image. It
contains PyTorch 2.6.0a0+nv24.12, CUDA 12.6, separate training/loading Python
environments, source-built bitsandbytes 0.45.4, and the compiled MemRift CUDA
extension. See `docker/Dockerfile.jetson` and the requirement lock files.

The paper states that all configurations use FlashAttention. The imported
runtime does not explicitly select FlashAttention and the current image does not
contain the `flash_attn` Python package. Performance comparisons must remain
non-reportable until the authors identify whether the original runs used
Transformers `flash_attention_2`, PyTorch SDPA flash kernels, or another path.

## Datasets

The AE release pins these immutable review snapshots:

| Dataset | Revision | License | Upstream storage |
|---|---|---|---:|
| `tatsu-lab/alpaca` | `dce01c9b08f87459cf36a430d809084718273017` | CC-BY-NC-4.0 | 318,636,537 B |
| `akoksal/LongForm` | `c4d2836f581bf1b7ad4ba1243af81b6b1b8626f1` | MIT | 1,018,249,527 B |
| `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | Apache-2.0 | 398,331,795 B |

The paper did not record historical dataset commits. These are reproducible AE
inputs selected later, not recovered identities for the original measurements.
Populate the Hugging Face cache outside the network-isolated evaluation run.

```bash
make cache-dataset CACHE_DIR=/path/to/huggingface-cache
```

This network-enabled preparation step loads the exact revision and writes a
receipt containing its resolved fingerprint and row count. Offline evaluation
rejects a cache whose receipt does not match `DATASET_ID` and
`DATASET_REVISION`.

## Models

`manifests/models.json` pins TinyLlama commit
`de253fa9783f8bd558c9ed398c8ffbe3c55cedb3` and hashes its model,
configuration, and tokenizer files. Final manifests for Llama-3.2-3B-Instruct,
Mistral-7B, and Llama-3.1-8B remain required. Meta model access requires license
acceptance; do not redistribute those weights in the artifact image.

## Installation

```bash
make image
make validate
```

The locally measured image size is approximately 10.63 GB. The source tree is
approximately 1.25 MB. Reserve at least 40 GB for image layers and build cache,
plus model and checkpoint storage. The verified TinyLlama snapshot is 2.20 GB
and its located MemRift checkpoint is 1.55 GB. The complete four-model matrix
requires substantially more space and must be recalculated after the final
model manifests are populated.

## Experiment Workflow

### Functional Smoke

Prepare a schema-compliant checkpoint:

```bash
make prepare \
  MODEL_DIR=/path/to/TinyLlama-1.1B-Chat-v1.0 \
  CHECKPOINT_OUTPUT_DIR=/path/to/tinyllama-memrift
```

Run the deterministic synthetic smoke:

```bash
make smoke \
  MODEL_DIR=/path/to/TinyLlama-1.1B-Chat-v1.0 \
  CHECKPOINT_DIR=/path/to/tinyllama-memrift
```

The local two-round smoke completed in under one minute after image setup. It
is a functional check and does not reproduce a paper table.

### Comparative Memory Evaluation

```bash
make memory-comparison \
  TAG="$MEMRIFT_IMAGE" \
  MEMRIFT_IMAGE_DIGEST="${MEMRIFT_IMAGE##*@}" \
  MODEL_DIR=/path/to/TinyLlama-1.1B-Chat-v1.0 \
  CHECKPOINT_DIR=/path/to/tinyllama-memrift \
  CACHE_DIR=/path/to/prepopulated/huggingface-cache \
  RESULTS_DIR="$PWD/results"
make check EXPERIMENT=memory \
  OUTPUT="$PWD/results/memory-comparison-tinyllama-1.1b-chat-v1.0/summary.json"
```

The reviewer profile runs three balanced repetitions at context 2048 and batch
3. It compares non-GC LoRA, online QLoRA, and MemRift with identical pinned
inputs, seven rounds, one warmup, calibrated 16/4/4 concurrency, and the 4 GiB
safety guard. It reports whole-system RAM, CUDA allocator memory, process RSS,
CPU/GPU utilization, timing, and links to raw `tegrastats.csv` files.

The automatic claim check passes only when every matched run completes and
MemRift's median whole-system peak is lower than both baselines. Its scope is
that exact reviewer configuration; it does not establish universal superiority
or reproduce the exact batch-4 paper point.

The verified run completed all nine workers. Median peak whole-system RAM was
23,140 MiB for LoRA, 24,698 MiB for online QLoRA, and 22,124 MiB for MemRift,
giving MemRift reductions of 4.39% and 10.42%. Median measured-round time was
12.77 s, 13.85 s, and 15.45 s respectively; the artifact therefore claims a
memory reduction for this profile, not a training-speed improvement.

`make evaluate` remains available for a single MemRift+GC point. `make gc`
compares LoRA+GC, QLoRA+GC, and corrected MemRift+GC as a separate experiment.

### Paper Results

| Paper item | Current reviewer workflow |
|---|---|
| Table 1 | Entropy collector is available under `experiments/entropy`; three final model manifests remain. |
| Tables 2-3 | `make tables23` produces a provenance-checked TinyLlama AE re-execution; the other models and activation attribution remain unavailable. |
| Figures 1 and 6 | Core methods run; four-model sweep and activation attribution remain. |
| Table 4 | Use `make gc GC_CONTEXT=2944`, repeat at 8192, then use `make gc-max-context` for the 128-token search. |
| Table 5 | `make correctness-quick` runs 10 steps; `make correctness-full` retains the optional 100-step stress protocol. Mistral checkpoints and LM-Eval inputs remain required. |
| Table 6 | Use `make backends`; it serializes LoRA plus LZ4, Zstd, EBC-LZ4, and EBC-Zstd runs and writes `table6_backends.csv`. |
| Figures 3 and 7 | Controlled co-runner supervisor still required. |
| Figure 8 | `experiments/ablation/run.py` serializes LoRA, weight-only, and weight-plus-activation runs and computes reductions. |
| Figures 9-10 | NVTX ranges exist; Nsight launcher/postprocessor and component timers remain. |
| Figure 11 | `experiments/lookahead/run.py` sweeps `Nw,Np in {0,1,2,4,8}` and records each run incrementally. |

No incomplete row should be presented as artifact-reproduced. The machine-
readable claim matrix is authoritative for status and expected output names.

Table 6 holds prepared weights at EBC-Zstd and varies activation compression,
matching the located historical runtime paths. Its ratio denominator includes
the complete retained payload. Located historical runs used an unpinned
OpenAssistant Guanaco dataset, so the default pinned Alpaca workflow is a new AE
configuration and records that discrepancy in `table6_manifest.json`.

## Evaluation and Expected Results

On the exact recorded platform, use these default acceptance limits unless the
final author-approved release provides tighter measured distributions:

- Peak system RAM: within 5% of the paper value.
- Step time: within 10% after warmup.
- Warm-cache model loading median: within 15% over five fresh processes.
- Percentage metrics: within three percentage points.
- Maximum context: meet or exceed the reported context at the same batch size.
- Numerical fidelity: exactly zero reconstructed-tensor mismatches.
- Accuracy: MemRift must remain within the paper's reported confidence interval
  and must not differ from the deterministic LoRA reference beyond that range.

These tolerances account for sampling, thermal, filesystem-cache, and scheduling
variation. A software/dependency failure is never an acceptable OOM substitute.

## Experiment Customization

Safe configurable fields include context length, batch size, rounds, warmup,
timeout, memory guard, transition concurrency, dataset, and method variant. The
GC driver also supports opt-in maximum-context search. Keep model, dataset,
power mode, attention implementation, and measurement definition fixed when
comparing methods.

## Reusability

No external workflow framework is required. The JSON schemas and structured
drivers are intended to support alternative models and CUDA-capable
unified-memory systems, but paper-level timing claims apply only to the recorded
Jetson setup. No MLCommons Collective Mind interface is currently provided.

## Notes

- Generated results are excluded from the image and must be archived alongside
  the final release.
- Historical model-loading values in `experiments/model_loading/RESULTS.md` are
  not independently reproduced by the current image.
- `manifests/paper_claims.json` records known differences between the PDF and
  located historical records, including the swapped Mistral loading values.
- Several long-context configurations can trigger system-level OOM. Follow the
  watchdog and opt-in requirements rather than probing beyond the paper matrix.
