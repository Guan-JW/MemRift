# MemRift Implementation Reference

The reviewer-facing artifact abstract, checklist, resource requirements,
experiment status, and acceptance criteria are in [`ARTIFACT.md`](ARTIFACT.md).

This repository packages MemRift for controlled artifact evaluation on NVIDIA
Jetson AGX Orin. The local Docker image was built and validated on the recorded
host, its CUDA extension passed the in-image tests, and a deterministic
synthetic-data TinyLlama smoke run completed successfully. Generated run output
is excluded from version control; `manifests/source_manifest.json` records the
local verification evidence. The reviewer image is published by immutable GHCR
digest, and the archival release is published at
[doi:10.5281/zenodo.22119678](https://doi.org/10.5281/zenodo.22119678).

## Claims and non-claims

The artifact is designed to evaluate these implementation claims once the
permitted runtime sources, licensed models, and prepared checkpoints are
present:

- MemRift supports lossless split/Zstd weight compression, selectable
  full-tensor LZ4/Zstd and EBC-LZ4/EBC-Zstd activation compression,
  asynchronous weight/activation paths, LoRA/QLoRA, and gradient checkpointing
  on Jetson unified memory.
- The fixed MemRift+GC configuration uses `--gradient_checkpointing`,
  `--gc_keep_recompute_weights`, and `--gc_no_recompute_prefetch` together.
- Reported measured rounds exclude warmup rounds and distinguish PyTorch
  allocated/reserved memory, process RSS, and whole-system RAM when available.
- Model-loading measurements run in fresh processes. They are warm-cache unless
  the output explicitly records a successful cache drop.

This artifact does not claim x86 portability, numerical equivalence beyond the
validation outputs, cold-cache behavior, safe operation at arbitrary context
lengths, model-weight redistribution rights, or reproduction on a different
JetPack/PyTorch combination. It does not contain models, checkpoints,
credentials, or complete reportable paper results. The local image ID is
recorded, but it is not a portable registry digest.

`experiments/model_loading/RESULTS.md` is imported historical material, not
output generated or verified by this reviewer artifact. Its raw records are not
bundled here, so its numeric values and confirmatory wording must not be treated
as reproduced claims.

## Recorded sources and licensing

`manifests/source_manifest.json` records local snapshots without remotes:

- Training/runtime: `bb138185d2bd0b88d924d7ea20fe61d72571a7b6`
- Model-loading benchmark: `2fdc90fbcad7c20cc565480ddd7c8931af596531`

Neither inspected source snapshot originally contained a `LICENSE` or `COPYING`
file. The rights holder has released the assembled MemRift implementation,
CUDA extension, benchmark code, documentation, and reviewer infrastructure
under Apache License 2.0; see the top-level `LICENSE`. A credential was
reportedly present in a source repository remote before artifact assembly; no
remote is recorded here, but the repository owner must revoke that credential.

Models are separately licensed. TinyLlama and Mistral-7B-v0.1 report Apache-2.0
licensing; Meta Llama 3.1/3.2 use Meta's Llama community licenses and may require
acceptance and authenticated acquisition. Review the model card at the exact
revision. Never redistribute restricted weights through this repository or an
image layer.

## Tested host baseline

The host inspected while creating the infrastructure reported:

- Jetson Linux/L4T `R36.4.0`, package `36.4.0-20240912212859`
- JetPack 6.1 generation, `aarch64`
- Target GPU: Jetson AGX Orin, compute capability 8.7

The base is pinned exactly to:

```text
nvcr.io/nvidia/pytorch:24.12-py3-igpu@sha256:8b8ce6c8b0d34fea26602d52a0c1a4364339773e900047cbd9f3511eb7b79e64
```

That NGC image was validated with CUDA on this exact R36.4.0 host. PyTorch and
torchvision are inherited from NVIDIA, never replaced by public PyPI wheels.
The image builds `bitsandbytes==0.45.4` from immutable
upstream commit `f0735f95174136a71a097ce54942c1e9a9d89a3a` for SM 8.7 and
verifies it in both environments; it never installs a generic x86 wheel.
Imported historical loading values used a Jetson-specific `0.45.4.dev0` build
and must be revalidated before comparison. Two `--system-site-packages`
environments isolate the
training (`transformers==4.52.4`, `peft==0.15.2`) and model-loading
(`transformers==4.49.0`, `peft==0.14.0`) stacks. The loading versions must not be
changed silently because that benchmark patches private Transformers behavior.
Both environments use `numpy==1.26.4`, matching the NumPy 1.x ABI used to build
the pinned NVIDIA PyTorch image; the historical NumPy 2.2.3 environment is not
ABI-compatible with this base.

NVIDIA's Jetson PyTorch build reports distributed C10d unavailable. The
training image therefore removes Transformers 4.52.4's redundant eager DTensor
import and skips its tensor-parallel plan validation when Transformers disables
the parallel-style registry. Tensor parallelism remains disabled.

## Prerequisites

- Jetson AGX Orin with the L4T baseline above and sufficient cooling/power.
- Docker Engine with NVIDIA Container Runtime support. Verify `docker info`
  lists `nvidia`; JetPack installations normally supply the runtime integration.
- Access to `nvcr.io` while building. NGC authentication may be required.
- Docker BuildKit for OCI provenance/SBOM options. For example,
  `docker buildx build --sbom=true --provenance=true ...`.
- At least 40 GB free for the image/build cache, plus model and checkpoint
  storage. A 7B/8B BF16 source model alone is roughly 14-16 GB; actual storage
  must be measured and entered in the model manifest.
- Locally acquired model files and a writable results directory. Runtime is
  offline by default.
- Full evaluation uses `tatsu-lab/alpaca` at artifact-review revision
  `dce01c9b08f87459cf36a430d809084718273017`, pre-populated under `CACHE_DIR`.
  The paper did not record its historical dataset commit, so this immutable AE
  snapshot must not be described as the recovered paper snapshot. Dataset
  identities, revisions, sizes, and licenses are in `manifests/datasets.json`.
  Smoke uses deterministic synthetic data and does not require a dataset.

Normal evaluation does not require `--privileged`. Models and checkpoints are
mounted read-only. `/results` and `/cache/huggingface` are the only writable
persistent mounts. NVIDIA devices/libraries are injected by `--runtime=nvidia`.
If `tegrastats` is absent in the image, add the read-only mount
`--mount type=bind,src=/usr/bin/tegrastats,dst=/usr/bin/tegrastats,readonly`;
otherwise telemetry degrades gracefully.

## Build and validate

```bash
make image
make validate
```

The expected tag is `memrift-artifact:0.1.0-review`. `make validate` checks
`aarch64`, L4T R36.4.0, CUDA availability, SM 8.7, pinned package versions, an
odd non-contiguous BF16/FP32 roundtrip through the in-image
`float_split_stride` extension, and writable results. It does not load a model.
Record the immutable local image identifier after a successful build:

```bash
docker image inspect memrift-artifact:0.1.0-review \
  --format '{{index .RepoDigests 0}} {{.Id}}'
```

A locally built image may have only an image ID and no registry `RepoDigest`.
Do not fill `MEMRIFT_IMAGE_DIGEST` with a tag. The verified local image ID is
`sha256:4f360ec7b278814e1c97faa3b6d6ca8e69dafbb8e58df2fe81d817e9e8c8af4c`;
no registry digest is available.

## Model acquisition and checkpoint preparation

`manifests/models.json` contains the verified TinyLlama-1.1B-Chat-v1.0 snapshot
used by the smoke workflow. `configs/models.example.json` contains unresolved
templates for additional models; do not use those entries for reportable runs.

Acquire the verified TinyLlama snapshot with Git LFS outside the offline
evaluation container:

```bash
git lfs install
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
git -C TinyLlama-1.1B-Chat-v1.0 checkout \
  de253fa9783f8bd558c9ed398c8ffbe3c55cedb3
```

The preparation script ignores `.git` metadata and validates all model-content
bytes plus SHA-256 hashes for the weights, configuration, and tokenizer files.

For any additional model:

1. Acquire it directly from its publisher at an exact 40-hex Hugging Face
   commit. Do not let the evaluation container download it.
2. Add a completed entry to `manifests/models.json`.
3. Record the exact non-`.git` content size, expected files, and per-file
   SHA-256 values. Validate against `manifests/model-manifest.schema.json`.

Prepare the verified TinyLlama checkpoint from the host:

```bash
make prepare \
  MODEL_DIR=/path/to/TinyLlama-1.1B-Chat-v1.0 \
  CHECKPOINT_OUTPUT_DIR=/path/to/tinyllama-memrift
```

Preparation is local-only, verifies the manifest before loading, uses the
`float_split_stride` CUDA extension, writes `index.json` and `metadata.json`,
and rejects a non-empty destination unless `--overwrite` is explicit. Failed
preparation leaves `.incomplete`; never evaluate that directory. Preparation
can take tens of minutes and temporarily requires source model plus checkpoint
space. Measure duration and bytes on the review machine rather than relying on
an estimate.

The checkpoint contract is documented in `manifests/checkpoint.schema.json`.
Runtime validation rejects incomplete markers, malformed index or metadata,
unknown schemes, duplicate entries, unsafe paths, and every missing referenced
tensor file before training starts.

## Five-minute smoke test

The command setup takes under five minutes; model loading and two rounds may
take longer depending on model size and storage. Use a prepared small model for
the quickest test:

```bash
make smoke \
  MODEL_DIR=/path/to/model \
  CHECKPOINT_DIR=/path/to/checkpoint \
  RESULTS_DIR="$PWD/results"
```

The defaults are context 2048, batch size 1, two rounds, one warmup, activation
compaction concurrency 1, weight materialization concurrency 1, corrected GC,
offline mode, and W&B disabled. The command and resource estimate are printed
before launch.

## Full evaluation

```bash
make evaluate \
  MODEL_DIR=/path/to/model \
  CHECKPOINT_DIR=/path/to/checkpoint \
  CACHE_DIR=/path/to/prepopulated/huggingface-cache \
  RESULTS_DIR="$PWD/results"
python3 scripts/summarize_results.py results
```

Compose equivalents are `docker compose run --rm validate`,
`docker compose run --rm smoke`, and `docker compose run --rm evaluate` after
setting `MODEL_DIR`, `CHECKPOINT_DIR`, `RESULTS_DIR`, and `CACHE_DIR`.
Full evaluation additionally requires `DATASET_REVISION`; it must identify the
real dataset content acquired outside the network-isolated container.

Run the Table 6 backend matrix with:

```bash
make backends \
  MODEL_DIR=/path/to/model \
  CHECKPOINT_DIR=/path/to/checkpoint \
  CACHE_DIR=/path/to/prepopulated/huggingface-cache
```

This holds static weights at EBC-Zstd, runs an uncompressed LoRA timing
baseline, then serializes LZ4, Zstd, EBC-LZ4, and EBC-Zstd activation runs. It
writes `table6_backends.csv` incrementally and records the historical dataset
discrepancy in `table6_manifest.json`. Full-tensor and EBC ratios both account
for the complete retained activation payload.

Run the TinyLlama Tables 2-3 AE workflow with:

```bash
make tables23 \
  MODEL_DIR=/path/to/TinyLlama-1.1B-Chat-v1.0 \
  CHECKPOINT_DIR=/path/to/zstd18-training-checkpoint \
  LOADING_CHECKPOINT_DIR=/path/to/zstd3-and-nf4-loading-checkpoints \
  CACHE_DIR=/path/to/pinned-alpaca-cache \
  MEMRIFT_IMAGE_DIGEST=sha256:...
```

This validates model hashes, dataset and checkpoint receipts, serializes the
six training configurations, executes five randomized fresh-process loading
samples per method, checks a post-timing MemRift forward pass, and writes
`table2.csv`, `table3.csv`, and `tables23_manifest.json`. Results are explicitly
labeled as a pinned AE re-execution: activation attribution and the historical
FlashAttention selection remain unavailable.

Expected duration is hardware/model dependent: checkpoint preparation and 7B/
8B runs can take tens of minutes per configuration. Reserve at least the source
model size, prepared-checkpoint size, Docker layers, and logs. Report measured
times and `du -sb` values; this artifact supplies no fabricated timing or
storage result.

### Paper tables and figures

`ARTIFACT.md` and `manifests/paper_claims.json` provide the authoritative
table/figure mapping, expected values, tolerances, and current support status.
The lower-level implementation paths are:

- Training/peak-memory table: run each paper configuration with `evaluate.sh`,
  retaining the same model revision, context, batch, rounds, and warmup; then
  aggregate `run.json` files with `summarize_results.py`.
- Gradient-checkpointing comparison: compare plain LoRA+GC, QLoRA+GC, and fixed
  MemRift+GC. The fixed variant must include both recomputation correction
  flags. Never infer results from historical logs.
- Model-loading table: use `/opt/venvs/loading/bin/python` with the benchmark
  driver under `experiments/model_loading`; retain fresh-process randomized
  scheduling and summarize only requested methods.
- Timeline/telemetry figures: plot timestamped samples from the raw tegrastats
  output and label missing samples. The wrapper does not manufacture samples.

Unsupported rows remain labeled as such rather than inferred from historical
logs or reduced smoke runs.

## Output contract

Every wrapper run creates a unique directory containing:

- `raw.log` and `command.txt`
- `resolved-config.json` and `environment.json`
- `run.json`, including UTC start/end, source revisions, image digest field,
  logical model/checkpoint IDs, and exit classification
- the runtime's parsed `MEMRIFT_RESULT_JSON`

`manifests/result.schema.json` defines runtime results and
`manifests/run-record.schema.json` defines the envelope. `aggregate.json` and
`aggregate.csv` are generated at the results root. Paths in records are
relative. Exit classes are `success`, `oom`, `memory_guard`, `timeout`,
`dependency_failure`, `validation_failure`, `software_failure`, and
`user_termination`. `memory_guard` means the parent refused or terminated a run
at its configured `MemAvailable` threshold. `oom` is reserved for an observed
cgroup `oom_kill` increment.

Memory fields are not interchangeable: `peak_torch_allocated_bytes` and
`peak_torch_reserved_bytes` are CUDA allocator peaks; `peak_process_rss_bytes`
is a sampled process high-water value; `peak_system_used_bytes` and
`minimum_system_available_bytes` describe whole-system RAM. An instantaneous
RSS sample must not be called a peak. Older `*_MB_max` fields remain accepted
by the schema but should not replace the byte fields in new results.

## OOM safety

Unsafe runs reportedly rebooted a roughly 29 GiB Jetson. The following values
are imported historical source-note values. Their raw records are not bundled,
they were not reproduced by this artifact, and they must not be presented as
reviewer-artifact results:

- Plain LoRA at context 2944: about 30,068 MB system RAM.
- MemRift without GC at 2944: about 27,966 MB.
- MemRift without GC at 4096: machine-level OOM/reboot.
- Fixed MemRift+GC at 4096: about 21,841 MB.
- LoRA+GC at 4096: about 22,706 MB; QLoRA+GC: about 21,592 MB.

Never run plain LoRA or non-GC MemRift at 4096 on a 29 GiB device. There is no
maximum-context search in the safe defaults. The parent driver checks
`MemAvailable` (4 GiB default), performs the same check before launch,
terminates the entire child process group, enforces a timeout, and inspects
cgroup-v2 `memory.events` for `oom_kill`.
Cgroup limits may be added with Docker `--memory`, but Jetson GPU/unified-memory
accounting must be verified before treating that as protection. A watchdog
reduces risk; it cannot guarantee prevention of a kernel-level OOM.

## Offline operation and W&B

The image sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`WANDB_MODE=disabled`, and uses `--network=none`. The mounted model, checkpoint,
and cache must therefore be complete. No Hugging Face or W&B token is needed or
should be passed. To use W&B intentionally, remove network isolation, set
`WANDB_MODE=offline` or `online`, and provide credentials at runtime only; never
bake credentials into an image or result manifest.

## Troubleshooting

- `architecture is ..., expected aarch64`: this evaluation image is not an x86
  compatibility image. Build and run on Jetson.
- `PyTorch CUDA support is unavailable`: confirm `--runtime=nvidia`, host L4T/
  image compatibility, and NVIDIA runtime configuration.
- compute capability is not `[8, 7]`: the target and extension architecture do
  not match; do not continue evaluation.
- extension import fails: ensure `.dockerignore` excluded host `.so` files and
  inspect the Docker build log for the in-image CUDA compilation.
- model lacks `config.json` or checkpoint lacks `index.json`: correct the
  read-only mount source; the entrypoint intentionally refuses evaluation.
- offline cache miss: acquire all model/tokenizer/dataset files outside the
  evaluation run, then remount. Do not disable offline mode accidentally.
- `validation_failure`: inspect `raw.log`; a zero process exit without valid
  `MEMRIFT_RESULT_JSON` is not a successful experiment.
- `oom` or abrupt host pressure: stop, reboot if necessary, lower context/rounds,
  close other workloads, and retain the failure record. Do not lower the search
  bound for dependency or software failures.
- no tegrastats data: add the documented binary mount or use
  `--disable-tegrastats`; do not substitute NVML process memory for system RAM.
- loading benchmark version mismatch: use `/opt/venvs/loading/bin/python`, not
  the training interpreter.

After successful verification, `make export` creates the image-only archive.
Create the complete release bundle, including source, ignored raw evidence,
provenance receipts, the image, and inner checksums, with:

```bash
make release \
  CHECKPOINT_DIR=/path/to/zstd18-checkpoint \
  LOADING_CHECKPOINT_DIR=/path/to/loading-checkpoints \
  CACHE_DIR=/path/to/pinned-alpaca-cache
```

For a Zenodo deposit that does not redistribute the NVIDIA-derived container
image, add `INCLUDE_IMAGE=0`. The generated `RELEASE.json` retains the immutable
GHCR reference while omitting `image.tar.zst`.

Do not report an archive path or checksum until that command completes and the
bundle has been extracted and verified with `sha256sum -c SHA256SUMS`.
