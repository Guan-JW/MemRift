# MemRift Reviewer Artifact

This repository packages MemRift for controlled artifact evaluation on NVIDIA
Jetson AGX Orin. It is an infrastructure snapshot, not a verified container
release. The Docker image has **not** been built, its CUDA extension has **not**
been tested in the image, and no smoke result is included. Do not cite the
presence of these files as evidence that an experiment completed.

## Claims and non-claims

The artifact is designed to evaluate these implementation claims once the
permitted runtime sources, licensed models, and prepared checkpoints are
present:

- MemRift supports lossless split/Zstd weight compression, activation
  compression, asynchronous weight/activation paths, LoRA/QLoRA, and gradient
  checkpointing on Jetson unified memory.
- The fixed MemRift+GC configuration uses `--gradient_checkpointing`,
  `--gc_keep_recompute_weights`, and `--gc_no_recompute_prefetch` together.
- Reported measured rounds exclude warmup rounds and distinguish PyTorch
  allocated/reserved memory, process RSS, and whole-system RAM when available.
- Model-loading measurements run in fresh processes. They are warm-cache unless
  the output explicitly records a successful cache drop.

This artifact does not claim x86 portability, numerical equivalence beyond the
validation outputs, cold-cache behavior, safe operation at arbitrary context
lengths, model-weight redistribution rights, or reproduction on a different
JetPack/PyTorch combination. It does not contain models, checkpoints, paper
results, credentials, or an immutable digest for an artifact image.

`experiments/model_loading/RESULTS.md` is imported historical material, not
output generated or verified by this reviewer artifact. Its raw records are not
bundled here, so its numeric values and confirmatory wording must not be treated
as reproduced claims.

## Recorded sources and licensing

`manifests/source_manifest.json` records local snapshots without remotes:

- Training/runtime: `bb138185d2bd0b88d924d7ea20fe61d72571a7b6`
- Model-loading benchmark: `2fdc90fbcad7c20cc565480ddd7c8931af596531`

Neither inspected snapshot contained a `LICENSE` or `COPYING` file. Consequently
the top-level `LICENSE` grants no rights to imported source. Obtain permission
from the copyright holders before use beyond artifact review. A credential was
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

The compatibility of that NGC image with this exact R36.4.0 host remains to be
verified. PyTorch and torchvision are inherited from NVIDIA, never replaced by
public PyPI wheels. The image builds `bitsandbytes==0.45.4` from immutable
upstream commit `f0735f95174136a71a097ce54942c1e9a9d89a3a` for SM 8.7 and
verifies it in both environments; it never installs a generic x86 wheel.
Imported historical loading values used a Jetson-specific `0.45.4.dev0` build
and must be revalidated before comparison. Two `--system-site-packages`
environments isolate the
training (`transformers==4.52.4`, `peft==0.15.2`) and model-loading
(`transformers==4.49.0`, `peft==0.14.0`) stacks. The loading versions must not be
changed silently because that benchmark patches private Transformers behavior.

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
- Full evaluation requires the real training dataset to be pre-populated under
  `CACHE_DIR` at one exact 40-hex Hugging Face commit. Set that commit as
  `DATASET_REVISION`; evaluation refuses an empty cache or unresolved revision.
  Smoke uses deterministic synthetic data and does not require this dataset.

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
Do not fill `MEMRIFT_IMAGE_DIGEST` with a tag. The current image digest is
`unknown-unverified`.

## Model acquisition and checkpoint preparation

`configs/models.example.json` contains examples for TinyLlama-1.1B-Chat-v1.0,
Llama-3.2-3B-Instruct, Mistral-7B-v0.1, and Llama-3.1-8B. The null revisions and
sizes are deliberate: guessing these values would defeat reproducibility.

1. Acquire a model directly from its publisher at an exact 40-hex Hugging Face
   commit. Do not let the evaluation container download it.
2. Copy the relevant examples into `manifests/models.json`.
3. Fill `revision`, exact total `expected_bytes`, expected files, and available
   per-file SHA-256 values. Validate against
   `manifests/model-manifest.schema.json`.
4. Mount the model read-only and run the recorded preparation command. For
   example, inside the training environment:

```bash
/opt/venvs/training/bin/python scripts/prepare_weights.py \
  --manifest manifests/models.json \
  --name llama-3.2-3b-instruct \
  --model /models/llama-3.2-3b-instruct \
  --output /checkpoints/llama-3.2-3b-instruct/memrift \
  --zstd-level 18
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
  DATASET_REVISION=0123456789abcdef0123456789abcdef01234567 \
  RESULTS_DIR="$PWD/results"
python3 scripts/summarize_results.py results
```

Compose equivalents are `docker compose run --rm validate`,
`docker compose run --rm smoke`, and `docker compose run --rm evaluate` after
setting `MODEL_DIR`, `CHECKPOINT_DIR`, `RESULTS_DIR`, and `CACHE_DIR`.
Full evaluation additionally requires `DATASET_REVISION`; it must identify the
real dataset content acquired outside the network-isolated container.

Expected duration is hardware/model dependent: checkpoint preparation and 7B/
8B runs can take tens of minutes per configuration. Reserve at least the source
model size, prepared-checkpoint size, Docker layers, and logs. Report measured
times and `du -sb` values; this unverified snapshot supplies no fabricated
timing or storage result.

### Paper tables and figures

No archival paper citation, table numbering, plotting program, or published
aggregate was supplied in the permitted infrastructure scope, so an exact
table/figure mapping cannot be asserted. Once those identifiers are supplied:

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

This is the complete reproducible guidance available without inventing paper
metadata. Any final artifact should replace this paragraph with exact commands
for every numbered table and figure.

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

After successful verification, export and checksum with `make export`. The
archive path and checksum do not exist yet and must not be reported until that
command completes.
