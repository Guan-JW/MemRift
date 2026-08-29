# MemRift Artifact: Reviewer Quick Start

This repository evaluates MemRift, a lossless weight-and-activation compression
runtime for LLM fine-tuning on NVIDIA unified-memory systems. Reviewers should
use the published container image rather than rebuild it.

The archived artifact is available from Zenodo at
[doi:10.5281/zenodo.22120118](https://doi.org/10.5281/zenodo.22120118).

## Platform

The released image is **not a general x86 or generic CUDA image**. It requires:

- NVIDIA Jetson AGX Orin with 32 GB unified memory
- Linux `aarch64`
- JetPack 6.1 / L4T R36.4.0
- NVIDIA Container Runtime
- GPU compute capability 8.7
- at least 40 GB free, plus model, checkpoint, dataset, and result storage

Some Python code may be adaptable to other CUDA systems, and host-only unit
tests can run elsewhere, but the container, CUDA extension, environment
validation, and reported measurements support only the platform above. Results
from another platform are not paper reproductions.

The reviewers are expected to provide the Jetson. No privileged container is
required. Evaluation runs are offline and mount models/checkpoints read-only.

## 1. Pull And Validate

Use the public immutable image reference; do not substitute the mutable version
tag when recording evaluation results.

```bash
export MEMRIFT_IMAGE='ghcr.io/guan-jw/memrift-artifact@sha256:3616f544510b46b4ccf329acb1b43c986e2a8ead5c6835cec7b3798d8c1e65d5'
docker pull "$MEMRIFT_IMAGE"
```

Validate the host, CUDA runtime, package versions, and native extension:

```bash
docker run --rm \
  --runtime=nvidia \
  --network=none \
  --ipc=host \
  --mount type=bind,src=/usr/bin/tegrastats,dst=/usr/bin/tegrastats,readonly \
  --tmpfs /results:rw,size=64m \
  "$MEMRIFT_IMAGE" validate
```

Expected: JSON containing `"ok": true`, architecture `aarch64`, L4T
`R36.4.0`, CUDA available, device `Orin`, and compute capability `[8, 7]`.

Verify and extract the checksummed Zenodo release bundle with:

```bash
sha256sum -c memrift-artifact-0.1.1-review-release.tar.zst.sha256
mkdir memrift-release
zstd -dc memrift-artifact-0.1.1-review-release.tar.zst | tar -xf - -C memrift-release
cd memrift-release/memrift-artifact-0.1.1-review
sha256sum -c SHA256SUMS
mkdir source
zstd -dc source.tar.zst | tar -xf - -C source
cd source
```

The Zenodo archive does not redistribute the NVIDIA-derived image. Pull the
immutable GHCR image as shown above.

## 2. Download Inputs

Only TinyLlama and the pinned Alpaca dataset are required for the supported
reviewer experiments. Model weights and dataset content are not embedded in the
image.

```bash
mkdir -p inputs checkpoints .cache/huggingface results

git lfs install
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  inputs/TinyLlama-1.1B-Chat-v1.0
git -C inputs/TinyLlama-1.1B-Chat-v1.0 checkout \
  de253fa9783f8bd558c9ed398c8ffbe3c55cedb3
```

The artifact validates the complete model snapshot against
`manifests/models.json`, including SHA-256 hashes. TinyLlama is approximately
2.20 GB.

Download the pinned Alpaca revision and create its receipt. This is the only
network-enabled container step:

```bash
make cache-dataset \
  TAG="$MEMRIFT_IMAGE" \
  CACHE_DIR="$PWD/.cache/huggingface"
```

The pinned revision is
`dce01c9b08f87459cf36a430d809084718273017`. The cached upstream content is
approximately 319 MB. Runtime experiments use `--network=none`.

## 3. Prepare Checkpoints

Prepare the Zstd-18 MemRift training checkpoint:

```bash
make prepare \
  TAG="$MEMRIFT_IMAGE" \
  MODEL_DIR="$PWD/inputs/TinyLlama-1.1B-Chat-v1.0" \
  CHECKPOINT_OUTPUT_DIR="$PWD/checkpoints/tinyllama-zstd18"
```

Prepare the serialized NF4 and Zstd-3 checkpoints used by loading experiments:

```bash
make prepare-loading \
  TAG="$MEMRIFT_IMAGE" \
  MODEL_DIR="$PWD/inputs/TinyLlama-1.1B-Chat-v1.0" \
  LOADING_CHECKPOINT_OUTPUT_DIR="$PWD/checkpoints/tinyllama-loading"
```

Both commands are offline, reject stale partial outputs, and write provenance
metadata. Preparation may take tens of minutes. The training checkpoint is
approximately 1.53 GB. Loading preparation creates a BF16 copy plus NF4 and
MemRift directories of approximately 2.20 GB, 0.76 GB, and 1.55 GB.

## 4. Automated Reviewer Workflow

After completing the input and checkpoint preparation above, set the common
paths and run the resumable core evaluation directly:

```bash
export MODEL_DIR="$PWD/inputs/TinyLlama-1.1B-Chat-v1.0"
export CHECKPOINT_DIR="$PWD/checkpoints/tinyllama-zstd18"
export LOADING_CHECKPOINT_DIR="$PWD/checkpoints/tinyllama-loading"
export CACHE_DIR="$PWD/.cache/huggingface"
export RESULTS_DIR="$PWD/results"

make reviewer TAG="$MEMRIFT_IMAGE" MODEL_DIR="$MODEL_DIR" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" CACHE_DIR="$CACHE_DIR" \
  RESULTS_DIR="$RESULTS_DIR"
```

This runs three stages: environment validation, quick correctness, and balanced
memory comparison. Each stage streams progress and preserves its command, logs, and raw
outputs below a configuration-identified `results/reviewer-*` directory.
`events.jsonl` records incremental progress and `evaluation.json` is the final
summary. Re-running the same command verifies the evidence hashes and skips
completed stages. The memory-optimized core is expected to take about 43 minutes
after setup.

Run the complete six-stage suite, including loading, entropy, and backends,
with:

```bash
make reviewer TAG="$MEMRIFT_IMAGE" MODEL_DIR="$MODEL_DIR" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  LOADING_CHECKPOINT_DIR="$LOADING_CHECKPOINT_DIR" \
  CACHE_DIR="$CACHE_DIR" RESULTS_DIR="$RESULTS_DIR" \
  REVIEWER_FLAGS=--full
```

The complete suite is expected to take about 54 minutes after setup based on the
43-minute core and 11-minute optional-stage runs. To run only
the three optional stages, replace `REVIEWER_FLAGS=--full` with
`REVIEWER_FLAGS='--stages loading,entropy,backends'`. The core stages remain the
default. A completed memory comparison that does not support the lower-memory
claim is retained as `valid_negative`, not treated as missing evidence.
A completed loading comparison reports MemRift's improvement relative to online
QLoRA. Missing outputs are reported as `missing_results` or `incomplete_run`;
configuration and provenance mismatches are `requirements_not_met` with the
exact observed and expected values. These outcomes never imply that a claim was
tested and rejected.

## 5. Run Individual Experiments

The recommended reviewer sequence is modular. Each experiment has one command,
one primary result, and an automated check.

| Module | Purpose | Current status | Expected result |
|---|---|---|---|
| Environment | Image and CUDA validation | Verified | `ok: true` |
| Quick correctness | 10-step lossless codec check | Reviewer default | exactly 0 mismatches |
| Memory comparison | Balanced LoRA/QLoRA/MemRift runs | Verified measurement | MemRift median below both baselines |
| Loading | Five-run TinyLlama loading comparison | Verified measurement | MemRift faster than online QLoRA |
| Entropy | TinyLlama Table 1 field entropy | Runnable | exponent entropy 2.61-2.93 bits |
| Backends | Table 6 backend implementation | Verified functional | four successful backends |

Commands and expected values are in [`REPRODUCING.md`](REPRODUCING.md).

Check an output with:

```bash
make check EXPERIMENT=smoke OUTPUT=/path/to/run.json
make check EXPERIMENT=fidelity OUTPUT=/path/to/fidelity.json
make check EXPERIMENT=memory OUTPUT=/path/to/memory-comparison/summary.json
make check EXPERIMENT=loading OUTPUT=/path/to/summary.json
make check EXPERIMENT=entropy OUTPUT=/path/to/table1.csv
make check EXPERIMENT=backends OUTPUT=/path/to/table6_backends.csv
```

The checker prints structured JSON and exits nonzero when acceptance fails.
The memory comparison uses peak whole-system RAM from `tegrastats`, not CUDA
allocator memory. A completed negative result is retained and causes the memory
claim check to fail rather than being filtered out.

The balanced memory-optimized profile validation measured median peak
whole-system RAM of 23,130 MiB for LoRA, 24,555 MiB for online QLoRA, and
21,912 MiB for MemRift. This is a 5.27% reduction from LoRA and a 10.76%
reduction from online QLoRA. Median measured-round times were 12.72 s, 13.77 s,
and 54.29 s respectively. The lower concurrency deliberately trades throughput
for a more stable whole-system memory reduction and is not presented as a
speedup. This profile is separate from the paper configuration and the
incomplete exact batch-4 Table 3 attempt.

## More Information

- [`REPRODUCING.md`](REPRODUCING.md): experiment-by-experiment commands and checks
- `manifests/models.json`: verified model identity and hashes
- `manifests/datasets.json`: pinned dataset identities and licenses
- `manifests/source_manifest.json`: local verification record

The artifact source is Apache-2.0. Models and datasets retain their upstream
licenses. TinyLlama reports Apache-2.0; the pinned Alpaca dataset reports
CC-BY-NC-4.0.
