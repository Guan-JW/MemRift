# MemRift Artifact: Reviewer Quick Start

This repository evaluates MemRift, a lossless weight-and-activation compression
runtime for LLM fine-tuning on NVIDIA unified-memory systems. Reviewers should
use the published container image rather than rebuild it.

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
export MEMRIFT_IMAGE='ghcr.io/guan-jw/memrift-artifact@sha256:45a7d409586dea875c504da09fd3e2215b2491476c151297b1c5d710d02b9979'
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

The checksummed release bundle is an alternative to the registry:

```bash
sha256sum -c memrift-artifact-0.1.0-review-release.tar.zst.sha256
mkdir memrift-release
zstd -dc memrift-artifact-0.1.0-review-release.tar.zst | tar -xf - -C memrift-release
cd memrift-release/memrift-artifact-0.1.0-review
sha256sum -c SHA256SUMS
zstd -dc image.tar.zst | docker load
```

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

## 4. Run Supported Experiments

The recommended reviewer sequence is modular. Each experiment has one command,
one primary result, and an automated check.

| Module | Purpose | Current status | Expected result |
|---|---|---|---|
| Environment | Image and CUDA validation | Verified | `ok: true` |
| Smoke | End-to-end MemRift training path | Verified functional | successful synthetic run |
| Fidelity | Lossless weight/activation codec | Verified functional | exactly 0 mismatches |
| Loading | Five-run TinyLlama loading comparison | Verified measurement | all medians within 15% |
| Entropy | TinyLlama Table 1 field entropy | Runnable | exponent entropy 2.61-2.93 bits |
| Backends | Table 6 backend implementation | Verified functional | four successful backends |

Commands and expected values are in [`REPRODUCING.md`](REPRODUCING.md).

Check an output with:

```bash
make check EXPERIMENT=smoke OUTPUT=/path/to/run.json
make check EXPERIMENT=fidelity OUTPUT=/path/to/fidelity.json
make check EXPERIMENT=loading OUTPUT=/path/to/summary.json
make check EXPERIMENT=entropy OUTPUT=/path/to/table1.csv
make check EXPERIMENT=backends OUTPUT=/path/to/table6_backends.csv
```

The checker prints structured JSON and exits nonzero when acceptance fails.

## Unsupported Or Partial Claims

The following must not be presented as reproduced by the current artifact:

- Complete Tables 2 and 3: the safe TinyLlama attempt was non-reportable because
  the 4 GiB watchdog stopped five training workers.
- Complete four-model tables: exact snapshots/checkpoints are missing for
  Llama-3.2-3B, Mistral-7B, and Llama-3.1-8B.
- Figures 3 and 7: controlled co-run orchestration is unavailable.
- Figures 9 and 10: Nsight automation and component postprocessing are
  unavailable.
- Historical FlashAttention timing: the image uses the available SDPA path.
- Activation-memory attribution and complete LM-Eval accuracy reproduction.

`manifests/paper_claims.json` is the authoritative claim matrix. A watchdog stop
is never reported as an observed OOM.

## More Information

- [`REPRODUCING.md`](REPRODUCING.md): experiment-by-experiment commands and checks
- [`ARTIFACT.md`](ARTIFACT.md): AE appendix, requirements, claims, and limitations
- [`REFERENCE.md`](REFERENCE.md): implementation, schemas, safety, and developer details
- `manifests/models.json`: verified model identity and hashes
- `manifests/datasets.json`: pinned dataset identities and licenses
- `manifests/source_manifest.json`: local verification record

The artifact source is Apache-2.0. Models and datasets retain their upstream
licenses. TinyLlama reports Apache-2.0; the pinned Alpaca dataset reports
CC-BY-NC-4.0.
