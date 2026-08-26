# Reproducing MemRift Experiments

Run these modules after completing the image, model, dataset, and checkpoint
steps in [`README.md`](README.md). Set common paths once:

```bash
export MODEL_DIR="$PWD/inputs/TinyLlama-1.1B-Chat-v1.0"
export CHECKPOINT_DIR="$PWD/checkpoints/tinyllama-zstd18"
export LOADING_CHECKPOINT_DIR="$PWD/checkpoints/tinyllama-loading"
export CACHE_DIR="$PWD/.cache/huggingface"
export RESULTS_DIR="$PWD/results"
```

All runtime experiments use the pinned image from `MEMRIFT_IMAGE`, execute with
network access disabled, and retain raw commands, logs, telemetry, and structured
results.

## A. End-To-End Smoke

```bash
make smoke \
  TAG="$MEMRIFT_IMAGE" \
  MEMRIFT_IMAGE_DIGEST="${MEMRIFT_IMAGE##*@}" \
  MODEL_DIR="$MODEL_DIR" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  CACHE_DIR="$CACHE_DIR" \
  RESULTS_DIR="$RESULTS_DIR"
```

Expected duration: approximately one minute after model/checkpoint preparation.
Expected output: `results/smoke-<timestamp>/run.json` with `status: success`, two
synthetic rounds, one warmup round, and activation compression ratio above 1.

```bash
make check \
  EXPERIMENT=smoke \
  OUTPUT="$RESULTS_DIR/smoke-<timestamp>/run.json"
```

This verifies functionality, not a paper timing claim.

## B. Lossless Codec Fidelity

Run the paper protocol of 100 training steps:

```bash
make fidelity \
  TAG="$MEMRIFT_IMAGE" \
  MODEL_DIR="$MODEL_DIR" \
  CACHE_DIR="$CACHE_DIR" \
  RESULTS_DIR="$RESULTS_DIR" \
  FIDELITY_STEPS=100 CONTEXT_TOKENS=2048 BATCH_SIZE=1
```

Expected output: `results/table5-fidelity-tinyllama-1.1b-chat-v1.0.json`.
Acceptance requires all 100 requested steps to complete and exactly zero weight
or activation tensor mismatches.

```bash
make check \
  EXPERIMENT=fidelity \
  OUTPUT="$RESULTS_DIR/table5-fidelity-tinyllama-1.1b-chat-v1.0.json"
```

The locally verified one-step functional check covered 201 weight tensors and
497 activation tensors with zero mismatches. The checker requires completion of
the number of steps recorded by the new invocation.

## C. Five-Run Model Loading

```bash
make model-loading \
  TAG="$MEMRIFT_IMAGE" \
  MODEL_DIR="$MODEL_DIR" \
  LOADING_CHECKPOINT_DIR="$LOADING_CHECKPOINT_DIR" \
  RESULTS_DIR="$RESULTS_DIR" \
  LOADING_RUNS=5
```

Expected output:
`results/model-loading/tinyllama-1.1b-chat-v1.0/summary.json`.

Each sample uses a fresh process. Results are warm-cache medians; privileged
cache dropping is not claimed. Acceptance is within 15% of these references:

| Method | Reference median | Locally observed AE median |
|---|---:|---:|
| LoRA BF16 | 2.15 s | 1.860 s |
| QLoRA online NF4 | 4.26 s | 4.090 s |
| QLoRA serialized NF4 | 2.07 s | 1.802 s |
| MemRift Zstd-3 | 2.66 s | 2.355 s |

```bash
make check \
  EXPERIMENT=loading \
  OUTPUT="$RESULTS_DIR/model-loading/tinyllama-1.1b-chat-v1.0/summary.json"
```

The checker also verifies five samples per method, warm-cache labeling, and that
the online and serialized NF4 instrumentation took different intended paths.

## D. TinyLlama Entropy

```bash
make entropy \
  TAG="$MEMRIFT_IMAGE" \
  MODEL_DIR="$MODEL_DIR" \
  CACHE_DIR="$CACHE_DIR" \
  RESULTS_DIR="$RESULTS_DIR"
```

Expected output: `results/table1-tinyllama-1.1b-chat-v1.0.csv`, containing one
weight row and one activation row. The paper-wide expected exponent entropy is
2.61-2.93 bits per 8-bit exponent field.

```bash
make check \
  EXPERIMENT=entropy \
  OUTPUT="$RESULTS_DIR/table1-tinyllama-1.1b-chat-v1.0.csv"
```

Only the TinyLlama portion is runnable with the supplied manifest. A complete
Table 1 requires three additional exact model snapshots.

## E. Compression Backends

```bash
make backends \
  TAG="$MEMRIFT_IMAGE" \
  MEMRIFT_IMAGE_DIGEST="${MEMRIFT_IMAGE##*@}" \
  MODEL_DIR="$MODEL_DIR" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  CACHE_DIR="$CACHE_DIR" \
  RESULTS_DIR="$RESULTS_DIR" \
  TABLE6_CONTEXT=2048 BATCH_SIZE=1
```

Expected output:
`results/table6-tinyllama-1.1b-chat-v1.0/table6_backends.csv`. All four methods
(`lz4`, `zstd`, `ebc-lz4`, and `ebc-zstd`) must complete and report compression
ratios above 1.

```bash
make check \
  EXPERIMENT=backends \
  OUTPUT="$RESULTS_DIR/table6-tinyllama-1.1b-chat-v1.0/table6_backends.csv"
```

This validates the Table 6 implementation. It is not an exact historical
reproduction because the located historical records used an unpinned Guanaco
dataset while the AE workflow uses pinned Alpaca.

## Extended Workflows

The following drivers are available for exploration but are not part of the
recommended reproducible reviewer set:

| Target | Paper item | Current limitation |
|---|---|---|
| `make ablation` | Figure 8 | only reduced synthetic local evidence |
| `make lookahead` | Figure 11 | only one reduced local point |
| `make gc` | Table 4 | requires missing Llama-3.2-3B inputs |
| `make gc-max-context` | Table 4 | missing inputs and deliberate OOM risk |
| `make tables23` | Tables 2-3 | safe TinyLlama run is non-reportable |
| `make evaluate` | Figures 1 and 6 points | no complete four-model sweep |

The current Tables 2-3 result can be checked explicitly and is expected to fail
reportability rather than be mistaken for a reproduction:

```bash
make check \
  EXPERIMENT=tables23 \
  OUTPUT=results/tables23-tinyllama-1.1b-chat-v1.0/tables23_manifest.json
```

Never lower the memory watchdog merely to turn a stopped point into a claimed
paper result. Preserve the raw record and report the stop separately from OOM.
