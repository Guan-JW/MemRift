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

For the least interactive core evaluation, run:

```bash
make reviewer TAG="$MEMRIFT_IMAGE" MODEL_DIR="$MODEL_DIR" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" CACHE_DIR="$CACHE_DIR" RESULTS_DIR="$RESULTS_DIR"
```

This resumable command runs environment validation, quick correctness, smoke,
and balanced comparative memory in isolated stage directories. Console progress
is mirrored to stage logs, while `events.jsonl` and `evaluation.json` provide
machine-readable progress and final outcomes. Use `REVIEWER_FLAGS='--rerun smoke'`
to repeat one stage or `REVIEWER_FLAGS='--stages validate,correctness'` to select
a subset. Input acquisition and checkpoint preparation remain separate because
they are not network-isolated evaluation steps.

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

Run the reviewer-default 10-step check:

```bash
make correctness-quick \
  TAG="$MEMRIFT_IMAGE" \
  MODEL_DIR="$MODEL_DIR" \
  CACHE_DIR="$CACHE_DIR" \
  RESULTS_DIR="$RESULTS_DIR"
```

Expected output: `results/correctness-quick-tinyllama-1.1b-chat-v1.0.json`.
Acceptance requires all 10 requested steps to complete and exactly zero weight
or activation tensor mismatches. This checks thousands of direct weight and
activation round trips and is not a timing benchmark.

```bash
make check \
  EXPERIMENT=fidelity \
  OUTPUT="$RESULTS_DIR/correctness-quick-tinyllama-1.1b-chat-v1.0.json"
```

The optional stress profile runs 100 steps and can take over one hour on the
recorded Jetson:

```bash
make correctness-full \
  TAG="$MEMRIFT_IMAGE" MODEL_DIR="$MODEL_DIR" CACHE_DIR="$CACHE_DIR" \
  RESULTS_DIR="$RESULTS_DIR"
```

## C. Comparative Training Memory

Run three balanced repetitions of non-GC LoRA, online QLoRA, and MemRift with
the pinned TinyLlama and Alpaca inputs:

```bash
make memory-comparison \
  TAG="$MEMRIFT_IMAGE" \
  MEMRIFT_IMAGE_DIGEST="${MEMRIFT_IMAGE##*@}" \
  MODEL_DIR="$MODEL_DIR" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  CACHE_DIR="$CACHE_DIR" \
  RESULTS_DIR="$RESULTS_DIR"
```

The review profile uses context 2048, batch 3, seven rounds, one warmup, and
the unchanged 4 GiB safety guard. Method order rotates across repetitions. Each
method runs in a fresh worker process. The primary metric is peak whole-system
used RAM sampled by `tegrastats`; CUDA allocation and process RSS are secondary
diagnostics and are not substituted for that metric.

Expected output:
`results/memory-comparison-tinyllama-1.1b-chat-v1.0/summary.json`, plus
`runs.csv` and each worker's command, raw log, result, and `tegrastats.csv`.

```bash
make check \
  EXPERIMENT=memory \
  OUTPUT="$RESULTS_DIR/memory-comparison-tinyllama-1.1b-chat-v1.0/summary.json"
```

Acceptance requires all nine matched workers to complete with telemetry and
MemRift's median whole-system peak to be lower than both baselines. A completed
negative result remains valid evidence but does not support the lower-memory
claim. This is a safe AE reviewer configuration, not the paper's exact batch-4
Table 3 point.

## D. Five-Run Model Loading

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

## E. TinyLlama Entropy

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

## F. Compression Backends

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

The exact TinyLlama Table 3 input is context 2048, batch 4, without gradient
checkpointing. The current safe run supports only the statement that MemRift
completed while LoRA and QLoRA reached the safety guard; it does not provide
completed baseline peaks or a percentage reduction.
