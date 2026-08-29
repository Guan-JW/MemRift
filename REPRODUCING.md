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

This resumable command runs environment validation, quick correctness, and
balanced comparative memory in isolated stage directories. Console progress
is mirrored to stage logs, while `events.jsonl` and `evaluation.json` provide
machine-readable progress and final outcomes. Use `REVIEWER_FLAGS='--rerun memory'`
to repeat one stage or `REVIEWER_FLAGS='--stages validate,correctness'` to select
a subset. Input acquisition and checkpoint preparation remain separate because
they are not network-isolated evaluation steps.

Use `REVIEWER_FLAGS=--full` and provide `LOADING_CHECKPOINT_DIR` to append the
five-run loading comparison, TinyLlama entropy collector, and four-backend
comparison. The default remains the shorter three-stage core workflow; the full
workflow contains six stages. The smoke experiment below remains available
individually but is not part of either reviewer workflow.

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

The memory-optimized review profile uses context 2048, batch 3, seven rounds,
one warmup, and the unchanged 4 GiB safety guard. MemRift uses one activation
compaction worker, one activation decode worker, and one weight materialization
worker to limit asynchronous transition buffers. Method order rotates across
repetitions. Each method runs in a fresh worker process. The primary metric is
peak whole-system used RAM sampled by `tegrastats`; CUDA allocation and process
RSS are secondary diagnostics and are not substituted for that metric.

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
Table 3 point. It is explicitly optimized for memory rather than throughput.

The verified artifact run completed all nine workers and produced these
whole-system medians:

| Method | Peak RAM | Measured-round time |
|---|---:|---:|
| LoRA | 23,140 MiB | 12.77 s |
| Online QLoRA | 24,698 MiB | 13.85 s |
| MemRift | 22,124 MiB | 15.45 s |

MemRift reduced median peak RAM by 4.39% relative to LoRA and 10.42% relative
to online QLoRA. It was slower in this profile, so no speedup is claimed.

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
cache dropping is not claimed. The historical reference and locally observed
values below are informational:

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
Acceptance requires positive finite medians and MemRift loading faster than
online QLoRA. The checker reports the relative improvement percentage. LoRA and
prequantized QLoRA remain informative comparisons, not acceptance baselines.
Missing output is reported as `missing_results` or `incomplete_run`; mismatched
requirements include exact observed and expected values.

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
