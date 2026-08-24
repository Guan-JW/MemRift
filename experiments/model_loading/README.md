# Model loading benchmark

This benchmark compares BF16 LoRA, online NF4 QLoRA, serialized NF4 QLoRA,
and MemRift loading. Every scheduled measurement runs in a fresh process.
Timing starts immediately before the first config/checkpoint read and ends after
device placement, rank-16 MLP LoRA installation, and CUDA synchronization.

The completed experiment is summarized in `RESULTS.md`.

## Environment

The loading worker instruments a private Transformers method and therefore
refuses to run unless the environment has exactly Transformers 4.49.0 and PEFT
0.14.0. The native extension is consistently imported as
`float_split_stride` from `/workspace/src/float_split_stride`.

The runner randomizes method order with a configurable seed while retaining one
fresh worker process per measurement. It does not attempt privileged Linux
cache dropping. Raw results and summaries are explicitly labeled
`cache_state: warm` and `cache_dropped: false`; do not describe them as
cold-cache measurements.

## Run

Use logical mount paths so generated metadata remains portable between hosts:

```bash
export PYTHONPATH=/workspace/src/float_split_stride
PY=/opt/venvs/loading/bin/python

$PY /workspace/experiments/model_loading/prepare_checkpoints.py \
  --model /models/TinyLlama-1.1B-Chat-v1.0 \
  --output-root /checkpoints/TinyLlama-1.1B \
  --zstd-level 3

$PY /workspace/experiments/model_loading/run_benchmarks.py \
  --name TinyLlama-1.1B \
  --model /models/TinyLlama-1.1B-Chat-v1.0 \
  --prepared /checkpoints/TinyLlama-1.1B \
  --output-root /results/model_loading --runs 5
```

`--python` defaults to the interpreter running the orchestration script.
`--prepared` is needed only when `--methods` includes `qlora-prequant` or
`memrift`. Existing nonempty `bf16`, `nf4`, or `memrift` preparation directories
are rejected before any work starts; pass `--overwrite` to replace only the
selected directories.

Benchmark and validation runners likewise reject invocation-specific output
files unless `--overwrite` is passed. Overwrite is scoped to the selected run's
files; unrelated neighboring results are retained. A `driver.json` records the
command, environment, timestamps, schedule/stages, and any failure. The summary
or comparison is created only after all prerequisite workers succeed.

Each worker has a 3600-second timeout by default. The parent also terminates the
worker process group after three consecutive samples below 1 GiB
`MemAvailable`, avoiding a decision based on a single transient sample. Adjust
these safeguards with `--worker-timeout-seconds` and
`--min-mem-available-bytes`; set either value to zero to disable that safeguard.

`online_quantized_tensor_calls` and `prequantized_tensor_calls` instrument the
two branches in `Bnb4BitHfQuantizer.create_quantized_param`. A valid serialized
NF4 run has zero online calls and nonzero prequantized calls. A summary contains
only methods and result files from its current invocation, even when its output
directory contains older JSON files.

Validation stores each logits filename relative to its adjacent JSON record.
Run both paths and their comparison with:

```bash
$PY /workspace/experiments/model_loading/run_validation.py \
  --name TinyLlama-1.1B \
  --model /models/TinyLlama-1.1B-Chat-v1.0 \
  --prepared /checkpoints/TinyLlama-1.1B \
  --output-root /results/model_loading-validation
```
