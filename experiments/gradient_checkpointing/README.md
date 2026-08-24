# Gradient Checkpointing Experiments

This driver compares LoRA, LoRA+GC, MemRift, and fixed MemRift+GC. The fixed
variant always enables `--gc_keep_recompute_weights` and
`--gc_no_recompute_prefetch` in addition to gradient checkpointing.

## Safe Default Run

The defaults use context 2048, batch size 1, two rounds with one warmup,
single-job async admission, W&B disabled, and no maximum-context search.

```bash
python3 experiments/gradient_checkpointing/run.py \
  --model /models/TinyLlama-1.1B-Chat-v1.0 \
  --checkpoint /checkpoints/TinyLlama-1.1B-Chat-v1.0/memrift \
  --results-dir /results/gradient_checkpointing
```

Use `--variants lora_gc memrift_gc` for the safest comparison. Maximum-context
search is opt-in with `--run-max-context`; it lowers the search bound only for
confirmed OOM exits. Non-GC runs at context 4096 or greater are refused unless
`--allow-unsafe` is explicitly supplied.

## Safety Controls

- `--min-available-mb` terminates the complete worker process group before
  system available memory falls below the threshold. The default is 4096 MiB.
- `--timeout-sec` terminates hung workers and classifies them as `timeout`.
- `--cgroup-memory-limit-mb` optionally creates a cgroup-v2 limit when the
  current user may write `/sys/fs/cgroup`; unsupported setups are recorded and
  continue under the memory watchdog.
- SIGINT and SIGTERM terminate the complete worker process group and are
  classified as `user_termination`.

Exit classes are `ok`, `oom`, `timeout`, `safety_stop`, `dependency_failure`,
`validation_failure`, `software_failure`, and `user_termination`.

## Outputs

All files are written below `--results-dir`: aggregate JSON/CSV, resolved
configuration, environment metadata, and a per-run directory containing the
raw log, exact command, resource estimate, structured result, and timestamps.
GPU memory fields separately report peak PyTorch allocated and reserved memory.
