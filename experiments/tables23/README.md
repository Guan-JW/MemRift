# TinyLlama Tables 2 and 3

This AE workflow runs the paper points for the verified TinyLlama snapshot. It
serializes LoRA, QLoRA, and MemRift at context 2048 with batch sizes 4 and 5,
then executes five randomized fresh-process warm-cache loading samples for each
loading method. It writes `table2.csv`, `table3.csv`, raw worker records,
telemetry, and `tables23_manifest.json`.

The workflow validates the model hashes, Zstd-18 training checkpoint, Zstd-3
loading checkpoint, double-quantized NF4 checkpoint, and pinned Alpaca receipt
before launching. It never equates a memory-watchdog stop with an observed OOM.

The pinned Alpaca revision is a new AE snapshot. The paper's activation-share
values are retained only as expected references because the imported runtime
does not attribute peak memory to activations. The unavailable historical
FlashAttention selection is also recorded, so timing results are re-execution
evidence rather than an undisclosed exact historical reconstruction.
