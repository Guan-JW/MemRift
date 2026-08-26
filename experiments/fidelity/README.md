# Table 5 fidelity

`roundtrip.py` runs LoRA training while directly checking every managed frozen
weight and unique BF16/FP32 tensor saved by autograd. Each check executes the
full EBC-Zstd split, compression, decompression, and merge path and compares
the reconstructed tensor's bytes. It stops on the first step with a mismatch.

The paper's second Table 5 phase requires three completed 100-step Mistral-7B
checkpoints and LM-Eval-Harness runs on GSM8K CoT and HellaSwag. Those inputs
are not bundled; codec fidelity is not a substitute for the benchmark scores.
