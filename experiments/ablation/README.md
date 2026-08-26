# Figure 8 pipeline ablation

The driver runs LoRA, MemRift weight-only, and MemRift weight-plus-activation
sequentially and reports peak-system-memory reduction relative to LoRA. Use the
paper configurations: TinyLlama 2048x4, Llama-3.2 3000x1, Mistral 1600x1, and
Llama-3.1 1024x1.
