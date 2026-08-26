# Figure 11 readiness lookahead

`run.py` performs the paper's `Nw,Np in {0,1,2,4,8}` sweep serially, plus
LoRA and quantized-weight baselines. It writes each row immediately to
`readiness_sweep.csv`, so completed runs survive a later timeout or OOM.

Use a 2k-token Llama-3.2-3B-Instruct run with batch size 1, then repeat for
TinyLlama-1.1B with `--batch-size 3`.
