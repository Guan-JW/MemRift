sync
sh -c 'echo 3 > /proc/sys/vm/drop_caches'
# swapoff -a
# swapon -a
python - <<'PY'
import torch, gc
torch.cuda.empty_cache()
try:
    torch.cuda._pin_memory_empty_cache()
except AttributeError:
    pass
gc.collect()
PY
