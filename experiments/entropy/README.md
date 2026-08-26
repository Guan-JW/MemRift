# Table 1 entropy

This utility reproduces the paper's per-tensor arithmetic mean of Shannon
entropy for raw BF16 bytes and the one-bit sign, eight-bit exponent, and
seven-bit mantissa fields. Weight collection excludes parameter names containing
`norm` or `embed`, matching the imported experiment. Activation collection uses
LoRA saved-tensor hooks over the pinned dataset input.

Run inside the training environment with the model and prepared dataset cache
mounted:

```bash
/opt/venvs/training/bin/python /workspace/experiments/entropy/collect.py \
  --model /models/model \
  --model-id tinyllama-1.1b-chat-v1.0 \
  --dataset tatsu-lab/alpaca \
  --dataset-revision dce01c9b08f87459cf36a430d809084718273017 \
  --output /results/table1_entropy.csv
```

Run each of the four manifested models separately, then concatenate the rows.
The exact historical activation input was not retained; the AE snapshot and
revision must be reported with reproduced values.
