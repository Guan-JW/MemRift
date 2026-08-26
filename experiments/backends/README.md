# Table 6 compression backends

This workflow compares four activation representations while holding prepared
weights at EBC-Zstd: full-tensor LZ4, full-tensor Zstd, EBC-LZ4, and EBC-Zstd.
It runs an uncompressed LoRA timing baseline first and writes
`table6_backends.csv` incrementally after every backend.

The compression ratio is the original activation bytes divided by all retained
payload bytes. For EBC this denominator includes both the compressed exponent
stream and the uncompressed sign/mantissa byte stream.

Located historical records used `timdettmers/openassistant-guanaco` without an
immutable revision. The reviewer command defaults to the pinned Alpaca snapshot
and records this discrepancy in `table6_manifest.json`.
