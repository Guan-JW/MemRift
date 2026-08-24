# Results directory

Generated runs are intentionally ignored by Git. Each run directory contains
`raw.log`, `command.txt`, `resolved-config.json`, `environment.json`, and
`run.json`. Run `python scripts/summarize_results.py results` to create
`aggregate.json` and `aggregate.csv`. Paths stored in records are relative;
model and checkpoint identity is represented by logical IDs.

No benchmark output is bundled. The image, native CUDA extension, and smoke
benchmark have not yet been verified for this artifact snapshot.

Numeric values under `experiments/model_loading/RESULTS.md` are imported
historical material without bundled raw records. They are not results produced
or verified by this reviewer artifact and must not be presented as such.
