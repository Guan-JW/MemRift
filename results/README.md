# Results directory

Generated runs are intentionally ignored by Git. Each run directory contains
`raw.log`, `command.txt`, `resolved-config.json`, `environment.json`, and
`run.json`. Run `python scripts/summarize_results.py results` to create
`aggregate.json` and `aggregate.csv`. Paths stored in records are relative;
model and checkpoint identity is represented by logical IDs.

Generated benchmark output is not bundled in source control. The local
`tables23-tinyllama-1.1b-chat-v1.0` run contains provenance-checked outputs, but
is non-reportable: the 4 GiB watchdog stopped five training workers while all
20 loading workers completed. Its status, hashes, canonical image ID, and test
counts are recorded in `manifests/source_manifest.json`. The complete result
directory, not only its CSV summaries, must be archived with the final release.

The local `memory-comparison-tinyllama-1.1b-chat-v1.0` run is reportable for its
exact batch-3 reviewer profile. All nine workers completed, raw telemetry passed
the independent checker, and MemRift reduced median peak whole-system RAM by
4.39% versus LoRA and 10.42% versus online QLoRA. The tracked provenance and
summary hashes are recorded in `manifests/source_manifest.json`; the raw result
directory must be included in the final evidence archive.

Numeric values under `experiments/model_loading/RESULTS.md` are imported
historical material without bundled raw records. They are not results produced
or verified by this reviewer artifact and must not be presented as such.
