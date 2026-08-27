# CGO 2027 Artifact Appendix Reference

This document is an authoring reference for the artifact appendix in the
accepted MemRift paper. It is not intended to be inserted verbatim. The CGO
2027 call requires the appendix to be at most two pages, placed before the
references, and to provide an artifact link and description. Artifact review is
single-blind, so author details are permitted.

Before camera-ready submission, replace the version DOI, tag, and commit below
if a corrected Zenodo version is published. Those three identifiers must always
refer to the same immutable source release.

The required Jetson AGX Orin is an unusual experimental setup. Before
submission, contact the AE chairs listed in the call, Olivia Hsu
(`owh@cmu.edu`) and Jackson Woodruff (`Jackson.Woodruff@ed.ac.uk`), to confirm
reviewer hardware access or an agreed remote-access arrangement.

## Suggested Appendix Text

### Artifact Availability

The MemRift artifact is archived on Zenodo at
[doi:10.5281/zenodo.22119678](https://doi.org/10.5281/zenodo.22119678).
The exact archived source is Git tag
[`v0.1.0-review`](https://github.com/Guan-JW/MemRift/tree/v0.1.0-review),
commit `72aea62c89817ed6cfe80605352ecc0543fda776`. The executable Jetson
container is publicly available by immutable digest:

```text
ghcr.io/guan-jw/memrift-artifact@sha256:45a7d409586dea875c504da09fd3e2215b2491476c151297b1c5d710d02b9979
```

The Zenodo record contains checksummed source, experiment evidence, provenance
receipts, and a clean-room verification report. It references rather than
redistributes the NVIDIA-derived container image. Model weights, dataset
content, and prepared checkpoint payloads are not redistributed; the artifact
pins their identities and provides acquisition and preparation commands.

### Artifact Description and Scope

The artifact packages the MemRift lossless weight-and-activation compression
runtime, CUDA split/merge extension, checkpoint preparation, LoRA and online
QLoRA baselines, model-loading benchmarks, safety watchdogs, and structured
result collection. The automated reviewer workflow validates the environment,
checks lossless reconstruction, runs an end-to-end smoke experiment, and
performs a balanced system-memory comparison on TinyLlama-1.1B-Chat-v1.0.
Optional stages evaluate model loading, field entropy, and four activation
compression backends.

The supported reviewer profile uses context length 2048, batch size 3, seven
rounds with one warmup, three rotated repetitions, and peak whole-system RAM
sampled by `tegrastats`. The verified profile measured median peaks of 23,140
MiB for LoRA, 24,698 MiB for online QLoRA, and 22,124 MiB for MemRift. These
measurements support a memory-reduction claim for this reviewer profile, not a
training-speed claim or the paper's exact batch-4 result.

### Hardware and Software Requirements

Reportable measurements require a 32 GB NVIDIA Jetson AGX Orin with Linux
`aarch64`, JetPack 6.1/L4T R36.4.0, NVIDIA Container Runtime, compute capability
8.7, adequate active cooling, stable power, and at least 40 GB of free storage
plus space for inputs and results. The pinned image contains CUDA 12.6, a
Jetson PyTorch 2.6 development build, Transformers, PEFT, bitsandbytes, and the
compiled MemRift CUDA extension. Runtime experiments execute offline, mount the
model and checkpoints read-only, and use the prepared local dataset cache.

The supported input is the pinned TinyLlama revision
`de253fa9783f8bd558c9ed398c8ffbe3c55cedb3`. The workflow uses the pinned
Alpaca revision `dce01c9b08f87459cf36a430d809084718273017` and prepares a
Zstd-18 training checkpoint locally. The optional loading stage additionally
prepares serialized NF4 and Zstd-3 checkpoints.

### Automated Evaluation

After following the input and checkpoint preparation in `README.md`, reviewers
set `MODEL_DIR`, `CHECKPOINT_DIR`, `CACHE_DIR`, and `RESULTS_DIR`, then run:

```bash
make reviewer TAG="$MEMRIFT_IMAGE" MODEL_DIR="$MODEL_DIR" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" CACHE_DIR="$CACHE_DIR" \
  RESULTS_DIR="$RESULTS_DIR"
```

The command is resumable and runs four isolated stages: environment validation,
quick correctness, smoke, and balanced memory comparison. It streams progress
and writes stage logs, raw outputs, `events.jsonl`, and a final
`evaluation.json`. On the recorded host, a clean core evaluation took about 31
minutes after input and checkpoint preparation. Preparation can take tens of
minutes and requires network access only while obtaining the model and pinned
dataset. The optional loading, entropy, and backend stages are selected with
`REVIEWER_FLAGS='--stages loading,entropy,backends'` and
`LOADING_CHECKPOINT_DIR`; the recorded optional-only run took about 11 minutes.
`REVIEWER_FLAGS=--full` runs all seven stages and is expected to take about 42
minutes after setup based on the two separately recorded workflows.

Expected acceptance conditions are: successful Jetson/CUDA validation; exactly
zero weight or activation reconstruction mismatches; a successful synthetic
MemRift smoke run; and completed matched memory runs whose median shows lower
MemRift peak system RAM than both baselines. A completed comparison that does
not meet the memory condition is retained as a valid negative result rather
than hidden as an execution failure. Claims in the paper appendix should be
limited to the configurations and acceptance conditions stated above.

## Two-Page Editing Checklist

- Place the final appendix before the paper references and keep it within two
  pages.
- Include the Zenodo version DOI, immutable Git tag, and container digest.
- State the exact Jetson, software, storage, and external-input requirements.
- Give the one-command core workflow, expected duration, outputs, and acceptance
  conditions.
- Map each claim requested from reviewers to a command and output file.
- Keep the requested evaluation scope aligned with the listed commands,
  configurations, and outputs.
- State how reviewers will access the unusual hardware and remain available
  during the clarification period.
- Request the Artifacts Available badge based on Zenodo. Position an Artifacts
  Evaluated badge according to the committee audit; do not claim Results
  Reproduced solely from the authors' validation runs.

The source requirements for this reference are the CGO 2027
[Artifact Evaluation call](https://2027.cgo.org/track/cgo-2027-artifact-evaluation)
and its linked appendix template and submission guidance.
