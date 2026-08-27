IMAGE ?= memrift-artifact
VERSION ?= 0.1.0-review
DOI ?= 10.5281/zenodo.22119678
TAG ?= $(IMAGE):$(VERSION)
MODEL_DIR ?=
CHECKPOINT_DIR ?=
CHECKPOINT_OUTPUT_DIR ?=
LOADING_CHECKPOINT_DIR ?=
LOADING_CHECKPOINT_OUTPUT_DIR ?=
LOADING_RUNS ?= 5
MODEL_NAME ?= tinyllama-1.1b-chat-v1.0
MODEL_LOGICAL_ID ?= $(MODEL_NAME)
MODEL_REVISION ?= de253fa9783f8bd558c9ed398c8ffbe3c55cedb3
MODEL_WEIGHT_SHA256 ?= 6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933
CHECKPOINT_LOGICAL_ID ?= $(MODEL_NAME)-memrift
DATASET_ID ?= tatsu-lab/alpaca
DATASET_NAME ?= alpaca
DATASET_REVISION ?= dce01c9b08f87459cf36a430d809084718273017
CONTEXT_TOKENS ?= 2048
BATCH_SIZE ?= 1
ROUNDS ?= 7
WARMUP_ROUNDS ?= 1
TIMEOUT_SECONDS ?= 2400
MIN_AVAILABLE_GIB ?= 4
MIN_AVAILABLE_MB ?= 4096
NVP_MODEL ?= unknown-unrecorded
TABLES23_FLAGS ?=
EXPERIMENT ?=
OUTPUT ?=
GC_CONTEXT ?= 2944
LOOKAHEAD_BATCH ?= 1
FIDELITY_STEPS ?= 10
FIDELITY_OUTPUT ?= table5-fidelity-$(MODEL_NAME).json
MEMORY_REPETITIONS ?= 3
MEMORY_BATCH_SIZE ?= 3
MEMORY_CONTEXT ?= 2048
MEMORY_MIN_REDUCTION_PERCENT ?= 0
ABLATION_CONTEXT ?= 2048
TABLE6_CONTEXT ?= 2048
TABLE6_LEVEL ?= 1
TABLE6_WARMUP_ROUNDS ?= 2
ZSTD_LEVEL ?= 18
PREPARE_FLAGS ?=
MEMRIFT_IMAGE_DIGEST ?= unknown-unverified
SOURCE_REVISION ?= unknown-uncommitted
WHEELHOUSE_DIR ?= $(CURDIR)/wheelhouse
RESULTS_DIR ?= $(CURDIR)/results
CACHE_DIR ?= $(CURDIR)/.cache/huggingface
RELEASE_DIR ?= $(CURDIR)/dist
DOCKER ?= docker
RUN = $(DOCKER) run --rm --runtime=nvidia --network=none --ipc=host \
	--mount type=bind,src=/usr/bin/tegrastats,dst=/usr/bin/tegrastats,readonly
MOUNTS = --mount type=bind,src=$(abspath $(MODEL_DIR)),dst=/models/model,readonly \
	--mount type=bind,src=$(abspath $(CHECKPOINT_DIR)),dst=/checkpoints/model,readonly \
	--mount type=bind,src=$(abspath $(RESULTS_DIR)),dst=/results \
	--mount type=bind,src=$(abspath $(CACHE_DIR)),dst=/cache/huggingface

.PHONY: help image validate cache-dataset prepare prepare-loading model-loading entropy fidelity correctness-quick correctness-full smoke evaluate memory-comparison reviewer gc gc-max-context lookahead ablation backends tables23 check summarize export release syntax test

help:
	@printf '%s\n' 'make image' 'make validate' \
	  'make cache-dataset CACHE_DIR=/path' \
	  'make prepare MODEL_DIR=/path CHECKPOINT_OUTPUT_DIR=/path' \
	  'make prepare-loading MODEL_DIR=/path LOADING_CHECKPOINT_OUTPUT_DIR=/path' \
	  'make model-loading MODEL_DIR=/path LOADING_CHECKPOINT_DIR=/path' \
	  'make entropy MODEL_DIR=/path CACHE_DIR=/path' \
	  'make correctness-quick MODEL_DIR=/path CACHE_DIR=/path' \
	  'make correctness-full MODEL_DIR=/path CACHE_DIR=/path' \
	  'make smoke MODEL_DIR=/path CHECKPOINT_DIR=/path' \
	  'make memory-comparison MODEL_DIR=/path CHECKPOINT_DIR=/path CACHE_DIR=/path' \
	  'make reviewer MODEL_DIR=/path CHECKPOINT_DIR=/path CACHE_DIR=/path' \
	  'make evaluate MODEL_DIR=/path CHECKPOINT_DIR=/path' \
	  'make gc MODEL_DIR=/path CHECKPOINT_DIR=/path GC_CONTEXT=2944' \
	  'make lookahead MODEL_DIR=/path CHECKPOINT_DIR=/path LOOKAHEAD_BATCH=1' \
	  'make ablation MODEL_DIR=/path CHECKPOINT_DIR=/path ABLATION_CONTEXT=2048 BATCH_SIZE=4' \
	  'make backends MODEL_DIR=/path CHECKPOINT_DIR=/path TABLE6_CONTEXT=2048' \
	  'make tables23 MODEL_DIR=/path CHECKPOINT_DIR=/path LOADING_CHECKPOINT_DIR=/path CACHE_DIR=/path' \
	  'make check EXPERIMENT=loading OUTPUT=/path/to/summary.json' \
	  'make release CHECKPOINT_DIR=/path LOADING_CHECKPOINT_DIR=/path CACHE_DIR=/path'

image:
	$(DOCKER) build --network=host --platform linux/arm64 \
	  --build-context wheelhouse="$(WHEELHOUSE_DIR)" \
	  --build-arg HTTP_PROXY="$(HTTP_PROXY)" \
	  --build-arg HTTPS_PROXY="$(HTTPS_PROXY)" \
	  --build-arg NO_PROXY="$(NO_PROXY)" \
	  --build-arg BUILD_DATE=$$(date -u +%Y-%m-%dT%H:%M:%SZ) \
	  --build-arg SOURCE_REVISION="$(SOURCE_REVISION)" \
	  -f docker/Dockerfile.jetson -t $(TAG) .

validate:
	$(RUN) --tmpfs /results:rw,size=64m $(TAG) validate

cache-dataset:
	@mkdir -p "$(CACHE_DIR)"
	$(DOCKER) run --rm --runtime=nvidia --network=host \
	  --entrypoint /opt/venvs/training/bin/python \
	  --mount type=bind,src=$(abspath $(CACHE_DIR)),dst=/cache/huggingface \
	  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
	  $(TAG) /workspace/scripts/cache_datasets.py --name "$(DATASET_NAME)"

prepare:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -n "$(CHECKPOINT_OUTPUT_DIR)" || { printf '%s\n' 'CHECKPOINT_OUTPUT_DIR is required' >&2; exit 2; }
	@mkdir -p "$(CHECKPOINT_OUTPUT_DIR)"
	$(RUN) \
	  --mount type=bind,src=$(abspath $(MODEL_DIR)),dst=/models/model,readonly \
	  --mount type=bind,src=$(abspath $(CHECKPOINT_OUTPUT_DIR)),dst=/checkpoints/output \
	  $(TAG) training /workspace/scripts/prepare_weights.py \
	  --manifest /workspace/manifests/models.json --name "$(MODEL_NAME)" \
	  --model /models/model --output /checkpoints/output --zstd-level "$(ZSTD_LEVEL)" $(PREPARE_FLAGS)

prepare-loading:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -n "$(LOADING_CHECKPOINT_OUTPUT_DIR)" || { printf '%s\n' 'LOADING_CHECKPOINT_OUTPUT_DIR is required' >&2; exit 2; }
	@mkdir -p "$(LOADING_CHECKPOINT_OUTPUT_DIR)"
	$(RUN) \
	  --mount type=bind,src=$(abspath $(MODEL_DIR)),dst=/models/model,readonly \
	  --mount type=bind,src=$(abspath $(LOADING_CHECKPOINT_OUTPUT_DIR)),dst=/checkpoints/output \
	  $(TAG) loading /workspace/experiments/model_loading/prepare_checkpoints.py \
	  --model /models/model --output-root /checkpoints/output --zstd-level 3 \
	  --model-id "$(MODEL_LOGICAL_ID)" --source-revision "$(MODEL_REVISION)" \
	  --source-weight-sha256 "$(MODEL_WEIGHT_SHA256)"

model-loading:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -d "$(LOADING_CHECKPOINT_DIR)" || { printf '%s\n' 'LOADING_CHECKPOINT_DIR must name an existing directory' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)/model-loading"
	$(RUN) \
	  --mount type=bind,src=$(abspath $(MODEL_DIR)),dst=/models/model,readonly \
	  --mount type=bind,src=$(abspath $(LOADING_CHECKPOINT_DIR)),dst=/checkpoints/loading,readonly \
	  --mount type=bind,src=$(abspath $(RESULTS_DIR)),dst=/results \
	  $(TAG) loading /workspace/experiments/model_loading/run_benchmarks.py \
	  --name "$(MODEL_NAME)" --model /models/model --prepared /checkpoints/loading \
	  --output-root /results/model-loading --runs "$(LOADING_RUNS)"

entropy:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -f "$(CACHE_DIR)/memrift-dataset-receipt.json" || { printf '%s\n' 'CACHE_DIR must contain a dataset preparation receipt' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)"
	$(RUN) \
	  --mount type=bind,src=$(abspath $(MODEL_DIR)),dst=/models/model,readonly \
	  --mount type=bind,src=$(abspath $(CACHE_DIR)),dst=/cache/huggingface \
	  --mount type=bind,src=$(abspath $(RESULTS_DIR)),dst=/results \
	  $(TAG) training /workspace/experiments/entropy/collect.py \
	  --model /models/model --model-id "$(MODEL_LOGICAL_ID)" \
	  --dataset "$(DATASET_ID)" --dataset-revision "$(DATASET_REVISION)" \
	  --output "/results/table1-$(MODEL_NAME).csv"

fidelity:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -f "$(CACHE_DIR)/memrift-dataset-receipt.json" || { printf '%s\n' 'CACHE_DIR must contain a dataset preparation receipt' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)"
	$(RUN) \
	  --mount type=bind,src=$(abspath $(MODEL_DIR)),dst=/models/model,readonly \
	  --mount type=bind,src=$(abspath $(CACHE_DIR)),dst=/cache/huggingface \
	  --mount type=bind,src=$(abspath $(RESULTS_DIR)),dst=/results \
	  $(TAG) training /workspace/experiments/fidelity/roundtrip.py \
	  --model /models/model --model-id "$(MODEL_LOGICAL_ID)" \
	  --dataset "$(DATASET_ID)" --dataset-revision "$(DATASET_REVISION)" \
	  --max-length "$(CONTEXT_TOKENS)" --batch-size "$(BATCH_SIZE)" --steps "$(FIDELITY_STEPS)" \
	  --output "/results/$(FIDELITY_OUTPUT)"

correctness-quick:
	$(MAKE) fidelity FIDELITY_STEPS=10 FIDELITY_OUTPUT="correctness-quick-$(MODEL_NAME).json"

correctness-full:
	$(MAKE) fidelity FIDELITY_STEPS=100 FIDELITY_OUTPUT="correctness-full-$(MODEL_NAME).json"

smoke:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -d "$(CHECKPOINT_DIR)" || { printf '%s\n' 'CHECKPOINT_DIR must name an existing directory' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)" "$(CACHE_DIR)"
	$(RUN) $(MOUNTS) \
	  -e MODEL_LOGICAL_ID="$(MODEL_LOGICAL_ID)" \
	  -e CHECKPOINT_LOGICAL_ID="$(CHECKPOINT_LOGICAL_ID)" \
	  -e MEMRIFT_IMAGE_DIGEST="$(MEMRIFT_IMAGE_DIGEST)" \
	  -e TIMEOUT_SECONDS=1200 -e MIN_AVAILABLE_GIB="$(MIN_AVAILABLE_GIB)" \
	  $(TAG) smoke

evaluate:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -d "$(CHECKPOINT_DIR)" || { printf '%s\n' 'CHECKPOINT_DIR must name an existing directory' >&2; exit 2; }
	@test -n "$(DATASET_REVISION)" || { printf '%s\n' 'DATASET_REVISION is required' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)" "$(CACHE_DIR)"
	$(RUN) $(MOUNTS) \
	  -e MODEL_LOGICAL_ID="$(MODEL_LOGICAL_ID)" \
	  -e CHECKPOINT_LOGICAL_ID="$(CHECKPOINT_LOGICAL_ID)" \
	  -e MEMRIFT_IMAGE_DIGEST="$(MEMRIFT_IMAGE_DIGEST)" \
	  -e DATASET_ID="$(DATASET_ID)" -e DATASET_REVISION="$(DATASET_REVISION)" \
	  -e CONTEXT_TOKENS="$(CONTEXT_TOKENS)" -e BATCH_SIZE="$(BATCH_SIZE)" \
	  -e ROUNDS="$(ROUNDS)" -e WARMUP_ROUNDS="$(WARMUP_ROUNDS)" \
	  -e TIMEOUT_SECONDS="$(TIMEOUT_SECONDS)" -e MIN_AVAILABLE_GIB="$(MIN_AVAILABLE_GIB)" \
	  $(TAG) evaluate

memory-comparison:
	@root=$$(pwd -P); git_root=$$(git rev-parse --show-toplevel 2>/dev/null || true); if test "$$git_root" = "$$root"; then \
	  test -z "$$(git status --porcelain)" || { printf '%s\n' 'memory-comparison requires a clean source tree' >&2; exit 2; }; \
	else test -f .memrift-source-revision || { printf '%s\n' 'memory-comparison requires Git metadata or .memrift-source-revision' >&2; exit 2; }; fi
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -d "$(CHECKPOINT_DIR)" || { printf '%s\n' 'CHECKPOINT_DIR must name an existing directory' >&2; exit 2; }
	@test -f "$(CACHE_DIR)/memrift-dataset-receipt.json" || { printf '%s\n' 'CACHE_DIR must contain a dataset preparation receipt' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)"
	python3 scripts/run_memory_comparison.py \
	  --image "$(TAG)" --image-digest "$(MEMRIFT_IMAGE_DIGEST)" \
	  --source-revision "$$(root=$$(pwd -P); git_root=$$(git rev-parse --show-toplevel 2>/dev/null || true); if test "$$git_root" = "$$root"; then git rev-parse HEAD; else cat .memrift-source-revision; fi)" \
	  --model "$(abspath $(MODEL_DIR))" --checkpoint "$(abspath $(CHECKPOINT_DIR))" \
	  --cache "$(abspath $(CACHE_DIR))" --results-root "$(abspath $(RESULTS_DIR))" \
	  --name "$(MODEL_NAME)" --dataset "$(DATASET_ID)" --dataset-revision "$(DATASET_REVISION)" \
	  --context "$(MEMORY_CONTEXT)" --batch-size "$(MEMORY_BATCH_SIZE)" \
	  --rounds "$(ROUNDS)" --warmup-rounds "$(WARMUP_ROUNDS)" \
	  --repetitions "$(MEMORY_REPETITIONS)" \
	  --minimum-reduction-percent "$(MEMORY_MIN_REDUCTION_PERCENT)" \
	  --timeout-seconds "$(TIMEOUT_SECONDS)" --min-available-mb "$(MIN_AVAILABLE_MB)" \
	  --docker "$(DOCKER)"

reviewer:
	python3 scripts/run_reviewer_evaluation.py \
	  --image "$(TAG)" --model "$(abspath $(MODEL_DIR))" \
	  --checkpoint "$(abspath $(CHECKPOINT_DIR))" --cache "$(abspath $(CACHE_DIR))" \
	  $(if $(strip $(LOADING_CHECKPOINT_DIR)),--loading-checkpoint "$(abspath $(LOADING_CHECKPOINT_DIR))",) \
	  --results-root "$(abspath $(RESULTS_DIR))" $(REVIEWER_FLAGS)

gc:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -d "$(CHECKPOINT_DIR)" || { printf '%s\n' 'CHECKPOINT_DIR must name an existing directory' >&2; exit 2; }
	@test -f "$(CACHE_DIR)/memrift-dataset-receipt.json" || { printf '%s\n' 'CACHE_DIR must contain a dataset preparation receipt' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)/table4-gc-$(GC_CONTEXT)"
	$(RUN) $(MOUNTS) \
	  -e MEMRIFT_IMAGE_DIGEST="$(MEMRIFT_IMAGE_DIGEST)" \
	  $(TAG) training /workspace/experiments/gradient_checkpointing/run.py \
	  --model /models/model --checkpoint /checkpoints/model \
	  --dataset "$(DATASET_ID)" --dataset-revision "$(DATASET_REVISION)" \
	  --dataset-cache /cache/huggingface --results-dir "/results/table4-gc-$(GC_CONTEXT)" \
	  --matched-context "$(GC_CONTEXT)" --batch-size 1 --rounds "$(ROUNDS)" \
	  --warmup-rounds "$(WARMUP_ROUNDS)" --variants lora_gc qlora_gc memrift_gc \
	  --activation-compaction-concurrency 16 --activation-decode-concurrency 4 \
	  --weight-materialization-concurrency 4 --timeout-sec "$(TIMEOUT_SECONDS)" \
	  --min-available-mb "$(MIN_AVAILABLE_MB)"

gc-max-context:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -d "$(CHECKPOINT_DIR)" || { printf '%s\n' 'CHECKPOINT_DIR must name an existing directory' >&2; exit 2; }
	@test -f "$(CACHE_DIR)/memrift-dataset-receipt.json" || { printf '%s\n' 'CACHE_DIR must contain a dataset preparation receipt' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)/table4-max-context"
	$(RUN) $(MOUNTS) \
	  -e MEMRIFT_IMAGE_DIGEST="$(MEMRIFT_IMAGE_DIGEST)" \
	  $(TAG) training /workspace/experiments/gradient_checkpointing/run.py \
	  --model /models/model --checkpoint /checkpoints/model \
	  --dataset "$(DATASET_ID)" --dataset-revision "$(DATASET_REVISION)" \
	  --dataset-cache /cache/huggingface --results-dir /results/table4-max-context \
	  --matched-context 2944 --batch-size 1 --rounds "$(ROUNDS)" \
	  --warmup-rounds "$(WARMUP_ROUNDS)" --variants lora_gc qlora_gc memrift_gc \
	  --run-max-context --max-context-low 8192 --max-context-high 9728 --context-step 128 \
	  --activation-compaction-concurrency 16 --activation-decode-concurrency 4 \
	  --weight-materialization-concurrency 4 --timeout-sec "$(TIMEOUT_SECONDS)" \
	  --min-available-mb "$(MIN_AVAILABLE_MB)"

lookahead:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -d "$(CHECKPOINT_DIR)" || { printf '%s\n' 'CHECKPOINT_DIR must name an existing directory' >&2; exit 2; }
	@test -f "$(CACHE_DIR)/memrift-dataset-receipt.json" || { printf '%s\n' 'CACHE_DIR must contain a dataset preparation receipt' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)/figure11-$(MODEL_NAME)"
	$(RUN) $(MOUNTS) \
	  -e MEMRIFT_IMAGE_DIGEST="$(MEMRIFT_IMAGE_DIGEST)" \
	  $(TAG) training /workspace/experiments/lookahead/run.py \
	  --model /models/model --model-id "$(MODEL_LOGICAL_ID)" --checkpoint /checkpoints/model \
	  --dataset "$(DATASET_ID)" --dataset-revision "$(DATASET_REVISION)" \
	  --dataset-cache /cache/huggingface --results-dir "/results/figure11-$(MODEL_NAME)" \
	  --context 2048 --batch-size "$(LOOKAHEAD_BATCH)" --rounds "$(ROUNDS)" \
	  --warmup-rounds "$(WARMUP_ROUNDS)" --timeout-sec "$(TIMEOUT_SECONDS)" \
	  --min-available-mb "$(MIN_AVAILABLE_MB)"

ablation:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -d "$(CHECKPOINT_DIR)" || { printf '%s\n' 'CHECKPOINT_DIR must name an existing directory' >&2; exit 2; }
	@test -f "$(CACHE_DIR)/memrift-dataset-receipt.json" || { printf '%s\n' 'CACHE_DIR must contain a dataset preparation receipt' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)/figure8-$(MODEL_NAME)"
	$(RUN) $(MOUNTS) \
	  -e MEMRIFT_IMAGE_DIGEST="$(MEMRIFT_IMAGE_DIGEST)" \
	  $(TAG) training /workspace/experiments/ablation/run.py \
	  --model /models/model --model-id "$(MODEL_LOGICAL_ID)" --checkpoint /checkpoints/model \
	  --dataset "$(DATASET_ID)" --dataset-revision "$(DATASET_REVISION)" \
	  --dataset-cache /cache/huggingface --results-dir "/results/figure8-$(MODEL_NAME)" \
	  --context "$(ABLATION_CONTEXT)" --batch-size "$(BATCH_SIZE)" --rounds "$(ROUNDS)" \
	  --warmup-rounds "$(WARMUP_ROUNDS)" --timeout-sec "$(TIMEOUT_SECONDS)" \
	  --min-available-mb "$(MIN_AVAILABLE_MB)"

backends:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -d "$(CHECKPOINT_DIR)" || { printf '%s\n' 'CHECKPOINT_DIR must name an existing directory' >&2; exit 2; }
	@test -f "$(CACHE_DIR)/memrift-dataset-receipt.json" || { printf '%s\n' 'CACHE_DIR must contain a dataset preparation receipt' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)/table6-$(MODEL_NAME)"
	$(RUN) $(MOUNTS) \
	  -e MEMRIFT_IMAGE_DIGEST="$(MEMRIFT_IMAGE_DIGEST)" \
	  $(TAG) training /workspace/experiments/backends/run.py \
	  --model /models/model --model-id "$(MODEL_LOGICAL_ID)" --checkpoint /checkpoints/model \
	  --dataset "$(DATASET_ID)" --dataset-revision "$(DATASET_REVISION)" \
	  --dataset-cache /cache/huggingface --results-dir "/results/table6-$(MODEL_NAME)" \
	  --context "$(TABLE6_CONTEXT)" --batch-size "$(BATCH_SIZE)" --rounds "$(ROUNDS)" \
	  --warmup-rounds "$(TABLE6_WARMUP_ROUNDS)" --compression-level "$(TABLE6_LEVEL)" \
	  --timeout-sec "$(TIMEOUT_SECONDS)" \
	  --min-available-mb "$(MIN_AVAILABLE_MB)"

tables23:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -d "$(CHECKPOINT_DIR)" || { printf '%s\n' 'CHECKPOINT_DIR must name an existing directory' >&2; exit 2; }
	@test -d "$(LOADING_CHECKPOINT_DIR)" || { printf '%s\n' 'LOADING_CHECKPOINT_DIR must name an existing directory' >&2; exit 2; }
	@test -f "$(CACHE_DIR)/memrift-dataset-receipt.json" || { printf '%s\n' 'CACHE_DIR must contain a dataset preparation receipt' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)/tables23-$(MODEL_NAME)"
	$(RUN) $(MOUNTS) \
	  --mount type=bind,src=$(abspath $(LOADING_CHECKPOINT_DIR)),dst=/checkpoints/loading,readonly \
	  -e MEMRIFT_IMAGE_DIGEST="$(MEMRIFT_IMAGE_DIGEST)" \
	  -e MEMRIFT_NVP_MODEL="$(NVP_MODEL)" \
	  $(TAG) training /workspace/experiments/tables23/run.py \
	  --model /models/model --model-id "$(MODEL_LOGICAL_ID)" \
	  --model-manifest /workspace/manifests/models.json \
	  --checkpoint /checkpoints/model --loading-prepared /checkpoints/loading \
	  --dataset "$(DATASET_ID)" --dataset-revision "$(DATASET_REVISION)" \
	  --dataset-cache /cache/huggingface --results-dir "/results/tables23-$(MODEL_NAME)" \
	  --rounds "$(ROUNDS)" --warmup-rounds "$(WARMUP_ROUNDS)" \
	  --timeout-sec "$(TIMEOUT_SECONDS)" \
	  --min-available-mb "$(MIN_AVAILABLE_MB)" $(TABLES23_FLAGS)

check:
	@test -n "$(EXPERIMENT)" || { printf '%s\n' 'EXPERIMENT is required' >&2; exit 2; }
	@test -f "$(OUTPUT)" || { printf '%s\n' 'OUTPUT must name an existing result file' >&2; exit 2; }
	python3 scripts/check_reproduction.py --experiment "$(EXPERIMENT)" --input "$(OUTPUT)"

summarize:
	python3 scripts/summarize_results.py "$(RESULTS_DIR)"

export:
	$(DOCKER) save $(TAG) | zstd -T0 -19 -o memrift-artifact-$(VERSION).tar.zst
	sha256sum memrift-artifact-$(VERSION).tar.zst > memrift-artifact-$(VERSION).tar.zst.sha256

release:
	TAG="$(TAG)" VERSION="$(VERSION)" DOI="$(DOI)" RELEASE_DIR="$(RELEASE_DIR)" \
	  RESULTS_DIR="$(RESULTS_DIR)" DATASET_RECEIPT="$(CACHE_DIR)/memrift-dataset-receipt.json" \
	  CHECKPOINT_DIR="$(CHECKPOINT_DIR)" LOADING_CHECKPOINT_DIR="$(LOADING_CHECKPOINT_DIR)" \
	  scripts/package_release.sh

syntax:
	python3 -m compileall -q scripts experiments src/train_memrift.py
	bash -n docker/entrypoint.sh scripts/smoke.sh scripts/evaluate.sh scripts/package_release.sh

test: syntax
	python3 -m pytest -q
