IMAGE ?= memrift-artifact
VERSION ?= 0.1.0-review
TAG ?= $(IMAGE):$(VERSION)
MODEL_DIR ?=
CHECKPOINT_DIR ?=
RESULTS_DIR ?= $(CURDIR)/results
CACHE_DIR ?= $(CURDIR)/.cache/huggingface
DOCKER ?= docker
RUN = $(DOCKER) run --rm --runtime=nvidia --network=none --ipc=host
MOUNTS = --mount type=bind,src=$(abspath $(MODEL_DIR)),dst=/models/model,readonly \
	--mount type=bind,src=$(abspath $(CHECKPOINT_DIR)),dst=/checkpoints/model,readonly \
	--mount type=bind,src=$(abspath $(RESULTS_DIR)),dst=/results \
	--mount type=bind,src=$(abspath $(CACHE_DIR)),dst=/cache/huggingface

.PHONY: help image validate smoke evaluate summarize export syntax test

help:
	@printf '%s\n' 'make image' 'make validate' \
	  'make smoke MODEL_DIR=/path CHECKPOINT_DIR=/path' \
	  'make evaluate MODEL_DIR=/path CHECKPOINT_DIR=/path RESULTS_DIR=/path'

image:
	$(DOCKER) build --platform linux/arm64 \
	  --build-arg BUILD_DATE=$$(date -u +%Y-%m-%dT%H:%M:%SZ) \
	  --build-arg SOURCE_REVISION=bb138185d2bd0b88d924d7ea20fe61d72571a7b6 \
	  -f docker/Dockerfile.jetson -t $(TAG) .

validate:
	$(RUN) --tmpfs /results:rw,size=64m $(TAG) validate

smoke evaluate:
	@test -d "$(MODEL_DIR)" || { printf '%s\n' 'MODEL_DIR must name an existing directory' >&2; exit 2; }
	@test -d "$(CHECKPOINT_DIR)" || { printf '%s\n' 'CHECKPOINT_DIR must name an existing directory' >&2; exit 2; }
	@mkdir -p "$(RESULTS_DIR)" "$(CACHE_DIR)"
	$(RUN) $(MOUNTS) $(TAG) $@

summarize:
	python3 scripts/summarize_results.py "$(RESULTS_DIR)"

export:
	$(DOCKER) save $(TAG) | zstd -T0 -19 -o memrift-artifact-$(VERSION).tar.zst
	sha256sum memrift-artifact-$(VERSION).tar.zst > memrift-artifact-$(VERSION).tar.zst.sha256

syntax:
	python3 -m compileall -q scripts
	bash -n docker/entrypoint.sh scripts/smoke.sh scripts/evaluate.sh

test: syntax
	python3 -m pytest -q
