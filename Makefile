# Makefile for the attention project (CUDA C++ vs cuTile).
#
# Linux / WSL2 is the target platform. Everything the kernels need is Python plus a CUDA toolkit,
# so there is nothing to "build" in the usual sense: the CUDA kernel is compiled at run time by
# NVRTC, and the cuTile kernel is compiled by tileiras on first launch. These targets therefore
# drive the checks and the measurements rather than a linker.
#
#   make setup      create the venv and install every dependency (run once)
#   make check-gpu  report the device and whether it can run cuTile at all
#   make test       the full correctness ladder; stops at the first failure
#   make bench      the 16-row timing sweep into results/results.csv
#   make figs       regenerate figs/ from the CSV
#   make sass       disassemble the CUDA kernel and print its resource usage
#   make clean      remove caches and generated figures, keep results/

SHELL := /bin/bash
.DEFAULT_GOAL := help

REPO    := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PY      := $(REPO)/.venv/bin/python
CUDA    ?= /usr/local/cuda-12.6
NVCC    := $(CUDA)/bin/nvcc
CUOBJ   := $(CUDA)/bin/cuobjdump

# Both models and the harness look for the toolkit through the environment.
export CUDA_HOME := $(CUDA)
export CUDA_PATH := $(CUDA)
export PATH      := $(CUDA)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA)/lib64:$(LD_LIBRARY_PATH)
export CUPY_CACHE_DIR := $(REPO)/tmp/cupy

PY_OK := $(shell test -x $(PY) && echo yes)

help:
	@echo "Attention, CUDA C++ vs cuTile"
	@echo
	@echo "  make setup      create .venv and install dependencies (once)"
	@echo "  make check-gpu  print the device, driver and compute capability"
	@echo "  make test       correctness: gate, kernel self-check, closed-form, extra, harness"
	@echo "  make bench      timing sweep -> results/results.csv"
	@echo "  make figs       regenerate figs/ from the CSV"
	@echo "  make sass       disassemble the CUDA kernel (registers, spills, instruction mix)"
	@echo "  make clean      remove caches and figures; results/ is kept"
	@echo
	@echo "  python: $(PY)"
	@echo "  cuda:   $(CUDA)"
ifneq ($(PY_OK),yes)
	@echo
	@echo "  NOTE: no venv yet. Run 'make setup' first."
endif

setup:
	bash $(REPO)/scripts/setup_linux.sh

check-gpu:
	@echo "== toolkit =="
	@$(NVCC) --version | tail -2 || echo "nvcc not found at $(NVCC)"
	@echo "== driver and device =="
	@nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm \
		--format=csv,noheader || echo "nvidia-smi failed"
	@echo "== compute capability and cuTile support =="
ifeq ($(PY_OK),yes)
	@$(PY) -c "import cupy; p = cupy.cuda.runtime.getDeviceProperties(0); \
	cc = f\"{p['major']}.{p['minor']}\"; \
	print(f'  device {p[\"name\"].decode()}  compute capability {cc}'); \
	print('  cuTile needs >= 8.0: ' + ('OK' if float(cc) >= 8.0 else 'NOT SUPPORTED on this GPU'))"
else
	@echo "  skipped: no venv (run 'make setup')"
endif

test: $(if $(PY_OK),,py-missing)
	@echo "== 1/5 gate: does cuTile run here =="
	@$(PY) $(REPO)/scripts/smoke_cutile.py
	@echo "== 2/5 cuTile kernel self-check =="
	@$(PY) $(REPO)/kernels/cutile-fused-attn/flash.py
	@echo "== 3/5 closed-form checks, no oracle =="
	@$(PY) $(REPO)/scripts/closed_form_check.py
	@echo "== 4/5 determinism, bandwidth, fp16 limits =="
	@$(PY) $(REPO)/scripts/extra_checks.py
	@echo "== 5/5 harness: oracle matrix, both models =="
	@$(PY) $(REPO)/bench/bench.py --tests-only
	@echo
	@echo "all checks passed"

bench: $(if $(PY_OK),,py-missing)
	@$(PY) $(REPO)/bench/bench.py --sizes 256,512,1024,2048

figs: $(if $(PY_OK),,py-missing)
	@$(PY) $(REPO)/bench/bench.py --figs-only

sass:
	@mkdir -p $(REPO)/tmp
	@$(NVCC) -arch=sm_86 -O3 -cubin -o $(REPO)/tmp/flash.cubin \
		$(REPO)/kernels/cuda-fused-attn/flash_v2.cu -Xptxas -v 2>&1 | grep -Ei 'registers|spill|smem|properties'
	@echo "== resource usage =="
	@$(CUOBJ) -res-usage $(REPO)/tmp/flash.cubin | grep -A2 'attn_fwd' || true
	@echo "== instruction mix =="
	@$(CUOBJ) -sass $(REPO)/tmp/flash.cubin > $(REPO)/tmp/flash.sass
	@for op in HMMA LDS STS LDGSTS LDG.E.128 LDG.E.U16 LDG.E MUFU BAR FFMA IMAD; do \
		n=$$(grep -oE " $$op[A-Z0-9.]*" $(REPO)/tmp/flash.sass | wc -l); \
		printf '  %-11s %s\n' "$$op" "$$n"; \
	done

peak:
	@mkdir -p $(REPO)/tmp
	@$(NVCC) -arch=sm_86 -O3 -o $(REPO)/tmp/peak $(REPO)/bench/peak.cu
	@$(REPO)/tmp/peak 20000

profile:
	@bash $(REPO)/scripts/profile.sh cuda 2048 0

report:
	@bash $(REPO)/scripts/build_report.sh

slides:
	@$(PY) $(REPO)/slides/build_fused_attention_demo.py

clean:
	rm -rf $(REPO)/tmp/cupy $(REPO)/tmp/cuda_tile $(REPO)/tmp/*.cubin $(REPO)/tmp/*.sass
	rm -f $(REPO)/figs/*.png
	@echo "cleaned caches and figures; results/results.csv kept"

py-missing:
	@echo "no venv at $(PY): run 'make setup' first" >&2
	@exit 1

.PHONY: help setup check-gpu test bench figs sass peak profile report slides clean py-missing
