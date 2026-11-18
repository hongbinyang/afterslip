# Afterslip environment and test tasks.
#
# Two-layer environment (docs/setup.md §4):
#   conda layer : environment.yml                          -> conda-lock.yml
#   pip layer   : requirements.in + requirements-overrides.txt
#                                                          -> requirements-macos.lock
#                                                          -> requirements-linux.lock
# The *.in / *.yml / overrides files are edited by hand; the lock files are generated
# by `make lock` and committed together with their inputs.
#
# Quick start (fresh machine with conda, uv and conda-lock installed):
#   make setup        # create the conda env from the locks and install everything
#   make test         # run the test suite in that env
#
# After editing requirements.in, requirements-overrides.txt or environment.yml:
#   make lock         # regenerate ALL lock files (every platform) in one go
#   make sync         # bring the existing env in line with the new locks (and reinstall the project)
#
# Variables (override on the command line, e.g. `make setup ENV=afterslip-test`):
#   ENV               conda environment name (default: afterslip)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV ?= afterslip
PYTHON_VERSION := 3.8

# Platforms we lock for. conda-lock uses conda platform names; uv uses target triples.
CONDA_PLATFORMS := -p osx-arm64 -p linux-64
UV_PLATFORM_MACOS := aarch64-apple-darwin
UV_PLATFORM_LINUX := x86_64-unknown-linux-gnu

# Pick the pip lock for the machine we are running on.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  PIP_LOCK := requirements-macos.lock
else
  PIP_LOCK := requirements-linux.lock
endif

# Absolute path of the env's Python, so commands do not depend on `conda activate`.
CONDA_BASE := $(shell conda info --base 2>/dev/null)
ENV_PYTHON := $(CONDA_BASE)/envs/$(ENV)/bin/python

# Common uv compile options: same inputs and overrides for every platform, with hashes.
UV_COMPILE := uv pip compile requirements.in --override requirements-overrides.txt \
              --python-version $(PYTHON_VERSION) --generate-hashes --quiet

.PHONY: help check-tools lock env sync install setup test smoke remove-env

help:  ## List the available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

check-tools:  ## Verify conda, uv and conda-lock are installed
	@command -v conda >/dev/null      || { echo "conda not found (install Miniconda or Miniforge)"; exit 1; }
	@command -v uv >/dev/null         || { echo "uv not found (e.g. brew install uv)"; exit 1; }
	@command -v conda-lock >/dev/null || { echo "conda-lock not found (e.g. uv tool install conda-lock)"; exit 1; }
	@echo "tools OK: $$(conda --version), $$(uv --version), $$(conda-lock --version)"

# ---------------------------------------------------------------------------
# Lock files (run after changing any input file)
# ---------------------------------------------------------------------------

lock: check-tools  ## Regenerate conda-lock.yml and both pip locks from the inputs
	@# Conda layer: Python and pip only, for every platform, into conda-lock.yml.
	conda-lock lock -f environment.yml $(CONDA_PLATFORMS)
	@# Pip layer: one lock per platform. uv resolves for the target platform without
	@# needing that platform, so both locks always come from the same inputs.
	$(UV_COMPILE) --python-platform $(UV_PLATFORM_MACOS) -o requirements-macos.lock
	$(UV_COMPILE) --python-platform $(UV_PLATFORM_LINUX) -o requirements-linux.lock
	@echo "Locks regenerated. Review the diff, then commit inputs and locks together."

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

env: check-tools  ## Create the conda env (name: ENV) from conda-lock.yml (fails if it exists)
	@if [ -x "$(ENV_PYTHON)" ]; then \
	  echo "Environment '$(ENV)' already exists. Use 'make sync', or 'make remove-env' first."; exit 1; \
	fi
	conda-lock install --name $(ENV) conda-lock.yml

sync: check-tools  ## Make the env's Python packages exactly match the pip lock
	@test -x "$(ENV_PYTHON)" || { echo "Environment '$(ENV)' not found; run 'make env' first."; exit 1; }
	@# Preview first. `uv pip sync` removes every package that is not in the lock. Two
	@# kinds of removal are expected: Python's own packaging tools that the conda layer
	@# brings (pip, setuptools, wheel; harmless, builds run in isolated environments), and
	@# the afterslip package itself, which is reinstalled right after. Any other planned
	@# removal means the env was modified by hand: stop and show it.
	@unexpected=$$(uv pip sync --python "$(ENV_PYTHON)" --dry-run $(PIP_LOCK) 2>&1 \
	    | grep '^ - ' | grep -Ev '^ - (pip|setuptools|wheel|afterslip)==' || true); \
	if [ -n "$$unexpected" ]; then \
	  echo "Sync would uninstall packages that are not packaging tools:"; echo "$$unexpected"; \
	  echo "Check the env (or recreate it with 'make remove-env setup')."; exit 1; \
	fi
	uv pip sync --python "$(ENV_PYTHON)" $(PIP_LOCK)
	@# Sync removed the project itself (it is not in the lock); reinstall it.
	$(MAKE) --no-print-directory install

install:  ## Install the afterslip package itself (editable, no extra dependencies)
	@# --no-deps: every dependency comes from the lock, never from pyproject.toml.
	uv pip install --python "$(ENV_PYTHON)" --no-deps -e .

setup: env sync  ## Fresh setup: create env, install locked packages and the project
	@echo "Done. Activate with: conda activate $(ENV)"

remove-env:  ## Delete the conda env (name: ENV)
	conda env remove -y -n $(ENV)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test:  ## Run the full test suite in the env
	"$(ENV_PYTHON)" -m pytest

smoke:  ## Run only the smoke tests in the env
	"$(ENV_PYTHON)" -m pytest -m smoke
