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
# Supported platforms (one pip lock each):
#   macOS on Apple Silicon : conda osx-arm64, uv aarch64-apple-darwin,     requirements-macos.lock
#   Linux on x86-64        : conda linux-64,  uv x86_64-unknown-linux-gnu, requirements-linux.lock
# Other platforms (Intel Mac, Linux on ARM, Windows) are not supported yet; adding one
# means adding a row here, a conda platform, a uv target, and a new lock file.
CONDA_PLATFORMS := -p osx-arm64 -p linux-64
UV_PLATFORM_MACOS := aarch64-apple-darwin
UV_PLATFORM_LINUX := x86_64-unknown-linux-gnu

# Detect the machine we are running on: operating system (uname -s) and CPU (uname -m).
# The detected platform selects the pip lock to install; `make platform` shows it.
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S)-$(UNAME_M),Darwin-arm64)
  PLATFORM := macos-arm64
  PIP_LOCK := requirements-macos.lock
else ifeq ($(UNAME_S)-$(UNAME_M),Linux-x86_64)
  PLATFORM := linux-x86_64
  PIP_LOCK := requirements-linux.lock
else
  PLATFORM := unsupported ($(UNAME_S) $(UNAME_M))
  PIP_LOCK :=
endif

# Absolute path of the env's Python, so commands do not depend on `conda activate`.
CONDA_BASE := $(shell conda info --base 2>/dev/null)
ENV_PYTHON := $(CONDA_BASE)/envs/$(ENV)/bin/python

# Upstream reference code (ReinFlow), a git submodule pinned to one commit.
REINFLOW := third_party/ReinFlow

# Common uv compile options: same inputs and overrides for every platform, with hashes.
UV_COMPILE := uv pip compile requirements.in --override requirements-overrides.txt \
              --python-version $(PYTHON_VERSION) --generate-hashes --quiet

.PHONY: help platform check-platform check-tools check-submodules lock env sync install setup data-can test smoke format lint typecheck remove-env

help:  ## List the available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Platform and tools
# ---------------------------------------------------------------------------

platform:  ## Show the detected platform and the pip lock it uses
	@echo "detected: $(UNAME_S) $(UNAME_M) -> $(PLATFORM)"
	@echo "pip lock: $(if $(PIP_LOCK),$(PIP_LOCK),none)"

check-platform:
	@if [ -z "$(PIP_LOCK)" ]; then \
	  echo "Unsupported platform: $(UNAME_S) $(UNAME_M). Supported: macOS arm64, Linux x86_64."; \
	  echo "See the platform notes at the top of this Makefile."; exit 1; \
	fi

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

env: check-platform check-tools check-submodules  ## Create the conda env (name: ENV) from conda-lock.yml (fails if it exists)
	@if [ -x "$(ENV_PYTHON)" ]; then \
	  echo "Environment '$(ENV)' already exists. Use 'make sync', or 'make remove-env' first."; exit 1; \
	fi
	conda-lock install --name $(ENV) conda-lock.yml

sync: check-platform check-tools check-submodules  ## Make the env's Python packages exactly match the pip lock
	@test -x "$(ENV_PYTHON)" || { echo "Environment '$(ENV)' not found; run 'make env' first."; exit 1; }
	@# Preview first. `uv pip sync` removes every package that is not in the lock. A version
	@# change shows as a removal plus an install of the same name and is fine. Two kinds
	@# of removal without reinstall are expected: Python's own packaging tools that the conda layer
	@# brings (pip, setuptools, wheel; harmless, builds run in isolated environments), and
	@# the two editable installs (afterslip, reinflow), which are reinstalled right after.
	@# Any other planned removal means the env was modified by hand: stop and show it.
	@# The preview's own exit status is checked first: if the preview fails (missing
	@# lock, network error, broken env), stop instead of treating "no output" as safe.
	@preview=$$(uv pip sync --python "$(ENV_PYTHON)" --dry-run $(PIP_LOCK) 2>&1); status=$$?; \
	if [ $$status -ne 0 ]; then \
	  echo "$$preview"; echo "Sync preview failed (exit $$status); nothing was changed."; exit 1; \
	fi; \
	unexpected=$$(printf '%s\n' "$$preview" | awk ' \
	  /^ \+ / { split($$2, a, /[=@ ]/); added[a[1]] = 1 } \
	  /^ - /  { split($$2, r, /[=@ ]/); removed[r[1]] = 1 } \
	  END { for (n in removed) \
	          if (!(n in added) && n !~ /^(pip|setuptools|wheel|afterslip|reinflow)$$/) print n }'); \
	if [ -n "$$unexpected" ]; then \
	  echo "Sync would uninstall packages that are not in the lock:"; echo "$$unexpected"; \
	  echo "Nothing was changed. Check the env, or recreate it: make remove-env, then make setup"; exit 1; \
	fi
	uv pip sync --python "$(ENV_PYTHON)" $(PIP_LOCK)
	@# Sync removed the editable installs (they are not in the lock); reinstall them.
	$(MAKE) --no-print-directory install

install: check-submodules  ## Install afterslip and the pinned upstream code (editable, no extra dependencies)
	@# --no-deps: every dependency comes from the lock, never from pyproject.toml. The
	@# upstream code's own extras (d4rl -> mujoco-py) are deliberately not installed.
	uv pip install --python "$(ENV_PYTHON)" --no-deps -e . -e $(REINFLOW)

check-submodules:
	@# The upstream code is a git submodule pinned to one commit (ARTIFACTS.md §3).
	@# A prerequisite of env, sync and install, so a missing submodule stops before any
	@# change to an environment.
	@test -f "$(REINFLOW)/pyproject.toml" || { \
	  echo "$(REINFLOW) is missing; run: git submodule update --init"; exit 1; }

setup:  ## Fresh setup: create env, install locked packages and the project
	@# Run the steps strictly in order, even under `make -j` (parallel make would
	@# otherwise be free to start sync before the env exists).
	$(MAKE) --no-print-directory env
	$(MAKE) --no-print-directory sync
	@echo "Done. Activate with: conda activate $(ENV)"

remove-env:  ## Delete the conda env (name: ENV); refuses base and the active env
	@# Safety: deletes only the single named env, never the base env, never the
	@# currently active env, and only if it exists.
	@if [ -z "$(ENV)" ] || [ "$(ENV)" = "base" ]; then \
	  echo "Refusing to remove '$(ENV)'."; exit 1; \
	fi
	@if [ "$(ENV)" = "$$CONDA_DEFAULT_ENV" ]; then \
	  echo "'$(ENV)' is the active env; run 'conda deactivate' first."; exit 1; \
	fi
	@if [ ! -x "$(ENV_PYTHON)" ]; then \
	  echo "Environment '$(ENV)' does not exist; nothing to remove."; \
	else \
	  echo "Removing conda env '$(ENV)' at $(CONDA_BASE)/envs/$(ENV)"; \
	  conda env remove -y -n "$(ENV)"; \
	fi

# ---------------------------------------------------------------------------
# Data (downloads and outputs under the git-ignored outputs/)
# ---------------------------------------------------------------------------

data-can:  ## Download the Can demonstrations (hash-checked) and regenerate them for robosuite 1.4.1
	"$(ENV_PYTHON)" -m afterslip.data.can_demos

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test:  ## Run the full test suite in the env
	"$(ENV_PYTHON)" -m pytest

smoke:  ## Run only the smoke tests in the env
	"$(ENV_PYTHON)" -m pytest -m smoke

# ---------------------------------------------------------------------------
# Code style (ruff; rules and settings in pyproject.toml)
# ---------------------------------------------------------------------------

format:  ## Auto-format all Python code and apply safe lint fixes
	"$(ENV_PYTHON)" -m ruff format .
	"$(ENV_PYTHON)" -m ruff check --fix .

lint:  ## Check formatting and lint rules without changing files (what CI runs)
	"$(ENV_PYTHON)" -m ruff format --check .
	"$(ENV_PYTHON)" -m ruff check .

typecheck:  ## Static type check of the package with mypy (settings in pyproject.toml)
	"$(ENV_PYTHON)" -m mypy afterslip
