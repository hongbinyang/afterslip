# Development environment and technology stack

Status: draft. Versions listed as **[candidate]** come from the upstream reference code's published install instructions and have not yet been installed or tested here. M0 items 0.0b and 0.2a confirm or replace them, and this document is updated with what was actually tested. Artifact licenses are recorded and verified in `ARTIFACTS.md` (FR-ART-02), not here.

The roadmap has two tracks (pending confirmation in the requirements): the **Core track** on the robosuite Panda / Robomimic Can benchmark, and an **FR3 track** on the Franka Research 3 model from MuJoCo Menagerie (learning exercises early, a sorting scene later). Whether both tracks can share one environment is to be confirmed by the clean setup (M0 item 0.0b); the FR3 check in §7 ran in the Core candidate environment.

## 1. Platforms

| Platform | Role | Status |
|---|---|---|
| Linux x86-64 (Ubuntu 22.04 or later) | Primary: training, evaluation, CPU smoke test (NFR-PORT-03 SHALL) | Untested |
| Linux x86-64 + NVIDIA GPU (CUDA) | Training at scale (local or rented cloud GPU, NFR-CMP-01) | Untested |
| macOS Apple Silicon (arm64) | Development, unit tests, inference smoke (NFR-PORT-03 SHOULD) | **Partially tested** (§7): environment creation, stepping and offscreen video pass. Checkpoint loading, policy inference, training and the upstream ReinFlow code are untested |

The upstream install instructions use Linux-only steps (`apt-get`, EGL via `/dev/dri`). Beyond the limited check in §7, macOS support is unverified and is recorded as a known limitation until shown otherwise (NFR-PORT-04).

## 2. Technology stack

| Layer | Choice | Version | Source / note |
|---|---|---|---|
| Language | Python | 3.8 **[candidate]** | Version used by the upstream install guide (`requires-python >= 3.8`). See §5, decision D-1 |
| Physics engine | MuJoCo via the official `mujoco` Python bindings | 3.2.3, pinned in the pip locks; tested on macOS and Linux x86-64 (§6.5) | FR-ENV-01: only physics engine |
| Robot simulation framework | robosuite | 1.4.1 | Upstream pins `v1.4.1` |
| Task suite and demonstrations | robomimic (task: Can) | v0.4.0, git commit `7c66e7a41b5d9dcc905b1a68346bfee1b49b79c9` | Not on PyPI; installed from git by commit (§2.2, D-3) |
| Environment API | gym | 0.22.0 **[candidate]** | Pinned upstream. Our task interface (FR-ENV-03) exposes a Gymnasium-style API and wraps whatever the upstream environment uses |
| Deep learning | PyTorch | ≥ 2.4.0 upstream; exact version pinned at 0.0b | Device chosen by config: `cpu`, `cuda`, `mps` (NFR-PORT-02) |
| RL fine-tuning algorithm | ReinFlow reference implementation | `release` branch, commit `e722e151bed767f3ffef47527cf697f2358af55d` (2026-04-24) | Git submodule `third_party/ReinFlow`, installed editable without its dependencies (§2.2). MIT license (repository `LICENSE`) |
| Configuration | Hydra + OmegaConf | hydra-core 1.3.2, omegaconf 2.3.0 **[candidate]** | Used by upstream; our configs follow the same system |
| Experiment tracking | Local file logger (always on) | — | FR-TRK-02. W&B (`wandb==0.17.7`, as pinned upstream) is installed because the upstream training code imports it, and is switched off by default (`afterslip.upstream`) |
| Rendering (headless) | EGL (GPU) or OSMesa (CPU fallback) | System libraries; OSMesa from Ubuntu `libosmesa6-dev` (tested, §6.5) | Needed for videos (FR-EVAL-06) and image observations |
| Testing | pytest (markers `unit`, `integration`, `smoke`), pytest-timeout, hypothesis (property-based tests) | pytest 8.3.5 installed locally; others pinned at 0.0b | NFR-COD-01; levels and gates in [test-plan.md](test-plan.md) |
| Code quality | ruff (lint + format), mypy (type checks) | Pinned in the pip locks; `make lint`, `make typecheck` | NFR-COD-01 (type hints) |
| Video recording | MuJoCo offscreen renderer → frames → imageio + imageio-ffmpeg → MP4 (H.264) | imageio 2.35.1, imageio-ffmpeg 0.5.1 (local) | FR-EVAL-06; see §2.1 |
| CI | GitHub Actions, `ubuntu-22.04`, CPU | `.github/workflows/ci.yml` | On every push and pull request: install from the locks, `make lint`, `make typecheck`, `make test` with OSMesa rendering (0.0c, T0.2) |

### 2.2 Upstream code and its dependencies (milestone 0.2a)

The upstream code is pinned as a git submodule and installed by `make install` with `--no-deps`, so every dependency comes from our locks. Its own `pyproject.toml` is the reference for versions; differences are listed here.

| Package | Upstream (ReinFlow `e722e15`) | This project | Reason |
|---|---|---|---|
| robosuite | git tag `v1.4.1` (extra `robomimic`) | 1.4.1 from PyPI | Same version |
| robomimic | git, unpinned HEAD (extra `robomimic`) | v0.4.0, git commit `7c66e7a` | PyPI 0.3.0 imports `mujoco_py` unconditionally in its robosuite wrapper, so the Can environment fails without mujoco-py; v0.4.0 guards that import. A pinned commit replaces the moving HEAD |
| MuJoCo bindings | not pinned for Robomimic | 3.2.3 | Resolved for robosuite 1.4.1; mujoco-py is **not** needed for Robomimic Can (tested, 0.2a) |
| d4rl, cython, patchelf (extra `robomimic`) | installed | **not installed** | d4rl requires mujoco-py. It is imported only by upstream's `script/run.py` launcher and by the dummy environment used for image or rendered runs; the state-based Robomimic path uses neither. Our training entry point will import the upstream agents directly (M1) |
| wandb | 0.17.7 | 0.17.7 | Imported at module level by the training agents; disabled by default |
| gdown, tqdm, matplotlib, hydra-core, omegaconf, einops, gym, imageio | pinned | same pins | Imported by the upstream code |
| av, pandas, seaborn, pypdf, pretty_errors | pinned | not installed | Not imported by the flow-policy and Robomimic code paths (plotting and utility scripts only). `av` is assessed separately as the video encoder (ARTIFACTS.md §5, G1) |

Pretraining runs through `afterslip.upstream.pretrain` (`make pretrain-can`), not upstream's `script/run.py`, which imports d4rl. It loads the config, applies `key=value` overrides and runs the agent named by `_target_`. One deviation: upstream's pretraining loop moves batches with a helper whose device defaults to `cuda:0`, which fails on CPU and MPS; the launcher binds it to the configured `device` without modifying the upstream code.

The upstream code reads `REINFLOW_DIR`, `REINFLOW_DATA_DIR` and `REINFLOW_LOG_DIR` at import time; `afterslip.upstream.configure_reinflow()` sets them (data and logs under the git-ignored `outputs/reinflow/`) and must run before any upstream import. The upstream packages have generic top-level names (`agent`, `model`, `env`, `util`); project code imports them only after that call.


### 2.1 Video recording

- **Rendering backend by platform:** on macOS, robosuite 1.4.1's offscreen renderer (`MjSim.render`) uses the native OpenGL context (CGL); this was tested in §7. On headless Linux the backends are EGL with a GPU, or OSMesa on CPU (`MUJOCO_GL=egl` or `osmesa`, with `PYOPENGL_PLATFORM` set to the same value). OSMesa was tested on Linux x86-64 CPU (§6.5), including the upright-frame check; EGL is **untested**. The backend is recorded in every video's metadata.
- **Cameras:** policies are state-based (decision D-2), so no camera is a policy input. Videos use **one observer camera**: a fixed, third-person view that records the scene and is never seen by the policy. Core uses robosuite's `agentview`; the FR3 track defines an equivalent fixed camera in its own scene file, since the Menagerie FR3 model contains no camera. Rendering for recording must not change simulator state or random number streams.
- **Settings in config (candidates):** 256×256 for routine evidence and 512×512 for demos; 20 fps (equal to the 20 Hz control frequency, one frame per control step).
- **Metadata per video:** episode or bank-state ID, seed, policy checkpoint hash, suite, camera, fps, resolution, backend, frame count and episode length (see milestones.md, "Video").
- **Image orientation:** with robosuite 1.4.1 `MjSim.render` on macOS (§7), frames are returned vertically flipped and are flipped back before encoding. Other render APIs and platforms are to be checked when used.
- Rendering costs throughput. In the §7 runs, 256×256 rendering every step ran at about 20 steps/s on the first run and 95 steps/s on a later run, on the same machine (the cause of the difference, e.g. warm-up, is not isolated). Training runs therefore do not record by default, and evaluation records only the configured subset.

Not needed for the Core scope: `mujoco-py` and the MuJoCo 2.1 binaries (`mujoco210`), `d4rl`, `dm_control`. Upstream uses them only for its Gym and Franka Kitchen benchmarks. Whether Robomimic Can is fully free of `mujoco-py` is checked at 0.2a.

## 3. Models, data and assets (candidates)

All entries are candidates until recorded and verified in `ARTIFACTS.md`. An artifact whose license or access cannot be verified is not used (FR-ART-02).

| Artifact | Candidate source | License status |
|---|---|---|
| Pretrained flow-matching policy π₀ for Robomimic Can (1-ReFlow) | ReinFlow checkpoints on Hugging Face: `huggingface.co/datasets/ReinFlow/ReinFlow-data-checkpoints-logs` (dataset card: MIT). The released Can 1-ReFlow checkpoint `log/log_robomimic_pretrained/can/ReFlow/state_2000.pt` is **image-based** (verified 2026-11-19: ViT image encoder weights; loaded by the image config), so it does not fit D-2 | Recorded in `ARTIFACTS.md` §3; not in use |
| Normalization statistics for Can | Same source (`normalization.npz`) | To verify |
| Robomimic Can demonstrations | robomimic v1.5 proficient-human low-dimensional file on Hugging Face; `make data-can` downloads it, checks its sha256 and regenerates the observations for robosuite 1.4.1 (design.md §10) | MIT (dataset card); recorded in `ARTIFACTS.md` §3 |
| Robot model (Franka Panda) and Can scene | Bundled with robosuite | robosuite license; to record |
| Franka Research 3 arm model (`franka_fr3`), FR3 track | github.com/google-deepmind/mujoco_menagerie, directory `franka_fr3`; candidate pin: commit `370c43965323c0944b433b202aac5058b5f907a9` (2026-11-10) | Model directory contains an Apache-2.0 `LICENSE` file (read 2026-11-16); to record in `ARTIFACTS.md` |
| Franka Hand (gripper) for the FR3 track | Menagerie `franka_emika_panda` includes a hand model; attaching it to the FR3 flange is a candidate approach | To verify; the tested Menagerie `franka_fr3` model at commit `370c439` has **no gripper** (§7); other FR3 models are not assessed |
| MuJoCo Menagerie Panda | github.com/google-deepmind/mujoco_menagerie | Per-model license; to verify if used |

Known configuration of the upstream Can fine-tuning setup (the image-based 1-ReFlow config; upstream has no state-based 1-ReFlow config for Can. Its state-based Can configs, for other policy types, use 23-dimensional observations: `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `object`):
- `action_dim` 7, `horizon_steps` 4, `act_steps` 4, `cond_steps` 1, `denoising_steps` 1, `max_episode_steps` 300
- Low-dimensional keys: `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`
- Image (image config only; not used under D-2): `robot0_eye_in_hand_image`, 96×96 RGB

`cond_steps` 1 means no observation history, so the FR-DIS-10 reconstruction rule reduces to: reset, discard the pending action chunk, replan. This still needs confirming for the chosen modality.

## 4. Package and dependency management

### 4.1 Approach (candidate)

This approach is a candidate until a clean install from lock files has been tested on Linux and macOS (M0 item 0.0b). The environment has two layers. Each layer has its own lock.

1. **Conda layer (interpreter and native libraries).** Miniforge or Miniconda provides this layer. With Miniconda, the environment is restricted to the conda-forge channel (`channels: [conda-forge]` with `nodefaults` in `environment.yml`, and `channel_priority: strict`), so that the same packages resolve as with Miniforge. The layer provides Python and non-Python libraries such as OSMesa/EGL and ffmpeg. The layer is declared in `environment.yml` and locked per platform with `conda-lock` (`conda-lock.yml` for `linux-64` and `osx-arm64`).
2. **pip layer (Python packages).** Direct dependencies are declared in `requirements.in`. They are compiled to fully pinned, hashed lock files with `uv pip compile`, one per platform: `requirements-linux.lock` and `requirements-macos.lock`. **Caution, tested 2026-11-16 (uv 0.12.5):** `uv pip sync` inside a conda env uninstalls conda-installed Python packages that are not in the lock (it left `pip`, `setuptools` and `wheel` alone). The candidate safe procedure is therefore: (a) the conda layer contains no Python packages other than `python` and `pip`; (b) install with `uv pip sync` only into such an environment; (c) run `uv pip sync --dry-run` first and stop if it plans to uninstall anything not installed by the pip layer. Alternatively `uv pip install -r <lock>` never uninstalls, at the cost of not removing stray packages; a check that compares `pip freeze` with the lock then replaces that guarantee.
3. **Project package.** `afterslip` is declared in `pyproject.toml` and installed in editable mode (`pip install -e . --no-deps`), so dependencies come only from the lock files.
4. **Upstream code.** ReinFlow is a git submodule under `third_party/ReinFlow`, pinned to one commit. Local changes live as reviewable patch files in `third_party/patches/` (NFR-COD-02), never as edits inside the submodule.
5. **Runtime record.** The lock files define the intended environment. Every research run additionally records the actual runtime versions, hardware and backend in its run manifest under `outputs/` (FR-TRK-01), using standard tooling (`importlib.metadata`, `pip freeze`, `conda list --explicit`) and the git and submodule commits. This catches a mismatch between the lock and what actually ran; no custom package tooling is involved.

### 4.2 Rules

- Pin exact versions (`==`) in lock files; ranges are allowed only in `requirements.in`.
- Never `pip install` an ad-hoc package into the shared environment. Add it to `requirements.in`, re-lock, and commit the input and lock files together.
- Install the conda layer first, then the pip layer. Never install the same package from both, and keep Python packages out of the conda layer (see the `uv pip sync` caution in §4.1).
- Upgrade one dependency at a time, in its own commit, with the test suite run before and after.
- Lock files are committed. Setup commands are reproducible from a clean checkout and documented in §6.

### 4.3 Why this approach, and alternatives

| Option | Pros | Cons |
|---|---|---|
| **conda + uv-compiled pip locks (candidate)** | Matches the upstream install method; native rendering libraries are reproducible on Linux and macOS; pip lock files with hashes are exact and fast to install | Two lock files to maintain |
| uv only (`uv.lock`) | One tool, one cross-platform lock, fastest | No management of native libraries (EGL/OSMesa, ffmpeg); these would be undocumented system installs |
| conda only (`environment.yml` + conda-lock) | One tool | Several packages (robosuite, robomimic, ReinFlow) are pip/git only, so pip ends up mixed in without a lock |
| Docker image | Strongest reproducibility on Linux | No GPU passthrough on macOS; x86 emulation on Apple Silicon is slow. Useful later as a release artifact for M5, not as the daily environment |

## 5. Open decisions

| ID | Decision | Options | Recommendation |
|---|---|---|---|
| D-1 | Python version | (a) 3.8, as tested upstream; (b) 3.10 | (a) for M0. Reproducing π₀ (FR-POL-01) is most reliable on the upstream-tested stack. Python 3.8 is past end-of-life, so a move to 3.10 is reconsidered after M0, once reproduction has passed, using that result as the regression check |
| D-2 | Observation modality for π₀ | (a) state (low-dimensional); (b) image | **Decided 2026-11-17: (a) state-based**, for both tracks. Reasons: lower complexity and compute; the research question (recovery vs retention) stays separate from perception; the Menagerie FR3 model has no camera. Consequences: Core needs a licensed state-based Can checkpoint, otherwise Path 2 (pretrain a state-based π₀). **2026-11-19:** the released Can flow checkpoint is image-based, and the lead decided to keep D-2 and follow Path 2 (design.md §10); policies receive exact object poses from the simulator, a stated limitation for any real-robot transfer (sim-to-real remains out of scope, requirements §3.2); videos need one observer camera only |
| D-3 | robomimic pin | PyPI `0.3.0` or a git commit | **Resolved at 0.2a:** v0.4.0 by commit `7c66e7a` (§2.2). PyPI 0.3.0 cannot create the Can environment without mujoco-py; the released Can flow checkpoint is image-based and not used under D-2, so no training version needs to be matched |

### 4.4 Generated files

All generated files (run outputs, logs, videos, checkpoints, banks) are written under `outputs/`, which is git-ignored. They are intended to be regenerable from the committed source, configs, lock files and recorded seeds, with limits: stochastic or nondeterministic computation (e.g. GPU kernels, multi-worker rollouts) may reproduce results only statistically rather than exactly (NFR-REP-02), and regeneration depends on external artifacts (checkpoints, datasets) remaining available under their recorded versions and hashes. Generated results that cannot be cheaply or exactly regenerated are therefore candidates for durable storage as described below. Results that must be kept and shared (FR-TRK-06 deliverables) are published as versioned artifacts outside the source tree, for example GitHub Release assets, with their hashes listed in the committed milestone report; the store is chosen when the first such deliverable exists.

## 6. Setup procedure

All steps are `make` targets, defined and commented in the [`Makefile`](../Makefile); `make help` lists them.

### 6.1 Files involved

| File | Role | Edited by hand? |
|---|---|---|
| `environment.yml` | Conda layer input: Python 3.8 and pip from conda-forge | Yes |
| `requirements.in` | Pip layer input: direct Python dependencies | Yes |
| `requirements-overrides.txt` | Pip layer input: corrections to what dependencies declare (e.g. `egl-probe` only on Linux) | Yes |
| `conda-lock.yml` | Conda layer lock, all platforms | No, generated by `make lock` |
| `requirements-macos.lock`, `requirements-linux.lock` | Pip layer locks, one per platform, with hashes | No, generated by `make lock` |
| `Makefile` | All setup, lock and test commands | Yes |

`make setup` **uses** the committed locks and never re-resolves versions, so every machine gets the same environment. `make lock` **creates** the locks and is only run when a dependency changes.

### 6.2 Fresh start on a new machine

1. **Install the tools** (once per machine):
   - conda: Miniconda or Miniforge
   - uv: e.g. `brew install uv` (macOS) or the official installer
   - conda-lock: `uv tool install conda-lock`
   - GNU make (preinstalled on macOS and most Linux distributions)
   - Linux only, system libraries (Ubuntu/Debian names): `sudo apt-get install build-essential cmake libosmesa6-dev libgl1 libglib2.0-0`. A C toolchain and CMake build `egl-probe` (a robomimic dependency without wheels); OSMesa renders headless on CPU; GLib is needed by OpenCV, which robosuite imports. These packages, and FFmpeg on macOS (Homebrew), are not pinned by the locks: a known reproducibility gap (NFR-REP-01; ARTIFACTS.md §5, G1 and G4)
2. **Get the code and build the environment:**

```bash
git clone --recurse-submodules https://github.com/hongbinyang/afterslip.git
cd afterslip
make check-tools    # confirms conda, uv and conda-lock are available
make platform       # shows the detected OS/CPU and the pip lock that will be used
make setup          # creates conda env "afterslip" from conda-lock.yml, installs the pip lock, installs the project
conda activate afterslip
make smoke          # quick check; expected today: 2 passed, 1 xfailed
make test           # full suite
```

On headless Linux, set the renderer before running tests: `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` (CPU), or `egl` for both with a GPU. `make lint` and `make typecheck` run the same checks as CI.

`make setup ENV=<name>` uses a different env name, e.g. for a clean trial next to an existing env. `make setup` stops if the env already exists.

**Platform detection:** the Makefile reads the operating system and CPU and selects the pip lock: macOS on Apple Silicon → `requirements-macos.lock`; Linux on x86-64 (including CI runners and WSL) → `requirements-linux.lock`. Other platforms (Intel Mac, Linux on ARM, Windows) stop with a message before changing anything. The conda layer is selected automatically by `conda-lock install`.

**PyTorch and NVIDIA libraries on Linux:** the Linux lock installs PyTorch's CUDA 12.1 build, which brings 12 NVIDIA CUDA library packages (`nvidia-*-cu12`) under NVIDIA's proprietary End User License Agreement. The project accepts these terms for its use: installing and using them, no redistribution, no safety-critical use (ARTIFACTS.md §5, G2). What this means in practice:

- **CPU-only Linux machines and CI** (tested): nothing extra is needed. The libraries are installed but unused, and PyTorch runs on the CPU.
- **Linux with an NVIDIA GPU** (untested): install the NVIDIA driver from the system's package manager or NVIDIA (a driver that supports CUDA 12.1; the CUDA toolkit itself is not needed, the pip packages provide it). Then check: `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
- **Redistributing an environment that contains these libraries** (e.g. a Docker image): check the agreement's distribution terms first; the license text is in each package's `*.dist-info/License.txt`.

### 6.3 Everyday tasks

| Situation | Commands |
|---|---|
| Pulled commits that changed the locks | `make sync` |
| Changed `environment.yml`, `requirements.in` or `requirements-overrides.txt` | `make lock`, review the diff, `make sync`, `make test`, then commit the inputs and **all** regenerated locks together |
| Environment broken or modified by hand | `conda deactivate`, `make remove-env`, `make setup` |
| Check what platform and lock apply | `make platform` |
| Pretrain or evaluate the Can policy | `make pretrain-can DEVICE=mps`, `make eval-can RUN=<pretraining run folder>`; see [run-cards.md](run-cards.md) |
| Prepare the Can demonstrations for pretraining (0.2b) | `make data-can` (downloads 47 MB once to `outputs/`, checks its sha256, regenerates the observations) |

### 6.4 Safety behaviour

- `make sync` previews the change first. It stops without changing anything if the preview fails, or if it would uninstall anything other than the packaging tools (pip, setuptools, wheel) and the project itself, which it reinstalls.
- `make setup` runs its steps strictly in order, also under parallel `make -j`.
- `make env`, `make sync` and `make install` first check that the upstream submodule is present; if it is missing they stop with `git submodule update --init` before changing any environment (checked 2026-11-19: the package list of an existing env is unchanged, and no env is created).
- `make remove-env` deletes only the named env, refuses `base` and the currently active env, and does nothing if the env does not exist.

### 6.5 Tested status

| Date | Platform | Tool versions | Result |
|---|---|---|---|
| 2026-11-18 | macOS 26.6, Apple Silicon | conda 26.3.2, uv 0.12.5, conda-lock 4.1.0, GNU make 3.81 | `make lock` succeeds for both platforms. Fresh `make setup` (also with `-j4`) succeeds; `make smoke`: 2 passed, 1 xfailed. Guards checked: failed preview stops; an extra package (`tabulate`) stops `make sync`; `remove-env` refuses `base`, the active env, and reports a missing env |
| 2026-11-19 | Linux x86-64: Ubuntu 22.04 container (`--platform linux/amd64`, emulated on Apple Silicon), CPU only | Miniforge (conda 26.9.4), uv 0.12.5, conda-lock 4.1.0 | Same steps as CI from a clean container: system libraries above, fresh `make setup` succeeds (egl-probe 1.0.2 builds from source), `make lint` and `make typecheck` pass, `make test` with `MUJOCO_GL=osmesa`: 22 passed, 1 xfailed. |
| 2026-11-19 | Linux x86-64: GitHub Actions `ubuntu-22.04` runner, CPU | Miniforge (latest), uv 0.12.5, conda-lock 4.1.0 | `.github/workflows/ci.yml` on commit `bc56825` | [Run 44641939078](https://github.com/hongbinyang/afterslip/actions/runs/44641939078): install from the locks, `make lint`, `make typecheck`, `make test` all pass |
| 2026-11-19 | macOS 26.6 Apple Silicon; Linux x86-64 (Ubuntu 22.04 container, emulated) | as above | 0.2a: fresh `make setup` with the ReinFlow submodule and robomimic v0.4.0 from git; `make lint`, `make typecheck`, `make test` (including T0.4, upstream Can environment) | **Pass** on both: 30 passed, 1 xfailed. The `make sync` guard was corrected to accept version changes (a removal plus an install of the same name) and still stops on a stray package (`tabulate`) |
| 2026-11-19 | macOS 26.6 Apple Silicon; Linux x86-64 (Ubuntu 22.04 container, emulated) | as above | 0.2b-1: `make data-can` (download, hash check, observation regeneration: 180 train / 20 validation demos) and `make test` | **Pass** on both: 40 passed, 1 xfailed. On headless Linux, `MUJOCO_GL=osmesa` must be set for `make data-can` as well as for the tests |

Tool versions are those observed on the tested machine; minimum supported tool versions are not established.

### 6.6 Measured compute (milestone 0.7)

Measured 2026-11-20 on an Apple M2 Pro (12 cores, 16 GB), macOS 26.6, torch 2.4.1, with the full regenerated Can data (180 demos, 20,343 training chunks, 79 batches of 256 per epoch). Scripts: `scripts/compute/`.

| Workload | Device | Measured | Full-run estimate |
|---|---|---|---|
| State-based 1-ReFlow pretraining (`configs/pretrain/can_reflow_mlp_state.yaml`) | CPU (8 torch threads) | 7.0 s per epoch (20 epochs in 144 s) | 2000 epochs ≈ 3.9 h |
| same | Apple MPS | 1.7 s per epoch after the first epoch (20 epochs in 37 s) | 2000 epochs ≈ 1 h |
| Can environment, random actions, no rendering | CPU, one environment | 165 control steps/s, including resets | about 1.7 h per million environment steps per process |

**Limits of these numbers:** the benchmark disables the held-out scoring and intermediate checkpoints (`val_dataset_path=null`, no checkpoint until the end), so the full-run estimates exclude that overhead: with the config's defaults, four checkpoint saves and one held-out scoring of the final checkpoint are added. A 20-epoch pilot says nothing about the final model's quality, and the exact duration of a 2000-epoch run is only known once one has finished (thermal throttling and other load on the machine can change it).

**Not measured:** CUDA GPUs; RL fine-tuning (policy updates plus many environments in parallel); rendering cost during training (none is planned for state-based training). The per-million-steps figure is for one process; how it scales with parallel environments on more cores is an assumption until measured in M1. Loss after 20 epochs was similar on both devices (about 0.39 on CPU, 0.42 on MPS); results are not bitwise identical across devices.

**Recommendation for the full pretraining (0.3d):** Apple MPS on this machine, about 1 hour, with no cloud GPU needed; CPU is a fallback at about 4 hours. Command: `make pretrain-can DEVICE=mps`; step-by-step run card: [run-cards.md](run-cards.md).

**Main guard:** any script that starts pretraining must run its work under `if __name__ == "__main__":`. On macOS the data-loader workers start new Python processes that re-import the script; without the guard, each worker starts another run and the job hangs (observed while measuring).

Environment variables expected by the upstream code (`REINFLOW_DIR`, `REINFLOW_DATA_DIR`, `REINFLOW_LOG_DIR`) are set by `afterslip.upstream.configure_reinflow()` instead of by hand (§2.2). An existing clone without the submodule needs `git submodule update --init`; `make install` stops with that hint.

## 7. Tested configurations

Filled in from actual runs (NFR-PORT-04). Each row records OS, hardware, backend, Python, torch, mujoco, robosuite and robomimic versions, which checks passed, the date, and links to the command and evidence.

| Date | OS / hardware | Backend | Key versions | Checks | Result |
|---|---|---|---|---|---|
| 2026-11-16 | macOS 26.6 / Apple Silicon (arm64) | CPU sim; CGL rendering; torch MPS available | conda-forge Python 3.8.20; mujoco 3.2.3; robosuite 1.4.1; robomimic 0.3.0; torch 2.4.1; torchvision 0.19.1; gym 0.22.0; numpy 1.24.4; hydra-core 1.3.2; imageio 2.35.1 | Create `PickPlaceCan` (Panda, OSC_POSE, 20 Hz, H=300); 60 random steps; offscreen `agentview` video 256×256; sim-state-only snapshot round-trip | Env and video **pass** (action dim 7). **Observed failure:** the snapshot round-trip that restores the simulator state only does **not** reproduce the continuation (max abs state difference ≈ 0.037, differing from the first replayed step). Cause not yet isolated; leading hypothesis: controller or other environment-internal state not captured by `sim.get_state()` (milestone item 0.4b). No π₀ loaded; no upstream ReinFlow code installed |
| 2026-11-16 | macOS 26.6 / Apple Silicon (arm64) | CPU sim; `mujoco.Renderer` (CGL) | Same env as above (mujoco 3.2.3); Menagerie commit `370c439` | FR3 track feasibility: load `franka_fr3/scene.xml`, list joints and actuators, step with position control, render 320×320 video | **Pass:** model loads (Menagerie states MuJoCo ≥ 3.1.3 required; 3.2.3 used); 7 arm joints, 7 position actuators, keyframe `home`; video renders. **Finding:** this model at this commit is arm only, with no gripper; grasping with it needs a hand model attached. Other FR3 model sources were not assessed. Spike only: no script committed, so the FR3 exercises stay open as hands-on learning |
| 2026-11-19 | Ubuntu 22.04 / x86-64 (container, emulated on Apple Silicon) | CPU sim; OSMesa rendering | Installed from `conda-lock.yml` and `requirements-linux.lock`: conda-forge Python 3.8.20; mujoco 3.2.3; robosuite 1.4.1; robomimic 0.3.0; torch 2.4.1; torchvision 0.19.1; gym 0.22.0; numpy 1.24.4; imageio 2.35.1 | Full test suite (`make test`): task adapter, observer recorder, snapshots (restore and replay identity), smoke tests | **Pass:** 22 passed, 1 xfailed (the simulator-state-only diagnostic, expected). Upright-frame check passes with OSMesa. Confirmed on native x86-64 by the first GitHub Actions run ([44641939078](https://github.com/hongbinyang/afterslip/actions/runs/44641939078)) |

**Reproducing the macOS check.** The checks are the smoke tests in [`tests/smoke/test_env.py`](../tests/smoke/test_env.py): `pytest -m smoke` in the `afterslip` env. Expected result on the tested configuration: 2 passed, 1 xfailed (the simulator-state-only snapshot round-trip, a known failure kept as a strict xfail: simulator state alone is not the full snapshot; the full snapshot passes since milestone item 0.4b). Generated files (videos, logs) go to pytest's temporary directory or `outputs/`, which is git-ignored; nothing generated is committed. The environment itself was built by hand, with the commands below; it is not yet reproducible from lock files:

```bash
conda create -n afterslip -c conda-forge --override-channels python=3.8 pip
# constraints.txt: pyobjc-core<11, pyobjc-framework-Quartz<11, pyobjc-framework-ApplicationServices<11, pyobjc-framework-Cocoa<11
pip install -c constraints.txt robosuite==1.4.1
pip install -c constraints.txt torch==2.4.1 torchvision==0.19.1
pip install --no-deps robomimic==0.3.0
pip install -c constraints.txt h5py psutil tqdm termcolor tensorboard tensorboardX imageio imageio-ffmpeg matplotlib gym==0.22.0 hydra-core==1.3.2 omegaconf==2.3.0 einops==0.8.0 pytest pytest-timeout hypothesis
```

macOS installation notes from this run (to be folded into the lock files):
- `pyobjc-core`/`pyobjc-framework-*` must be constrained to `< 11` on Python 3.8 (a robosuite → pynput dependency; newer releases do not build for 3.8).
- `robomimic==0.3.0` declares `egl-probe`, which does not build on macOS. It is installed with `--no-deps` and its remaining dependencies are installed explicitly. Linux installs keep `egl-probe`.
- robosuite warns about a missing private macro file; run `robosuite/scripts/setup_macros.py` once per environment.
- The conda env is named `afterslip`. An existing, unrelated `mujoco` env (Python 3.12, mujoco 3.11) is left untouched.
