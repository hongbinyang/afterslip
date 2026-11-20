# Artifact register

Every external artifact the project uses (code, checkpoint, dataset, robot model, scene asset) is recorded here with its source, exact version or content hash, license as stated by its own source, use and redistribution, and the date checked (requirements FR-ART-02, NFR-LIC-01, FR-FR3-01). An artifact whose license or access cannot be verified is not used. Checked by test-plan T0.3.

**Status (2026-11-19): incomplete; T0.3 not passed.** Some artifacts already in use are not yet fully verified or pinned; they are listed in §5. This is an open non-conformance with the rule above, not an accepted deviation: each gap is closed by verification, by replacing the artifact, or by a deviation that the project lead approves and that is then recorded here.

**Redistribution by this project:** the repository (MIT, [LICENSE](LICENSE)) contains only the project's own code and documents and currently redistributes none of the artifacts below; packages are installed by each user from PyPI, models and data are downloaded from their sources. This describes how the artifacts are used; it does not replace verifying each one. Any future redistribution (e.g. intervention-state banks, fine-tuned checkpoints) is checked against these terms first.

**Content hashes:** Python packages are pinned by version and by the sha256 hashes of their release files in `requirements-macos.lock` and `requirements-linux.lock`; the installer verifies these hashes on every install (`make setup`, `make sync`). Files outside the locks (checkpoints, datasets, models) get their own sha256 here.

**Keeping it current:** every package in the pip locks must appear in Appendix A at its locked version; `tests/unit/test_artifacts_register.py` fails otherwise, so a lock change and its register update land in the same commit.

## 1. Simulation, learning and recording code (in use)

The direct dependencies, with license evidence beyond the package metadata where needed. Every locked package, direct or transitive, is listed in Appendix A.

| Artifact | Source | Version / hash | License (as stated by source) | Use and redistribution | Checked |
|---|---|---|---|---|---|
| robosuite | [ARISE-Initiative/robosuite](https://github.com/ARISE-Initiative/robosuite), PyPI | 1.4.1; hashes in the pip locks | MIT (`LICENSE` at tag `v1.4.1`; the PyPI metadata states no license) | Benchmark environment `PickPlaceCan`, Panda robot, OSC controller; installed, not redistributed | 2026-11-19 |
| robomimic | [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic), PyPI | 0.3.0; hashes in the pip locks | MIT (`LICENSE` at tag `v0.3.0`; the PyPI metadata states no license) | Dependency of the upstream reference stack; installed, not redistributed | 2026-11-19 |
| MuJoCo Python bindings | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco), PyPI | 3.2.3; hashes in the pip locks | Apache-2.0 | Physics engine (FR-ENV-01); installed, not redistributed | 2026-11-19 |
| egl-probe | [StanfordVL/egl_probe](https://github.com/StanfordVL/egl_probe), PyPI (source only) | 1.0.2; hash in the Linux pip lock | MIT (repository `LICENSE`; no PyPI license metadata) | robomimic dependency, Linux only; built from source at install; not redistributed | 2026-11-19 |
| PyTorch, torchvision | [pytorch.org](https://pytorch.org/), PyPI | torch 2.4.1, torchvision 0.19.1; hashes in the pip locks | BSD-3 (torch), BSD (torchvision), package metadata. On Linux, torch pulls in 12 NVIDIA CUDA library packages under NVIDIA's proprietary license (§4, gap G2) | Learning; installed, not redistributed | 2026-11-19 |
| imageio, imageio-ffmpeg | [imageio](https://github.com/imageio), PyPI | 2.35.1, 0.5.1; hashes in the pip locks | BSD-2-Clause (both, package metadata). The FFmpeg program that does the encoding has its own license, GPL-3.0-or-later on both platforms (§4) | MP4 writing for observer videos; FFmpeg runs as a separate program; not redistributed | 2026-11-19 |

## 2. Robot models and scene assets

| Artifact | Source | Version / hash | License (as stated by source) | Use and redistribution | Status | Checked |
|---|---|---|---|---|---|---|
| Panda robot, Panda gripper, mount, bins arena, can and other object meshes and textures of the `PickPlaceCan` scene | Bundled in robosuite, `robosuite/models/assets/` | robosuite 1.4.1 | Repository-level MIT `LICENSE` at `v1.4.1`. **File level:** the scene loads 80 asset files (STL, OBJ, MSH, PNG) from `robots/panda/`, `grippers/meshes/panda_gripper/`, `mounts/meshes/rethink_mount/`, `objects/meshes/` and `textures/`; a scan of all 80 files and the 6 MJCF sources for license or copyright text found no license (the only matches: an empty `Copyright` field in `textures/steel-brushed.png`, and the copyright line of the embedded Adobe color profile in `textures/ceramic.png`, which concerns the profile, not the image). The assets tree has no license, notice or readme files. **Panda mesh origin:** 8 of the 13 loaded Panda STL files (`robots/panda/meshes/link0,2–7.stl`, `grippers/meshes/panda_gripper/hand.stl`) are byte-identical (same git blob hash) to the collision meshes in [frankaemika/franka_ros](https://github.com/frankaemika/franka_ros) `franka_description/meshes/collision/`, licensed Apache-2.0 (repository `LICENSE`); `link1.stl`, `finger.stl`, `finger_longer.stl`, the `*_vis.stl` files and the 50 `obj_meshes` files differ from those files and robosuite states no separate origin for them (gap G3) | Core task scene; used through robosuite, not redistributed. Rendered videos show the scene but do not contain the mesh files | **In use** | 2026-11-19 |
| Franka Research 3 model (`franka_fr3`) | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) | commit `370c43965323c0944b433b202aac5058b5f907a9` | Apache-2.0 (`franka_fr3/LICENSE`, per model; the repository has no single license) | Feasibility spike only (setup.md §7); arm only, no gripper | **Not in use**; FR3 track, F0 | 2026-11-19 |

## 3. Upstream code, checkpoints and datasets (not yet in use)

Each entry is completed, with file hashes, before first use.

| Artifact | Source | License (as stated by source) | Needed for | Status |
|---|---|---|---|---|
| ReinFlow reference code | [ReinFlow/ReinFlow](https://github.com/ReinFlow/ReinFlow) | MIT (repository `LICENSE`) | RL fine-tuning; pinned as a submodule at 0.2a | Commit not pinned yet (0.2a) |
| State-based Can checkpoint (π₀) | Upstream release linked from the ReinFlow project | To verify per file (the project page states MIT for its released materials; requirements background, ReinFlow entry) | Path 1 (FR-ART-01, §6.1.1) | Not identified yet (0.2b) |
| Robomimic Can demonstrations (low-dimensional) | robomimic dataset release | To verify | Path 2 (FR-POL-03), normalization statistics | Not downloaded (0.2b) |

## 4. System components outside the locks

Not installed by the project's pip locks; recorded with the evidence for their terms.

| Component | Source | Version | License (as stated by source) | Use | Checked |
|---|---|---|---|---|---|
| FFmpeg, Linux | Bundled in the locked imageio-ffmpeg 0.5.1 Linux wheel (`manylinux2010_x86_64`, sha256 `5289f75c…a316`, matching the Linux pip lock): executable `ffmpeg-linux64-v4.2.2` | 4.2.2 static build from johnvansickle.com; pinned through the wheel hash | GPL-3.0-or-later: the executable's own `-L` output; built with `--enable-gpl --enable-version3`, no `--enable-nonfree`; includes libx264, libx265, libxvid | Video encoding as a separate program; not redistributed | 2026-11-19 |
| FFmpeg, macOS | Homebrew formula `ffmpeg` (`/opt/homebrew/bin/ffmpeg`); imageio-ffmpeg has no macOS arm64 wheel, installs from source without a bundled executable, and finds FFmpeg on `PATH` | 7.1.1 on the tested machine; **not pinned** (gap G1) | GPL-3.0-or-later: Homebrew formula metadata and the executable's own `-L` output; built with `--enable-gpl --enable-version3`, includes libx264, libx265 | Video encoding as a separate program; not redistributed | 2026-11-19 |
| NVIDIA CUDA libraries (`nvidia-*-cu12`, 12 packages, Linux) | PyPI, installed as torch 2.4.1 dependencies; pinned and hash-verified in the Linux pip lock | versions in Appendix A | "NVIDIA Proprietary Software" (PyPI metadata). The NVIDIA End User License Agreement in the wheel (read in `nvidia-nvtx-cu12` 12.1.105, `License.txt`, wheel sha256 `dc21cf30…09e5`, matching the lock) grants a non-exclusive license to install and use the SDK; it forbids redistribution except for portions identified as distributable and incorporated into an application, forbids distribution as a stand-alone product, and excludes safety-critical use without an agreement with NVIDIA. The other 11 packages state the same license on PyPI; their bundled texts were not read one by one (gap G2) | GPU computation for torch; installed on every Linux setup, also CPU-only machines; not redistributed | 2026-11-19 |
| Mesa OSMesa (`libosmesa6-dev`) | Ubuntu 22.04 archive | archive version at install time; **not pinned** (gap G4) | MIT (Mesa project license) | Headless CPU rendering on Linux | 2026-11-19 |
| GLib (`libglib2.0-0`) | Ubuntu 22.04 archive | as above | LGPL-2.1-or-later | Runtime library of OpenCV, which robosuite imports | 2026-11-19 |
| C toolchain and CMake (`build-essential`, `cmake`) | Ubuntu 22.04 archive | as above | GPL (GCC) and BSD-3-Clause (CMake); build tools only, nothing of them is linked into project code | Building egl-probe at install | 2026-11-19 |

## 5. Open gaps

| # | Gap | Evidence so far | Recommendation | Needs |
|---|---|---|---|---|
| G1 | FFmpeg on macOS is an unpinned system program, and the two platforms use different FFmpeg versions (Linux 4.2.2, macOS 7.1.1) | Licenses verified (§4): GPL-3.0-or-later on both; use as a separate, unmodified program without redistribution is permitted by the GPL. **Tried 2026-11-19 (approved by the lead):** conda-forge `ffmpeg` 9.0.2 (GPL build) in the conda layer. It locks for both platforms (with a macOS 12 minimum), but on macOS it adds conda-forge `libcxx` 23 to the environment, after which the pip-installed `llvmlite` (numba) fails to load (missing C++ symbol); no older `libcxx` is compatible with that FFmpeg. Not adopted | Interim (in place): each video's metadata records the FFmpeg version that encoded it (`encoder`). Next: assess PyAV (`av`, pip-locked with hashes on both platforms; the upstream code already pins `av==12.3.0`) as the single encoder | Spike and then lead decision |
| G2 | NVIDIA CUDA libraries are proprietary | EULA read for one package (§4): install and use are granted; redistribution is restricted | **Decided 2026-11-19 by the project lead:** use accepted under the EULA (no redistribution, no safety-critical use); the CUDA build stays, since GPU training is planned. Linux usage notes in setup.md §6.2. Before any release that bundles the environment (e.g. a Docker image at M5), the distribution terms are checked | Closed for use; re-check before redistribution |
| G3 | Panda meshes that are not byte-identical to franka_ros (link1, fingers, visual meshes) have no stated origin or terms of their own | 8 of 13 loaded STL files traced to franka_ros (Apache-2.0); the rest are covered only by robosuite's repository-level MIT license | Accept robosuite's MIT license as the stated terms for use; trace the remaining files before any redistribution of scene files | Product review |
| G4 | System packages (Ubuntu archive, Homebrew) are not pinned | Listed in §4 and setup.md §6.2 | Record their installed versions in each run manifest (FR-TRK-01, M1); pin FFmpeg per G1 | Implementation at M1 |

## Appendix A. Every locked Python package

All packages pinned in `requirements-linux.lock` and `requirements-macos.lock`, with the license as stated in the package's own PyPI metadata (SPDX expression, license field, or license classifier, in that order of preference), or by its repository where the metadata states none. Source: `https://pypi.org/project/<name>/<version>/`; hashes in the lock files. Checked 2026-11-19. None is redistributed by this project.

| Package | Version | Platforms | License (as stated by source) | Note |
|---|---|---|---|---|
| absl-py | 2.3.1 | Linux, macOS | Apache-2.0 |  |
| antlr4-python3-runtime | 4.9.3 | Linux, macOS | BSD |  |
| attrs | 25.3.0 | Linux, macOS | MIT |  |
| certifi | 2026.10.23 | Linux, macOS | MPL-2.0 |  |
| cffi | 1.17.1 | Linux, macOS | MIT |  |
| charset-normalizer | 3.5.2 | Linux, macOS | MIT |  |
| cloudpickle | 3.1.3 | Linux, macOS | BSD-3-Clause |  |
| contourpy | 1.1.1 | Linux, macOS | BSD License (classifier) |  |
| cryptography | 47.0.0 | Linux, macOS | Apache-2.0 OR BSD-3-Clause |  |
| cycler | 0.12.1 | Linux, macOS | BSD License (classifier) |  |
| egl-probe | 1.0.2 | Linux | MIT (repository LICENSE; no PyPI license metadata) |  |
| einops | 0.8.0 | Linux, macOS | MIT |  |
| etils | 1.3.0 | Linux, macOS | Apache Software License (classifier) |  |
| evdev | 1.9.2 | Linux | BSD License (classifier) |  |
| exceptiongroup | 1.3.1 | Linux, macOS | MIT License (classifier) |  |
| filelock | 3.16.1 | Linux, macOS | The Unlicense (Unlicense) (classifier) |  |
| fonttools | 4.57.0 | Linux, macOS | MIT |  |
| fsspec | 2025.3.0 | Linux, macOS | BSD 3-Clause License |  |
| glfw | 2.11.0 | Linux, macOS | MIT |  |
| google-auth | 2.50.0 | Linux, macOS | Apache 2.0 |  |
| google-auth-oauthlib | 1.0.0 | Linux, macOS | Apache 2.0 |  |
| grpcio | 1.70.0 | Linux, macOS | Apache License 2.0 |  |
| gym | 0.22.0 | Linux, macOS | MIT |  |
| gym-notices | 0.1.0 | Linux, macOS | MIT License (classifier) |  |
| h5py | 3.11.0 | Linux, macOS | BSD-3-Clause |  |
| hydra-core | 1.3.2 | Linux, macOS | MIT |  |
| hypothesis | 6.113.0 | Linux, macOS | MPL-2.0 |  |
| idna | 3.15 | Linux, macOS | BSD-3-Clause |  |
| imageio | 2.35.1 | Linux, macOS | BSD-2-Clause |  |
| imageio-ffmpeg | 0.5.1 | Linux, macOS | BSD-2-Clause |  |
| importlib-metadata | 8.5.0 | Linux, macOS | Apache Software License (classifier) |  |
| importlib-resources | 6.4.5 | Linux, macOS | Apache Software License (classifier) |  |
| iniconfig | 2.1.0 | Linux, macOS | MIT |  |
| jinja2 | 3.1.6 | Linux, macOS | BSD License (classifier) |  |
| kiwisolver | 1.4.7 | Linux, macOS | BSD License (classifier) |  |
| llvmlite | 0.41.1 | Linux, macOS | BSD |  |
| markdown | 3.7 | Linux, macOS | BSD License (classifier) |  |
| markupsafe | 2.1.5 | Linux, macOS | BSD-3-Clause |  |
| matplotlib | 3.7.5 | Linux, macOS | PSF |  |
| mpmath | 1.3.0 | Linux, macOS | BSD |  |
| mujoco | 3.2.3 | Linux, macOS | Apache License 2.0 |  |
| mypy | 1.14.1 | Linux, macOS | MIT |  |
| mypy-extensions | 1.1.0 | Linux, macOS | MIT (repository LICENSE; no PyPI license metadata) |  |
| networkx | 3.1 | Linux, macOS | BSD License (classifier) |  |
| numba | 0.58.1 | Linux, macOS | BSD |  |
| numpy | 1.24.4 | Linux, macOS | BSD-3-Clause |  |
| nvidia-cublas-cu12 | 12.1.3.1 | Linux | NVIDIA Proprietary Software | CUDA runtime pulled in by torch on Linux; proprietary terms **not yet reviewed** |
| nvidia-cuda-cupti-cu12 | 12.1.105 | Linux | NVIDIA Proprietary Software | CUDA runtime pulled in by torch on Linux; proprietary terms **not yet reviewed** |
| nvidia-cuda-nvrtc-cu12 | 12.1.105 | Linux | NVIDIA Proprietary Software | CUDA runtime pulled in by torch on Linux; proprietary terms **not yet reviewed** |
| nvidia-cuda-runtime-cu12 | 12.1.105 | Linux | NVIDIA Proprietary Software | CUDA runtime pulled in by torch on Linux; proprietary terms **not yet reviewed** |
| nvidia-cudnn-cu12 | 9.1.0.70 | Linux | NVIDIA Proprietary Software | CUDA runtime pulled in by torch on Linux; proprietary terms **not yet reviewed** |
| nvidia-cufft-cu12 | 11.0.2.54 | Linux | NVIDIA Proprietary Software | CUDA runtime pulled in by torch on Linux; proprietary terms **not yet reviewed** |
| nvidia-curand-cu12 | 10.3.2.106 | Linux | NVIDIA Proprietary Software | CUDA runtime pulled in by torch on Linux; proprietary terms **not yet reviewed** |
| nvidia-cusolver-cu12 | 11.4.5.107 | Linux | NVIDIA Proprietary Software | CUDA runtime pulled in by torch on Linux; proprietary terms **not yet reviewed** |
| nvidia-cusparse-cu12 | 12.1.0.106 | Linux | NVIDIA Proprietary Software | CUDA runtime pulled in by torch on Linux; proprietary terms **not yet reviewed** |
| nvidia-nccl-cu12 | 2.20.5 | Linux | NVIDIA Proprietary Software | CUDA runtime pulled in by torch on Linux; proprietary terms **not yet reviewed** |
| nvidia-nvjitlink-cu12 | 12.9.86 | Linux | LicenseRef-NVIDIA-Proprietary | CUDA runtime pulled in by torch on Linux; proprietary terms **not yet reviewed** |
| nvidia-nvtx-cu12 | 12.1.105 | Linux | NVIDIA Proprietary Software | CUDA runtime pulled in by torch on Linux; proprietary terms **not yet reviewed** |
| oauthlib | 3.3.1 | Linux, macOS | BSD-3-Clause |  |
| omegaconf | 2.3.0 | Linux, macOS | BSD License (classifier) |  |
| opencv-python | 5.1.0.95 | Linux, macOS | Apache 2.0 |  |
| packaging | 26.2 | Linux, macOS | Apache-2.0 OR BSD-2-Clause |  |
| pillow | 10.4.0 | Linux, macOS | HPND |  |
| pluggy | 1.5.0 | Linux, macOS | MIT |  |
| protobuf | 5.29.6 | Linux, macOS | 3-Clause BSD License |  |
| psutil | 7.2.2 | Linux, macOS | BSD-3-Clause |  |
| pyasn1 | 0.6.5 | Linux, macOS | BSD-2-Clause |  |
| pyasn1-modules | 0.4.2 | Linux, macOS | BSD |  |
| pycparser | 2.23 | Linux, macOS | BSD-3-Clause |  |
| pynput | 1.8.2 | Linux, macOS | LGPLv3 | LGPL; used as an unmodified installed library |
| pyobjc-core | 10.3.2 | macOS | MIT License |  |
| pyobjc-framework-applicationservices | 10.3.2 | macOS | MIT License |  |
| pyobjc-framework-cocoa | 10.3.2 | macOS | MIT License |  |
| pyobjc-framework-coretext | 10.3.2 | macOS | MIT License |  |
| pyobjc-framework-quartz | 10.3.2 | macOS | MIT License |  |
| pyopengl | 3.1.10 | Linux, macOS | BSD License (classifier) |  |
| pyparsing | 3.1.4 | Linux, macOS | MIT License (classifier) |  |
| pytest | 8.3.5 | Linux, macOS | MIT |  |
| pytest-timeout | 2.4.0 | Linux, macOS | MIT |  |
| python-dateutil | 3.0.0 | Linux, macOS | Dual License |  |
| python-xlib | 0.33 | Linux | LGPLv2+ | LGPL; used as an unmodified installed library |
| pyyaml | 6.0.3 | Linux, macOS | MIT |  |
| requests | 2.32.4 | Linux, macOS | Apache-2.0 |  |
| requests-oauthlib | 2.0.0 | Linux, macOS | ISC |  |
| robomimic | 0.3.0 | Linux, macOS | MIT (repository LICENSE at v0.3.0; no PyPI license metadata) |  |
| robosuite | 1.4.1 | Linux, macOS | MIT (repository LICENSE at v1.4.1; no PyPI license metadata) |  |
| ruff | 0.17.5 | Linux, macOS | MIT |  |
| scipy | 1.10.1 | Linux, macOS | BSD License (classifier) |  |
| setuptools | 75.3.4 | Linux, macOS | MIT License (classifier) |  |
| six | 1.17.0 | Linux, macOS | MIT |  |
| sortedcontainers | 2.4.0 | Linux, macOS | Apache 2.0 |  |
| sympy | 1.13.3 | Linux, macOS | BSD |  |
| tensorboard | 2.14.0 | Linux, macOS | Apache 2.0 |  |
| tensorboard-data-server | 0.7.2 | Linux, macOS | Apache 2.0 |  |
| tensorboardx | 2.6.2.2 | Linux, macOS | MIT license |  |
| termcolor | 2.4.0 | Linux, macOS | MIT |  |
| tomli | 2.5.0 | Linux, macOS | MIT |  |
| torch | 2.4.1 | Linux, macOS | BSD-3 |  |
| torchvision | 0.19.1 | Linux, macOS | BSD |  |
| tqdm | 4.70.1 | Linux, macOS | MPL-2.0 AND MIT |  |
| triton | 3.0.0 | Linux | MIT License (classifier) |  |
| typing-extensions | 4.13.2 | Linux, macOS | PSF-2.0 |  |
| tzdata | 2026.6 | Linux, macOS | Apache-2.0 |  |
| urllib3 | 2.2.3 | Linux, macOS | MIT License (classifier) |  |
| werkzeug | 3.0.6 | Linux, macOS | BSD License (classifier) |  |
| wheel | 0.45.1 | Linux, macOS | MIT License (classifier) |  |
| zipp | 3.20.2 | Linux, macOS | MIT License (classifier) |  |
