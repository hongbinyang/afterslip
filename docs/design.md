# Technical design

Status: draft, M0 scope. This document describes the architecture, interfaces and technical decisions that implement the [requirements](requirements.md) (v1.1 draft). Work items and checks are in [milestones.md](milestones.md) and [test-plan.md](test-plan.md); environment and versions are in [setup.md](setup.md). Later milestones add sections (disturbances and bank in M2, training and mitigations in M1–M3) when their design is needed; the interfaces below are chosen so those additions do not change them.

**Markers:** **[unverified]** = depends on an artifact, version or behaviour not yet checked; **[decided]** = recorded decision with its date; **[open]** = a decision point listed in milestones.md.

---

## 1. Design principles

1. **Standard tools first.** Packaging with `pyproject.toml`; dependencies from lock files; tests with pytest; configuration with Hydra/OmegaConf (as upstream); CI with GitHub Actions. No custom mechanism where a mature one exists.
2. **Generated files never enter the repository.** Every run writes under `outputs/` (git-ignored). Tests write to pytest's temporary directories.
3. **Upstream code is pinned, not edited.** The reference RL implementation is a git submodule at a fixed commit; local changes are patch files (NFR-COD-02).
4. **Robot- and task-specific code sits behind one adapter.** Research machinery (snapshots, disturbances, banks, evaluation, recording) talks only to the task interface, so the FR3 track adds an adapter, not a rewrite.
5. **State-based policies, one observer camera [decided 2026-11-17, D-2, FR-POL-04].** No camera is a policy input. One fixed third-person camera records videos and never feeds the policy. Which camera is used, and the tooling choices in this document (CI service, lock tools, configuration system), are candidates until tested and recorded in setup.md.
6. **Reproducibility by construction.** A run is specified by its seeds, config, git and submodule commits, artifact hashes and environment lock files; its manifest records the actual runtime environment (FR-TRK-01). On the same backend this aims at exact repetition; across backends it does not, and reproducibility is judged by the requirements' predeclared statistical comparison (NFR-REP-02, NFR-REP-03), not bitwise identity.

## 2. Repository layout (target for M0)

```
afterslip/                     Python package (src layout below is equivalent; flat chosen for simplicity)
  __init__.py
  tasks/
    interface.py               TaskInterface protocol, Snapshot type (§3, §4)
    robosuite_can.py           Core adapter: robosuite PickPlaceCan, Panda, OSC_POSE
  recording/
    recorder.py                ObserverRecorder: offscreen frames -> MP4 + metadata (§5)
  policies/
    base.py                    Policy protocol, reset/reconstruction rule (§6)
  eval/
    episodes.py                episode runner, per-episode JSONL records (§7)
  runtime.py                   run manifest: commit, config, versions, hardware (§8)
configs/                       Hydra configs (task, recorder, eval)
tests/
  conftest.py                  shared fixtures: seeds, environment factory, generated fixture bank (§9)
  unit/  integration/  smoke/
third_party/
  ReinFlow/                    git submodule, pinned commit [to add at 0.2a]
  patches/                     reviewable local patches, if any
outputs/                       git-ignored: runs, videos, logs, manifests
pyproject.toml                 package metadata, tool configuration (pytest, ruff, mypy)
environment.yml, conda-lock.yml, requirements-*.lock    environment definition (setup.md §4)
```

`pytest.ini` (current) moves into `pyproject.toml` under `[tool.pytest.ini_options]` when the package is created (0.0b).

## 3. Task interface (FR-ENV-03, -05; milestone 0.2c)

### 3.1 Purpose

A Gymnasium-style wrapper that is the **only** way training, evaluation, snapshot and recording code touch the simulator. It hides robosuite (Core) or plain MuJoCo (FR3) behind one contract.

### 3.2 Contract

```python
class TaskInterface(Protocol):
    horizon: int                         # H, from the task's standard horizon (FR-ENV-05)
    action_dim: int                      # 7 for Can with OSC_POSE (6 pose deltas + gripper) [unverified for π₀]
    control_freq_hz: float               # 20 for robosuite Can

    def reset(self, seed: int) -> Observation: ...
    def step(self, action: np.ndarray) -> StepResult: ...
    def get_state(self) -> Snapshot: ...          # §4
    def set_state(self, snapshot: Snapshot) -> Observation: ...
    def render(self, camera: str, width: int, height: int) -> np.ndarray: ...   # §5; must not change state

@dataclass(frozen=True)
class StepResult:
    observation: Observation             # dict[str, np.ndarray], low-dimensional state only (FR-POL-04)
    reward: float                        # benchmark sparse success reward (FR-RL-02)
    success: bool                        # benchmark's own success predicate, every step (FR-ENV-05)
    timeout: bool                        # step count reached H without success
    info: dict
```

Decisions inside the contract:

| Topic | Decision | Reason |
|---|---|---|
| Episode end | Candidate: end on `success` or `timeout`; reaching H without success is a failure (requirements §4). Whether the benchmark and the released checkpoint's evaluation also end early on success, or run to H, is **[unverified]** and is checked at 0.2b; the interface follows the benchmark's and checkpoint's convention, recorded in the config mapping | Success rates must be comparable with the released figure (FR-POL-01) |
| Observation | Dict of named low-dimensional arrays; key names and order come from the π₀ config (FR-ART-03) **[unverified until 0.2b]** | The policy adapter maps keys; the task never reorders |
| Seeding | `reset(seed)` seeds every random source the task uses, including robosuite's global-NumPy placement sampler | Seeded resets are the basis of fixed test seeds; robosuite uses `np.random` globally, a known pitfall |
| Rendering | `render` reads the simulator only | Recording must not change the trajectory (§5) |
| Gym vs Gymnasium | Candidate: our interface is Gymnasium-style; upstream uses `gym==0.22` and its own Robomimic environment wrapper. Whether a thin adapter can reproduce the upstream wrapper's behaviour (observation processing, action scaling, episode end) is **[unverified]**; equivalence is tested against the released checkpoint at 0.2b/0.3c before our interface is used for evaluation | Avoid editing upstream code without assuming equivalence |

### 3.3 Core adapter: `RobosuiteCanTask`

Wraps `robosuite.make("PickPlaceCan", robots="Panda", controller_configs=OSC_POSE, control_freq=20, horizon=300, use_camera_obs=False, has_offscreen_renderer=True)`. Values come from config and must match the π₀ config at 0.2b. Known so far (setup.md §7): action dim 7, observation keys include `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `Can_pos`, `Can_quat`.

### 3.4 FR3 adapter (F0/F1, later)

Plain MuJoCo (`mujoco` bindings), a project scene file that includes the Menagerie FR3 model unmodified, adds one fixed observer camera, and later a hand at `attachment_site` and the sorting scene. Same `TaskInterface`; no robosuite dependency.

## 4. Snapshot (FR-ENV-04, NFR-REP-03; milestones 0.4b, 2.1)

### 4.1 Requirement recap

A snapshot must restore the environment so that replaying the same actions reproduces the same trajectory exactly on the same backend. It contains environment state and the pre-intervention observation/action history, and **never** policy-internal state.

### 4.2 Evidence so far

The smoke test `test_sim_state_only_snapshot_round_trip` shows that restoring robosuite's `sim.get_state()` (time, qpos, qvel only) does **not** reproduce the continuation: the difference appears from the first replayed step (≈ 0.037). The cause is not yet isolated (0.4b).

### 4.3 Candidate state components (to be confirmed by the 0.4b spike)

| Group | Candidate contents | Why it may matter |
|---|---|---|
| Physics | `time`, `qpos`, `qvel`, `act`, `ctrl`, warm-start accelerations | `MjSimState` in robosuite 1.4.1 holds only time, qpos and qvel |
| Controller | operational-space controller goal pose and orientation reference, interpolator state, gripper controller state | robosuite controllers keep goals between steps outside the physics state |
| Environment | step counter, elapsed time, done flag, task flags (object placed) | Affects horizon and success bookkeeping |
| Randomness | environment RNG state(s), including the global NumPy RNG if used during stepping | Noise or sampling after restore |
| History | last k observations and actions (k from the π₀ config; `cond_steps` 1 upstream suggests k = 1) **[unverified]** | FR-DIS-10 policy re-initialisation |
| Metadata | schema version, task name, source seed, step index, creation commit | Validation and provenance |

FR-ENV-04 already **requires** the snapshot to hold simulator state, task state, controller internal state, environment RNG states, intervention metadata and the last-k observation and action history. The schema therefore keeps all of these groups regardless of the spike's outcome. The spike determines **how** each group is captured and restored in robosuite (which attributes, in what order, and what must be re-derived after restoring), and whether further components (e.g. warm-start accelerations) are also needed for exact repetition. **The 0.4b exercise is a hands-on learning task**: the table above is the search space for the capture method, not the answer.

Two properties are tested separately:
1. **Restore identity:** immediately after `set_state(s)`, `get_state()` returns a snapshot equal to `s` in every required field.
2. **Replay identity:** from a restored snapshot, the same actions reproduce the original continuation exactly on the same backend.

### 4.4 Format

- In-memory (increment D2, milestone 0.4b): the frozen dataclass `Snapshot` in `afterslip/tasks/interface.py`, with the groups `physics`, `task`, `controller`, `rng`, `obs_history`, `action_history` and `metadata` (`schema_version`, task, seed, step, `intervention`). A snapshot is an independent copy of the environment state.
- **History semantics:** only what happened in the current episode, never padded. After reset: the reset observation and no actions. After n steps: `min(k, n + 1)` observations and `min(k, n)` actions, oldest first; `action_history[-1]` produced `obs_history[-1]`.
- **Intervention metadata:** `metadata["intervention"]` is `None` until the disturbance code calls `record_intervention(...)` in an episode; it is then stored by `get_state`, cleared by `reset` and restored by `set_state`.
- **Scope of D2:** in-memory snapshots on one backend, verified by restore identity and by replay identity of observations and hidden state (qpos, qvel, controller goal) at every step (test-plan T0.11). Serialization to disk and hashing follow in M2 (milestone 2.1, test-plan T2.1).
- **Open limitation:** robosuite draws object placements from NumPy's global RNG, which every environment in a process shares. Saving and restoring it is exact for one environment; with several environments in one process (e.g. vectorized rollouts, FR-ENV-06) their draws interleave. Untested; to be resolved before vectorized bank generation (M2), e.g. one environment per process.
- On disk (banks, M2): one compressed `.npz` per snapshot or one per bank file, plus a JSON manifest. Hashes are computed over a canonical serialization of the arrays (sorted keys, fixed dtype and byte order), not over file bytes, so compression changes do not change identity (test-plan T2.4).
- No pickle, so files are portable and safe to load.

## 5. Observer recording (FR-EVAL-06; milestone 0.3e)

### 5.1 Contract

```python
class ObserverRecorder:
    def __init__(self, camera: str, width: int, height: int, fps: float, out_dir: Path): ...
    def start(self, episode_meta: EpisodeMeta) -> None: ...
    def capture(self, task: TaskInterface) -> None: ...       # one frame per control step
    def finish(self) -> VideoRecord: ...                       # writes MP4 + JSON sidecar
```

### 5.2 Decisions

| Topic | Decision |
|---|---|
| Camera | One fixed observer camera **[decided D-2]**. Candidate for Core: robosuite `agentview`; FR3: a fixed camera defined in the project scene file. Final camera name and pose recorded in config and metadata |
| Frame rate | One frame per control step, fps = control frequency (20) |
| Resolution | 256×256 routine evidence; 512×512 demos (config) |
| Encoding | imageio + imageio-ffmpeg, H.264 MP4 |
| Orientation | Frames from robosuite 1.4.1 `MjSim.render` were vertically flipped on macOS; the adapter returns upright frames, so the recorder never flips **[verify per platform]** |
| Side effects | Rendering must not change simulator state or any RNG; a test compares trajectories with and without recording |
| Metadata sidecar | episode or bank-state ID, seed, policy checkpoint hash, suite, camera name and pose, fps, resolution, backend, frame count, episode length, git commit, encoder (FFmpeg version actually used) |
| Location | `outputs/<run>/videos/<episode_id>.mp4` and `.json` |
| Cost | Rendering lowers throughput, so training does not record; evaluation records only the configured subset |

## 6. Policy interface (FR-ART-03, FR-DIS-10, FR-POL-04)

```python
class Policy(Protocol):
    def reset(self) -> None: ...                               # clear internal state and pending action chunk
    def act(self, observation: Observation) -> np.ndarray: ... # returns one action for the current step
    def warm_start(self, obs_history: list[Observation], act_history: list[np.ndarray]) -> None: ...
```

- The π₀ adapter wraps the upstream model: input key order and normalization from the released config; action chunking handled inside the adapter (execute `act_steps` actions per chunk, replan afterwards).
- **Reconstruction rule (FR-DIS-10):** at a disturbed-episode start, `reset()`, then `warm_start()` with the stored history, discard any pending chunk, replan.
- Toy policies (constant, seeded random) implement the same protocol for harness tests.
- Loading checks (test T0.5): observation keys and order, normalization shapes, action dimension, chunk length, history steps, denoising steps, and no image inputs.

## 7. Episode records (FR-EVAL-04 subset; milestone 0.3b)

One JSON line per episode under `outputs/<run>/episodes.jsonl`:

`episode_id, suite, seed, bank_state_id (null for nominal), policy_hash, success, steps, timeout, video (path or null), git_commit`

Fields for disturbances and re-grasps are added in M2 without renaming these.

## 8. Run manifest and outputs (FR-TRK-01, -02)

Each run directory `outputs/<YYYYMMDD-HHMMSS>-<name>/` contains the resolved config, `manifest.json` (git commit and dirty flag, submodule commit, platform, CPU/GPU, backend, Python and key package versions via `importlib.metadata`, start and end time), logs, episode records and videos. The manifest is written with standard library tools only.

## 9. Testing architecture

| Topic | Decision |
|---|---|
| Runner | pytest, markers `unit`, `integration`, `smoke`; strict xfail |
| Missing dependencies | Never skipped: a broken environment is a failure |
| Fixtures (`tests/conftest.py`) | fixture seeds from `fixtures/env_seeds.yaml` (committed input); a session-scoped **generated fixture bank** built from those seeds into a temporary directory by the same code as real banks; toy policies |
| Data separation | Unit, integration and smoke tests and their fixtures never read validation or test seeds or banks. Protocol evaluations defined in the test plan (e.g. T0.8, T2.11, T3.3) read validation or test data exactly as the test plan specifies, and test data only after the design lock |
| CI | Candidate: GitHub Actions running lint, type check and unit tests on every push, and integration and smoke tests on Linux CPU (NFR-PORT-03) **[open, 0.0c]** |
| Determinism tests | snapshot round trip (T0.11), rendering has no side effects (§5.2), seeded reset gives identical initial state |

## 10. Artifact and environment verification (milestones 0.1, 0.2a, 0.2b)

1. **Upstream code (0.2a):** ReinFlow is a git submodule at commit `e722e15`, installed without its dependencies; the versions it requires and our deviations are in setup.md §2.2. **Verified:** the Robomimic Can path does not need `mujoco-py` with robomimic v0.4.0 (PyPI 0.3.0 does need it). Upstream modules are imported only after `afterslip.upstream.configure_reinflow()` (T0.4).
2. **State-based checkpoint (0.2b) [decided D-2]:** find the state-based Can configuration and checkpoint in the upstream release; record source URL, file hash and license in `ARTIFACTS.md`. Only image-based Can configs have been read so far **[unverified]**. The §6.1.1 decision table applies as written: **Path 1** needs a licensed state-based checkpoint whose reproduction passes; **Path 2** (pretrain a state-based π₀) needs **verified, licensed demonstrations** with low-dimensional observations, and its feasibility (configuration, compute) is assessed rather than assumed; if no licensed demonstrations exist, **Path 3** applies: work stops and a re-scope decision is requested. A missing checkpoint alone does not establish that Path 2 is feasible.
   **Finding (2026-11-19):** upstream has no state-based 1-ReFlow configuration for Can, and the released Can checkpoint is image-based (ARTIFACTS.md §3). Under D-2, Path 1 is therefore unavailable from this source. **Decided 2026-11-19 by the project lead: keep D-2 and follow Path 2**, pretraining a state-based 1-ReFlow π₀ with the upstream pretraining code on licensed low-dimensional Can demonstrations (FR-POL-03), subject to verifying those demonstrations (0.2b).
   **Demonstrations (decided 2026-11-19 by the project lead, product manager and architect, conditionally):** the robomimic v1.5 proficient-human Can file (MIT; recorded with robosuite 1.5.1) is used in robosuite 1.4.1. Only its recorded states and actions are used: robosuite 1.5.1 orders and defines the 14-D `object` observation differently, so every observation is regenerated from the states in our environment (`afterslip/data/can_demos.py`, `make data-can`); normalization statistics come from robomimic's `train` mask only, and its `valid` demos are held out. Conditions, each a committed check: the file's hash; regenerated robot observations equal the recorded ones (≤ 1e-6); the observation order equals the live environment (within the one-substep kinematic lag of a step's observation, about 1e-5); actions aligned with observations; and the open-loop replay comparison (`scripts/demo_replay/`: 193/200 in robosuite 1.4.1, 187–189/200 in robosuite 1.5.1, with similar end-effector drift). These do not establish equal fidelity; the decisive check is the pretrained policy's success under FR-POL-03. If that fails, migrating to robosuite 1.5.1 is the fallback.
3. **Config mapping (0.2b):** a table from the released config to our task settings (observation keys, normalization file, action dimension, `horizon_steps`, `act_steps`, `cond_steps`, denoising steps, control frequency, horizon). A test asserts each value.
   **State-based config (0.2b, Path 2):** `configs/pretrain/can_reflow_mlp_state.yaml`, run with `make pretrain-can` (`afterslip.upstream.pretrain`). Each value is asserted by `tests/integration/test_pretrain_config.py`:

   | Parameter | Value | Source |
   |---|---|---|
   | Observation | 23-D: `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `object` (regenerated, robosuite 1.4.1) | upstream state-based Can configs (`obs_dim` 23); `afterslip/data/can_demos.py` |
   | Normalization | min–max to [-1, 1], train demos only (`normalization.npz`) | upstream converter's mapping; split from robomimic masks |
   | Action | 7-D OSC_POSE delta + gripper, normalized to [-1, 1] | task; `can.json` controller |
   | `horizon_steps` / `act_steps` | 4 / 4 | upstream Can image ReFlow and state diffusion configs; fine-tuning config |
   | `cond_steps` | 1 (no observation history) | upstream Can configs |
   | Model | ReFlow with `FlowMLP`, ReLU, residual, no layer norm | upstream state ReFlow config (Robomimic Transport) |
   | Width / time embedding | `mlp_dims` [512, 512, 512], `time_dim` 32 | upstream Can image ReFlow config (Transport uses 1024 for a larger task) |
   | Denoising steps (max) | 100 | upstream Can ReFlow configs |
   | Optimizer | lr 1e-4, weight decay 1e-6, cosine schedule (2000 steps, warm-up 100, min 1e-5), EMA 0.995, 2000 epochs, batch 256 | upstream Can image ReFlow config |
   | Environment | PickPlaceCan, Panda, OSC_POSE, 20 Hz, horizon 300 | `can.json`, fine-tuning config, `CanTaskConfig` |
   | Logging | local only (`wandb: null`); no evaluation during pretraining (`test_in_mujoco: false`) | FR-TRK-02; evaluation follows the FR-POL-03 protocol |
   | Held-out validation | robomimic's 20 `valid` demos (`val.npz`), scored after training by `afterslip.upstream.pretrain`: mean flow loss of the final checkpoint (model and EMA weights; 5 passes, fixed seed), written to `<logdir>/heldout_validation.json` | upstream's validation is not used: it only splits the training file at random (`train.train_split`), and its code path fails at the pinned commit (`with torch.no_grad:` without a call) |

   Measured on CPU (macOS arm64): about 0.5 s per batch of 64, so a full 2000-epoch run is a GPU job (estimate at 0.7).

   **Use of the held-out loss before policy evaluation:** it is a diagnostic reported with each pretrained checkpoint, next to the training loss; a held-out loss far above the training loss indicates overfitting to the 180 training demos and is raised before the policy is evaluated. It does not select the policy: the checkpoint to evaluate (the final one, unless the FR-POL-03 protocol fixed in 0.3a says otherwise) and the success threshold are fixed before any evaluation run, so evaluation results never feed back into the choice.
4. **Licenses (0.1):** robosuite, robomimic and their datasets, ReinFlow code and checkpoints, Menagerie FR3 (Apache-2.0 file observed), each verified individually.

## 11. First demonstrable increments (M0)

Each increment is small, testable and ends in something visible. They are ordered so each builds on the last; none needs π₀ until D3.

| # | Increment | Milestone items | Visible result | Checks |
|---|---|---|---|---|
| D1 | Package skeleton + task interface v0 + observer recorder v0 | 0.0b, 0.2c, 0.3e | A video of a seeded random-action Can episode recorded through our own interface, with its metadata sidecar | T0.6, T0.15; rendering has no side effects |
| D2 | Snapshot spike | 0.4b | New full-snapshot tests (restore identity and replay identity) passing. The existing simulator-state-only diagnostic stays a separate test; it is expected to keep failing, because simulator state alone is not the full snapshot, and its xfail marker is only revisited if its own behaviour changes | T0.11 |
| D3 | π₀ loaded and evaluated | 0.2a, 0.2b, 0.3b | A video of π₀ performing Can on fixture seeds; episode JSONL | T0.5, T0.7 |
| D4 | Scripted controller | 0.5a | A video of the scripted pick-and-place; measured success rate | T0.12 |

D1 is proposed as the first hands-on implementation task. A code skeleton (interfaces, docstrings, `NotImplementedError` bodies, tests that define done) is prepared once this design is reviewed.

## 12. Dependencies and risks for M0

| Risk | Effect | Mitigation |
|---|---|---|
| No licensed state-based Can checkpoint | Path 2 if licensed demonstrations are verified (pretraining effort and compute); Path 3 and a re-scope decision if not | Detected at 0.2b; demonstrations verified under 0.1 before Path 2 is chosen |
| Snapshot cannot be made exact | Bank-based evaluation design at risk | Spike (D2) early; candidate components in §4.3 |
| Upstream gym/robosuite versions conflict with ours | Install blocked | Pin to upstream versions; adapter at the training boundary |
| macOS rendering or upstream code differ from Linux | Smoke test results differ by platform | Linux CPU is the reference platform (CI); macOS outcomes recorded separately |

## 13. Requirement coverage of this design

| Requirement | Section |
|---|---|
| FR-ENV-01, -03, -05 | §3 |
| FR-ENV-04, NFR-REP-03 | §4 |
| FR-EVAL-06 | §5 |
| FR-ART-03, FR-DIS-10, FR-POL-04 | §6 |
| FR-EVAL-04 (subset), FR-EVAL-07 | §7 |
| FR-TRK-01, -02 | §8 |
| NFR-COD-01, NFR-PORT-03 | §9 |
| FR-ART-01, -02, NFR-COD-02, NFR-LIC-01 | §10 |
| NFR-PORT-01, -02 | §1, §2 (device selection by config is added with training in M1) |
