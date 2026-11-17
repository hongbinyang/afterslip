# Milestone definitions and work breakdown

Status: draft. Scope and acceptance follow [requirements v1.1 draft](requirements.md). This document elaborates the work and does not change mandatory requirements; where the two disagree, the requirements govern. Technical approaches listed here are candidates until confirmed by evidence. A blocked or failed check is recorded, not treated as a pass. Test procedures for each check are in [test-plan.md](test-plan.md).

**Markers:** **[assumption]** = to be confirmed against upstream configs or measurements. **[open]** = a decision point not yet decided; the decision and its rationale are recorded when made. **Demo** and **effort** entries help planning; they are not acceptance gates.

**Cross-cutting:** Record artifact provenance, configs, commits, seeds, failures, hardware and costs. Keep test data separate from design and tuning; generate the test bank only after the M2 design lock. Record all deviations; consequential scope changes require an approved re-scope decision. Five training seeds and 200 episodes per suite are minimums under the requirements; bootstrap and failure reporting follow §7.3.

**Video:** videos serve as evidence as well as demonstration. One observer camera records scenes; the state-based policy does not consume its images. The recorder configures camera, frame rate, resolution and rendering backend. Every video has a metadata record: episode ID or bank-state ID, seed, policy checkpoint hash, suite, camera, frame rate, resolution, rendering backend, frame count and episode length. Videos are linked from the per-episode records, run manifests and milestone reports. The final evaluation videos of FR-EVAL-06 are required (M3). Earlier videos are demos and inspection aids: an unavailable renderer before M3 is reported as a blocker for that video, not as a failed milestone gate. Encoding and rendering details are candidate implementation choices ([setup.md](setup.md)).

**Tracks:** the **Core track** (M0–M5) answers the research question on the Core task selected under FR-ART-01 (candidate: Panda/Can). The **FR3 track** (F0, F1; requirements §6.9) runs alongside: F0 is a learning and feasibility sequence during M0–M2 and is not a gate for Core milestones; F1 applies the method to an FR3 sorting scene after M3, as a demonstration or a research comparison. FR3 compute, effort and schedule, and any impact on Core, are recorded and reviewed in every milestone report. Core protections (validation-only selection, design lock before the test bank, one test evaluation, §7.3 statistics) apply to Core and are reproduced separately for any FR3 research comparison.

**Effort:** no durations are estimated yet. M0 records actual effort and compute; a first effort and compute projection for M2 validation tuning and M3 final runs is made in the M1 report, from M1 actuals (NFR-CMP-02).

---

## M0 Verify

**Goal:** establish a licensed, compatible and reproducible starting point before research code is written.

**Entry criteria:** requirements and this milestone document accepted; canonical checkout available.

**Exit checks (requirements §10 M0):**
- [ ] `ARTIFACTS.md` complete for every artifact used (FR-ART-02, NFR-LIC-01)
- [ ] Decision-table path chosen and recorded (FR-ART-01, §6.1.1)
- [ ] FR-POL-01 reproduction result (Path 1) or FR-POL-03 pretraining result (Path 2) recorded
- [ ] Linux CPU smoke test (NFR-PORT-03) passed, or recorded as BLOCKED with cause and an approved deviation
- [ ] Novelty finding recorded (FR-NOV-01)
- [ ] M0 report (FR-TRK-05)

**Deliverables:** `ARTIFACTS.md`; artifact decision record; reproduction protocol and result; pinned environment lock files and setup instructions; smoke-test record; video recorder v0 and early recording check; snapshot spike test; scripted-controller feasibility result; mitigation proposal; prior-work review; `reports/M0.md`.

**Demo:** recorded video of π₀ performing Can on fixture seeds (any rendering blocker reported); the snapshot round-trip test passing.

**Decision points [open]:** canonical checkout and branch convention (0.0a); smoke-test platform (0.0c); artifact path (0.2b); reproduction tolerance, fixed before the run (0.3a); scripted-controller threshold and sample, fixed before assessment (0.5a); Core mitigation (0.5b); any BLOCKED deviation; re-scope if the novelty review finds the RQ answered.

**Review triggers:** no licensed demonstrations (Path 3); RQ already answered (0.6).

| # | Item | DoD |
|---|---|---|
| 0.0a | Canonical git checkout; branch and review convention | One checkout documented; existing files preserved |
| 0.0b | Python version and dependency lock tooling compatible with upstream; test and lint tools (see [setup.md](setup.md)) | Documented checks pass on supported environments; failures and limitations recorded (NFR-REP-01) |
| 0.0c | CI platform and CPU smoke route, subject to availability and artifact access **[open]** | Results or blockers recorded in M0 report |
| 0.1 | `ARTIFACTS.md` register (source, version/hash, license, redistribution, date) | Every artifact used has all FR-ART-02 / NFR-LIC-01 fields |
| 0.2a | Install upstream reference code at a pinned commit; record its exact robosuite / robomimic / MuJoCo binding versions. **[assumption]** Robomimic Can does not need `mujoco-py`; if it does, record an NFR-PORT-04 limitation and decide whether to patch | Pinned versions table and Linux CPU installation result or documented blocker |
| 0.2b | Verify a licensed state-based Can checkpoint or follow the artifact decision table; map obs keys + order, normalization stats, action dim (expected 7, OSC_POSE delta **[assumption]**), action chunk length, cond/obs history steps, denoising steps, control frequency, horizon H; artifact path per §6.1.1 **[open]** | Table in the artifact decision; a test that loads the checkpoint and asserts these values (FR-ART-03) |
| 0.2c | Task interface v0: Gymnasium-style wrapper (`reset`, `step`, `get_state`, `set_state`) over the pinned Can environment; benchmark success predicate recorded per step; horizon H from the task (FR-ENV-03, -05) | Wrapper runs a random-action episode; unit test that success flag and H come from the benchmark |
| 0.3a | Reproduction protocol (seeds, ≥ 200 episodes, CI method, tolerance **[open]**) written **before** running | Protocol committed before the first evaluation run |
| 0.3b | Minimal eval loop: load π₀ → N nominal episodes → per-episode JSONL (seed, success, steps) | Loop runs a small number of episodes on CPU or documents a blocker; JSONL schema matches FR-EVAL-04 subset |
| 0.3c | Full reproduction run (GPU or CPU, whichever is practical) and comparison with the released figure; freeze and content-hash π₀ | Result + CI + verdict vs tolerance in M0 report; π₀ content hash recorded (FR-POL-01, -02) |
| 0.3d | **Conditional, Path 2 only:** pretrain a state-based π₀ by flow-matching behaviour cloning on the verified demonstrations, using the configuration published with the reference code; freeze and hash π₀; evaluate by the FR-POL-01 protocol; estimate the extra compute before starting and report actuals. Path 3 (no licensed demonstrations) is not handled here: work stops pending a re-scope decision (§6.1.1) | Pretraining config provenance, state observation keys, π₀ hash, FR-POL-01 result and compute in M0 report (FR-POL-03) |
| 0.3e | Video recorder v0: configurable offscreen recording of an episode (camera, frame rate, resolution, rendering backend from config) with a metadata record per video; early recording check on fixture seeds on each available platform | Readable video with metadata produced, or rendering blocker recorded per platform (FR-EVAL-06 groundwork) |
| 0.4 | CPU smoke: load π₀, 3 episodes, load a fixture bank state; on Linux x86 CPU; in CI if feasible; attempt macOS arm64 | Smoke job passes, or BLOCKED with cause and an approved deviation; tested backends listed (NFR-PORT-03, -04) |
| 0.4b | Snapshot spike: run 50 steps → save snapshot → run 100 more (trajectory A); restore → run the same 100 actions (trajectory B); assert qpos/qvel/object pose equal with tolerance 0 (same backend). Include controller goal state and RNG | Test passes, or documented list of missing state with a fix plan |
| 0.5a | Candidate scripted Can controller feasibility: approach → grasp → lift → move → release using OSC deltas; nominal success on a predeclared sample of seeds | Success rate recorded; threshold, sample size and uncertainty fixed before assessment **[open]** |
| 0.5b | Core mitigation choice: written comparison (implementation cost in upstream code, hyperparameter count, prior evidence) **[open]** | Decision recorded with rationale (FR-RL-04) |
| 0.6 | Full-text prior-work review | Novelty classification in M0 report (FR-NOV-01) |
| 0.7 | M1 compute estimate (environment steps and measured training throughput from a representative pilot; otherwise labelled assumptions and uncertainty) | Number + assumptions in M0 report |
| 0.8 | M0 report: work done, measurements, costs, deviations, negative findings, go/review decision | `reports/M0.md` merged (FR-TRK-05) |

Dependencies: 0.0 → everything; 0.1 ∥ 0.6 (start immediately); 0.2a → 0.2b → 0.3*, 0.4; 0.3c fails tolerance or no usable checkpoint → 0.3d; 0.2a → 0.2c → 0.4b, 0.5a (these do not need π₀).

Key technical risks: binding/version mismatch (0.2a); snapshot does not capture controller/RNG state (0.4b); reproduction outside tolerance → Path 2 pretraining (adds GPU time).

---

## M1 Fine-tune (E1: nominal-only ReinFlow)

**Goal:** run RL fine-tuning end-to-end on nominal episodes and measure its effect and cost. No disturbance bank exists yet.

**Entry criteria:** M0 accepted; π₀ frozen and hashed; environment pinned; smoke test passed or deviation approved; artifact path chosen.

**Exit checks (requirements §10 M1):**
- [ ] 5 seeds launched; diverged or failed seeds kept and reported (FR-RL-07)
- [ ] If fewer than 5 complete, new seeds added and reported, up to a cap of 8 launches
- [ ] CIs over completed seeds; launches and completed runs reported separately
- [ ] If fewer than 5 runs complete after 8 launches, the shortfall is recorded with its effect on Core statistical support (Core comparisons use ≥ 5 training seeds, FR-EVAL-03, with CIs conditional on completion; ≥ 5 completed runs is a proposed quality target) and raised for review; failed runs are never discarded
- [ ] Compute recorded (NFR-CMP-02)
- [ ] Effect reported whether positive, null or negative
- [ ] M1 report (FR-TRK-05)

**Deliverables:** training entry point and configs; run manifests and logs for every launch; checkpoints; learning curves; validation-seed E1 results; RL step-budget proposal; hyperparameter provenance record; Core compute projection; `reports/M1.md`.

**Demo:** learning curve of nominal validation success vs environment steps, with π₀'s level marked; videos of π₀ and a fine-tuned checkpoint on the same validation seeds.

**Decision points [open]:** compute platform for training (cloud GPU or local); shared RL environment-step budget (1.10); continuation to M2.

**Review triggers:** none specific to M1 in requirements §10; the M1 actuals set the baseline for the "Core compute > 2× projection" trigger.

| # | Item | DoD |
|---|---|---|
| 1.1 | Training entry point wrapping upstream ReinFlow behind `afterslip/rl/`; project config overrides only; local patches in `third_party/patches/` (NFR-COD-02) | Training smoke runs on a supported backend; CPU attempted and any upstream limitation recorded |
| 1.2 | Run manifest + file logger: config, git commit, π₀ hash, seed, hardware, OS, backend, library versions, env steps, wall-clock | Manifest written for every run; unit test on schema (FR-TRK-01, -02) |
| 1.3 | Checkpointing at fixed env-step intervals; resume support | Checkpoints appear at expected steps; resume reproduces step counter (FR-RL-06) |
| 1.4 | Periodic validation-seed nominal evaluation during training (learning curves) | Curve plotted from logs (FR-TRK-04) |
| 1.5 | Device adapter (cpu/cuda/mps) and a one-line device switch in config | Device selection is configurable; tested backends and limitations recorded (NFR-PORT-02, -04) |
| 1.5b | Vectorized rollouts with configurable worker count | Throughput measured for 1 and N workers (FR-ENV-06) |
| 1.6 | Seed runner: launch 5 seeds, record failures, add seeds up to 8 launches; never overwrite a failed seed | Run table lists all launches incl. failures (FR-RL-07) |
| 1.7 | Standalone evaluation command on any checkpoint using validation seeds | One documented command (FR-EVAL-07) |
| 1.8 | Action-divergence metric vs π₀ on fixed validation nominal states | Unit test with a toy policy pair (FR-EVAL-05) |
| 1.9 | Hyperparameter provenance: start from the official ReinFlow Can config; record the source file and commit, and every deviation with a reason | Provenance table in M1 report (FR-RL-01) |
| 1.10 | RL environment-step budget proposal, shared by all RL variants in Core comparisons, based on where **validation-seed** E1 learning curves flatten and on the compute projection; test seeds and test banks are never used for this or any other selection **[open]** | Budget recorded; carried into the M2 design lock (§7.4) |
| 1.11 | M1 report: E1 effect with CI, compute actuals, and a Core compute and effort projection split into M2 validation tuning (2.9c) and M3 final runs (3.1–3.3), plus their total | `reports/M1.md` merged |

Note: E1 here uses validation seeds for development and is also the first round of nominal-only hyperparameter selection. Training and validation-based tuning start in M1 and continue in M2; all of it finishes before the M2 lock. The final E1 test numbers are produced in M3 with all other variants, so test seeds are used once.

---

## M2 Bank + design lock

**Goal:** build disturbances and train/validation banks, check headroom and recoverability, complete all validation-based tuning and selection for every Core variant and baseline, then freeze every design parameter and selected value before the test bank exists.

**Entry criteria:** M1 accepted; reference and recovery controllers chosen (from 0.5a); snapshot spike resolved (0.4b).

**Exit checks (requirements §10 M2):**
- [ ] Unit tests pass (triggers, disturbances, bank save/restore, recoverability labelling, metrics, FR-DIS-10 reconstruction; NFR-COD-01)
- [ ] Trigger rate and verified-recoverable fraction reported
- [ ] Design-lock record (all parameters) committed **before** the test bank exists
- [ ] M2 report (FR-TRK-05)

**Deliverables:** disturbance and trigger code; train and validation banks with seeds and hashes; recoverability labels; E0 validation results; validation tuning runs and selection log; unseen-arrangement distribution in config; re-grasp counting definition and test; bootstrap code and tests; design-lock record and `design-lock` git tag; test bank with hash (generated after the tag); `reports/M2.md`.

**Demo:** side-by-side video from one validation (or fixture) bank state: the drop, then π₀'s attempt to recover. Test-bank states are not used for demos before the M3 evaluation.

**Decision points [open]:** the parameter set in 2.0 (recorded in the design lock); response to any review trigger; δ (2.9); approval of the design lock (2.10).

**Review triggers (validation only):** π₀ succeeds on > 80% of disturbed validation states; < 50% of validation states verified recoverable. Thresholds are starting points and are revisited with rationale; any design change happens before the lock.

**Parameters to decide before the lock (2.0):**

| Parameter | Requirement | Status |
|---|---|---|
| Disturbance source during training: online injection or train-bank sampling | FR-DIS-08 | **[open]**; decided after 2.4 from a cost/bias comparison |
| History length k stored per snapshot | FR-ENV-04 | Set from π₀'s observation history (0.2b) **[assumption]** |
| N recovery attempts per state | FR-DIS-07 | **[open]**; set from labeller cost vs label stability on validation |
| Magnitude ranges per disturbance type; held-out range for E5 | FR-DIS-02, -09 | **[open]**; tuned on validation within the review-trigger bounds |
| Selection probability per disturbance type; nominal/disturbed mix ratio; curriculum | FR-DIS-02, FR-RL-03 | **[open]** |
| Bank split sizes (train / validation / test) | FR-DIS-05, FR-EVAL-02 | Test ≥ 200 states; attempts needed ≈ target ÷ measured trigger rate |
| Reward shaping (if any) | FR-RL-02 | Default: none (sparse success) |
| Unseen-arrangement distribution (object initial poses outside the training range, numeric) | FR-ENV-07 | Defined and sampler-tested by the lock even if E5 is deferred (2.9d) |
| Selected hyperparameters per Core variant and baseline (λ / d / β for the chosen mitigation; RL settings) | FR-RL-05 | Selected values from 2.9c, not only search spaces |
| Video subset rule: how the fixed test seeds and test-bank states to be filmed are chosen (e.g. the first k IDs in a deterministic order), and recorder settings | FR-EVAL-06 | **[open]**; the rule is fixed in the lock; concrete IDs are drawn by that rule after the test bank exists (2.11) |

| # | Item | DoD |
|---|---|---|
| 2.1 | Snapshot format v1 (from 0.4b): sim state, task state, controller state, RNG, last-k obs/action history, metadata; versioned schema; no policy internal state | Round-trip unit test (FR-ENV-04, NFR-REP-03) |
| 2.2 | Trigger predicates D1/D2/D3 as pure functions of sim/task state | Unit tests with hand-built states (fires / does not fire) (FR-DIS-03) |
| 2.3 | Disturbance operators (gripper force-open; object pose perturbation at closure; impulse) with magnitude ranges from config | Unit tests: state changes as specified, nothing else changes (FR-DIS-01, -02) |
| 2.4 | Bank generator: chosen fixed reference controller rolls out, trigger fires, disturbance applied, snapshot saved; trigger rate + non-firing count logged | Train + validation banks generated with seeds and hashes (FR-DIS-04, -05, -06) |
| 2.5 | Recoverability labeller: chosen reference recovery controller, N attempts per state | Labels + verified-recoverable fraction reported (FR-DIS-07) |
| 2.6 | Policy reset/reconstruction rule in the π₀ adapter; unit test per policy type | Test: after restore, no pending chunk, history equals stored history (FR-DIS-10) |
| 2.7 | Disturbed episode runner + end-to-end online-disturbance suite | Both suites run on validation; trigger rate reported (FR-EVAL-08) |
| 2.7b | Precise definition of re-grasp attempts (e.g. count of gripper close events after the first loss of object contact) and a verified counter | Definition in the design document; unit test on scripted trajectories with known re-grasp counts; done before any test evaluation (FR-EVAL-04) |
| 2.8 | E0 on validation; review-trigger check | Numbers + decision recorded |
| 2.9 | δ chosen from validation variance (§7.3) **[open]** | δ recorded with rationale |
| 2.9b | Hierarchical bootstrap (§7.3) implemented and unit-tested on synthetic data | Tests pass; commit hash recorded in 2.10 |
| 2.9c | Validation tuning and selection for every Core configuration: disturbed fine-tuning M-none (mix sampler), the chosen mitigation (mitigation hook in `rl/`), nominal-only fine-tuning, RL-from-scratch baseline; training on train bank and nominal episodes, selection on validation bank and validation seeds only | Selection log with candidates, validation scores and selected values (FR-RL-03, -04, -05) |
| 2.9d | Unseen-arrangement distribution specified numerically in config; sampler unit test | Config committed; test shows samples fall outside the training range (FR-ENV-07) |
| 2.10 | **Design-lock record**: all 2.0 parameters, step budget (1.10), mitigation, **selected** hyperparameter values for every Core configuration (2.9c) and the search spaces they came from, δ, re-grasp definition, unseen-arrangement distribution, bootstrap code commit | Lock approved and committed (git tag `design-lock`) **before** 2.11 |
| 2.11 | Test bank (≥ 200 states) generated and hashed after the lock tag; fixed test video subset drawn by the locked rule | Hash and video-subset IDs in a post-lock bank manifest; test bank never loaded by training code (enforced by access controls and tests appropriate to the chosen storage design) |
| 2.12 | M2 report | `reports/M2.md` merged |

---

## M3 Core

**Goal:** train the locked Core configurations (no further tuning or design changes), evaluate each on test exactly once, and give H1/H2 verdicts.

**Entry criteria:** design lock tagged; test bank generated and hashed; novelty review complete (FR-NOV-01 must be done before M3 begins).

**Exit checks (requirements §10 M3):**
- [ ] Headline figure with CIs (FR-TRK-03)
- [ ] H1 and H2 verdicts by §7.3
- [ ] M3 report (FR-TRK-05)

**Deliverables:** trained checkpoints for every Core variant and baseline (≥ 5 training seeds launched per configuration; ≥ 5 completed runs is a proposed quality target; launches, completions, failed-run rate and any shortfall reported for review; failed runs are never silently replaced); raw per-episode test logs; headline figure; learning curves; videos; verdict table; `reports/M3.md`.

**Demo:** the headline scatter plot (disturbed vs nominal success, CI bars, π₀ marked) and side-by-side videos of π₀ and each variant from the same test-bank state (from 3.7b).

**Decision points [open]:** response to the compute review trigger if it fires; acceptance of M3.

**Review triggers:** total Core compute (M2 validation tuning plus M3 final runs) exceeds the M1-based projection by more than 2×. Each part is also compared with its own projection, so that an overrun is located early.

| # | Item | DoD |
|---|---|---|
| 3.1 | Final disturbed fine-tuning runs (nominal/disturbed mix) for M-none and the chosen mitigation, with locked configs | ≥ 5 seeds launched per variant (target: ≥ 5 completed); launches, failures and any shortfall reported (FR-RL-03, -04, -07) |
| 3.2 | Final nominal-only fine-tuning runs (E1 / FR-BASE-04) at the locked budget, if the M1 runs do not already match the locked config | ≥ 5 seeds launched at the shared budget (target: ≥ 5 completed); failures reported |
| 3.3 | Final RL-from-scratch baseline runs (PPO or SAC) with same obs, reward, env-step budget and locked config | ≥ 5 seeds launched (target: ≥ 5 completed); failures reported (FR-BASE-03, E4a) |
| 3.4 | Lock compliance check: every run's manifest matches the `design-lock` record; any deviation is reported, not silently corrected | Compliance table in M3 report |
| 3.5 | One-shot test evaluation of π₀ and every selected configuration on Nominal + Disturbed + end-to-end suite | Raw per-episode logs archived (FR-EVAL-01..04, -08, FR-BASE-01) |
| 3.6 | Results tables: mean and 95% CI over seeds, per-seed values, **failed-run rate beside every CI**, S_dist(verified) with denominator | Tables generated from raw logs by script (FR-EVAL-03, §7.1, §7.3) |
| 3.7 | Headline figure, learning curves | Figures in report (FR-TRK-03, -04) |
| 3.7b | Final evaluation videos on the fixed test video subset for π₀ and every final variant: bank-based disturbed episodes start from identical bank states, and nominal episodes from identical test seeds, across policies. End-to-end suite videos share the nominal test seeds but follow policy-dependent trajectories, so their physical states are not identical across policies | Videos with metadata for every (policy, subset entry) pair; linked from episode records and the M3 report (FR-EVAL-06) |
| 3.8 | Cost accounting per variant; compute-trigger check of total Core compute (M2 tuning + M3 final runs) against the M1 projection, with each part compared to its own projection | Cost table; trigger status recorded (§7.4, NFR-CMP-02) |
| 3.9 | H1/H2 verdicts by the predeclared rules | Verdict table in M3 report |
| 3.10 | M3 report | `reports/M3.md` merged |

---

## M4 Extended (budget permitting)

**Goal:** trace the trade-off frontier and test generalization, as budget allows. Each experiment that is started meets the M3 standard; an experiment not started is listed as not done.

**Entry criteria:** M3 accepted; M4 budget and list of experiments approved.

**Exit checks (requirements §10 M4):** same standard as M3 for each completed experiment; M4 report.

**Deliverables (only for approved experiments):** E3b: extra mitigation runs and sweeps; E4b: extra-imitation demos, baseline and its declared budget, and the H3 verdict; E5: unseen-arrangement and held-out-magnitude results. Always: `reports/M4.md`, listing experiments not run.

Extended tuning follows the same rule as Core: selection on validation only, a lock addendum committed before any Extended configuration is evaluated on test, and one test evaluation per selected configuration.

**Demo:** frontier plot of all mitigations across their sweep values.

**Decision points [open]:** which Extended experiments to run; FR-BASE-02 budget, declared before running.

**Review triggers:** compute trigger as in M3.

| # | Item | DoD |
|---|---|---|
| 4.1 | Remaining mitigation hooks (M-reg / M-freeze / M-aux, whichever were not Core) and sweeps of λ, d, β on validation (E3b) | Each variant ≥ 5 seeds; selection on validation only |
| 4.2 | Extra-imitation baseline: budget declared before running; scripted recovery demos from **train-bank** states; separate resource accounting (E4b, FR-BASE-02) | Budget record committed before collection; baseline evaluated on test once |
| 4.3 | E5: best variants evaluated on the unseen-arrangement distribution locked in 2.9d and on held-out disturbance magnitudes | E5 results (FR-EVAL-01, FR-DIS-09) |
| 4.4 | Extended results tables with the M3 reporting standard; H3 verdict **only if E4b ran** | Tables; verdict table if applicable |
| 4.5 | M4 report | `reports/M4.md` merged |

---

## M5 Report

**Goal:** assemble reproducible deliverables and findings, including negative or inconclusive results.

**Entry criteria:** M3 (and M4, if run) accepted.

**Exit checks (requirements §10 M5):** all FR-TRK-06 deliverables present; negative and inconclusive results included.

**Deliverables:** source code; `ARTIFACTS.md`; environment lock files; banks with hashes; all configs; selected checkpoints; raw per-episode logs; figures; videos; milestone reports; final report.

**Demo:** a fresh checkout regenerates every figure from released logs with one command.

**Decision points [open]:** redistribution of banks and checkpoints (subject to NFR-LIC-01); final acceptance.

| # | Item | DoD |
|---|---|---|
| 5.1 | Redistribution check: every released file complies with the terms recorded in `ARTIFACTS.md` | Checklist complete (NFR-LIC-01) |
| 5.2 | Figure regeneration script: all figures and tables from raw logs | Clean-checkout run reproduces figures |
| 5.3 | Reproducibility package and rerun protocol: predeclared paired comparison and tolerance; known nondeterminism documented; cross-backend E0 differences reported if measured | Protocol document; results of any rerun performed (NFR-REP-02, -03) |
| 5.4 | Final report: RQ answer, H1–H3 verdicts, costs, limitations, deviations, negative findings | Report accepted and merged |
| 5.5 | Deliverables audit against FR-TRK-06 and the traceability appendix | All rows checked or listed as not done |

---

## F0 FR3 feasibility (FR3 track, alongside M0–M2)

**Goal:** learn and demonstrate the Franka Research 3 arm in MuJoCo step by step, and gather the evidence for the F1 path decision. Every step ends in a recorded video.

**Entry criteria:** environment from M0 item 0.0b available (a manual environment is acceptable for F0.1–F0.2 until lock files exist).

**Completion checks (requirements §10 F0):**
- [ ] Each FR-FR3-02 step (F0.1–F0.4) recorded with video, metadata and written result, or BLOCKED with cause
- [ ] FR3 model and hand sources verified in `ARTIFACTS.md` (FR-FR3-01)
- [ ] F1 path decision written (FR-FR3-03)
- [ ] F0 report (FR-TRK-05)

F0 is not a gate for M0–M3. A blocked or failed F0 step is a valid, reported outcome that feeds the F1 path decision.

**Deliverables:** F0 scripts and tests; videos with metadata; measured end-effector error and grasp success; `ARTIFACTS.md` entries; F1 path decision record; `reports/F0.md`.

**Demo:** the four F0 videos: model, joint motion, end-effector reaching, grasp and lift.

**Decision points [open]:** FR3 model source (F0.0); hand model and attachment method (F0.4); F1 path (F0.5).

**Local observation (2026-11-16, [setup.md](setup.md) §7):** the Menagerie `franka_fr3` model at the candidate commit loads in the project environment but contains no gripper. This applies to that model only, not to every FR3 model.

| # | Item | DoD |
|---|---|---|
| F0.0 | FR3 model source candidates identified and verified (source, version, license) **[open]** | `ARTIFACTS.md` entries (FR-FR3-01) |
| F0.1 | Load, inspect and render the FR3: joints, limits, actuators, keyframes; define one fixed observer camera in the project scene and record its name and pose | Inspection table; video (FR-FR3-02a) |
| F0.2 | Joint-space control to a fixed set of commanded configurations | Final joint error per target recorded; video (FR-FR3-02b) |
| F0.3 | Forward kinematics and an end-effector position controller (e.g. differential inverse kinematics) | End-effector position error on a fixed target set recorded; video (FR-FR3-02c) |
| F0.4 | Gripper on the arm (attached if the model has none) **[open]**; grasp and lift a block on a fixed set of trials | Grasp success rate on the fixed trials; video (FR-FR3-02d) |
| F0.5 | F1 path decision: compare (i) scripted demonstrations in the FR3 sorting scene, (ii) retargeting existing licensed demonstrations to FR3, (iii) any licensed FR3 checkpoints or demonstrations, with evidence, effort and compute estimates and risks **[open]** | Decision record (FR-FR3-03) |
| F0.6 | F0 report | `reports/F0.md` merged |

Dependencies: F0.0 → F0.1 → F0.2 → F0.3 → F0.4; F0.5 uses F0.3–F0.4 results and the M0 artifact findings.

---

## F1 FR3 sorting (FR3 track, after M3)

**Goal:** build the FR3 three-bin sorting scene, obtain an FR3 base policy by the path chosen in F0.5, and apply the Afterslip method to it within the chosen scope, with videos as the project showcase.

**Entry criteria:** M3 accepted; F0 report and F1 path decision accepted; F1 scope decided and recorded: (a) demonstration, or (b) research comparison (FR-FR3-07) **[open]**.

**Completion checks (requirements §10 F1):**
- [ ] Sorting task defined numerically and committed before any demonstrations or policies are produced (FR-FR3-04)
- [ ] Scope (a) or (b) recorded at entry (FR-FR3-07)
- [ ] Base policy obtained, or BLOCKED with cause; fine-tuning outcome recorded if attempted
- [ ] (a) Demonstration: each policy obtained evaluated with a stated episode count and 95% CI, labelled as a demonstration
- [ ] (b) Research comparison: the FR-FR3-05 protocol complete (FR3 train/validation/test separation, FR3 design lock before the FR3 test bank, ≥ 200 episodes per suite, one test evaluation per configuration, §7.3 statistics)
- [ ] Videos of each policy obtained; before/after videos on the same fixed FR3 states when both a base and a fine-tuned policy exist (FR-FR3-06)
- [ ] F1 report, including negative or blocked outcomes

**Deliverables (conditional on the chosen path, scope and actual outcomes):** always: sorting scene and task definition, the scope record, evaluation logs and videos for each policy obtained, and `reports/F1.md` (including blocked or negative outcomes). Per path: demonstration generator, retargeting code, or the provenance record of an existing FR3 checkpoint. If a base policy is obtained: the policy and its hash. If fine-tuning is attempted: FR3 disturbances and fine-tuned checkpoints. Research comparison only: FR3 train/validation banks, FR3 design-lock record, FR3 test bank with hash.

**Demo (illustrative):** side-by-side video from the same FR3 state after a drop: the base policy and, if one exists, the fine-tuned policy. It shows whatever behaviour occurs; recovery is not assumed. Quantitative claims come only from F1.7.

**Decision points [open]:** F1 scope (demonstration only or research comparison); FR3 parameter values, re-selected on FR3 validation data (Core values are not assumed to transfer).

**Review triggers:** F1 effort or compute exceeds the F0.5 estimate by more than 2×; FR3 π₀ nominal success too low to make disturbances meaningful (threshold fixed in the task definition).

| # | Item | DoD |
|---|---|---|
| F1.1 | Sorting task definition: objects and categories, bins and positions, initial-state distribution, success predicate, horizon | Committed before data generation; unit tests for the success predicate (FR-FR3-04) |
| F1.2 | Sorting scene and FR3 robot/scene adapter behind the Core task interface, with one fixed observer camera | Scene loads; random-action episode runs; video |
| F1.3 | **Path-dependent:** (i) scripted demonstrations: demonstration generator; (ii) retargeting: retargeting pipeline applied to licensed demonstrations; (iii) existing FR3 checkpoint: no data generation, the checkpoint is verified under FR-FR3-01 | (i)/(ii): demonstration set with provenance and success rate, videos of samples. (iii): `ARTIFACTS.md` entry and compatibility mapping |
| F1.4 | FR3 base policy π₀ by the chosen path: flow-matching pretraining on the F1.3 data (paths i, ii) or the verified checkpoint (path iii); freeze and hash | Provenance and π₀ hash; nominal success with CI on FR3 evaluation seeds (episode count per scope, FR-FR3-07); BLOCKED with cause if no base policy is obtained |
| F1.5 | FR3 disturbances (D1, D2) for fine-tuning (if fine-tuning is in scope). Research comparison: FR3 **train and validation** intervention-state banks only; no test bank yet | Disturbance unit tests; train/validation banks with seeds and hashes (FR-FR3-05) |
| F1.6 | Fine-tuning (if in scope) with the Core-selected method; hyperparameters selected on FR3 validation data only (Core-locked values are not assumed to transfer) | Selection log citing FR3 validation data only |
| F1.6b | Research comparison only: FR3 design lock with all selected values and the FR3 video-subset rule, committed and tagged | FR3 lock tag |
| F1.6c | Research comparison only: FR3 test bank generated and hashed **after** the FR3 lock tag; FR3 test video subset drawn by the locked rule | Test-bank manifest whose commit has the FR3 lock tag as an ancestor (FR-FR3-05) |
| F1.7 | Evaluation: demonstration-only results labelled as such, or one test evaluation per configuration with §7.3 statistics | Result tables; verdicts only for a research comparison |
| F1.8 | Videos of each policy obtained; before/after videos on the same fixed FR3 states when both policies exist | Videos with metadata (FR-FR3-06) |
| F1.9 | F1 report | `reports/F1.md` merged |

---

## Suggested implementation sequence

Ordered so that each step builds on the previous one:

1. Workspace, CI, tests (0.0): engineering habits used for the rest of the project.
2. Task interface and snapshot spike (0.2c, 0.4b): what "state" means in a physics simulator.
3. Scripted controller (0.5a): frames, OSC delta actions, grasping.
4. Eval loop + per-episode logging (0.3b, 1.7): measurement discipline.
5. Triggers and disturbances (2.2, 2.3): predicates on state, unit testing physics code.
6. Bootstrap (2.9b): statistics behind the verdicts.
7. Mitigation hook (2.9c): how fine-tuning objectives are modified.

FR3 track, in parallel: F0.1 → F0.2 → F0.3 → F0.4 build up from a robot model to a grasp: MJCF and kinematic chains, actuators and joint control, frames and Jacobians, contacts and grasping.

---

## Appendix A: Requirements traceability

Maps each requirement to the items that satisfy it. The requirements document remains authoritative. Tier follows requirements §3.1.

| Requirement | Items | Notes |
|---|---|---|
| FR-ART-01 | 0.2b | |
| FR-ART-02 | 0.1 | |
| FR-ART-03 | 0.2b | |
| FR-NOV-01 | 0.6 | Must be complete before M3 begins |
| FR-ENV-01 | — | Design constraint (MuJoCo only); checked in design review, no work item |
| FR-ENV-02 | 0.2b | Proposal confirmed or replaced by FR-ART-01 |
| FR-ENV-03 | 0.2c | |
| FR-ENV-04 | 0.4b, 2.1 | |
| FR-ENV-05 | 0.2c | |
| FR-ENV-06 | 1.5b | |
| FR-ENV-07 | 2.9d | Defined by the lock even if E5 is deferred; used by 4.3 |
| FR-DIS-01 | 2.3 | D3 is SHOULD |
| FR-DIS-02 | 2.0, 2.3 | |
| FR-DIS-03 | 2.2 | |
| FR-DIS-04 | 2.4 | |
| FR-DIS-05 | 2.0, 2.4, 2.9c, 2.10, 2.11 | All design and tuning on train/validation before the test bank exists |
| FR-DIS-06 | 2.4 | |
| FR-DIS-07 | 2.5 | |
| FR-DIS-08 | 2.0 | MAY; the choice is recorded |
| FR-DIS-09 | 2.0, 4.3 | Extended |
| FR-DIS-10 | 2.6 | |
| FR-POL-01 | 0.3a, 0.3c | Path 1 only |
| FR-POL-02 | 0.3c | |
| FR-POL-03 | 0.3d | Path 2 only (conditional item) |
| FR-POL-04 | 0.2b, 0.3d, F0.1, F1.2, F1.4 | State inputs; one observer camera |
| FR-RL-01 | 1.1, 1.9 | |
| FR-RL-02 | 2.0 | |
| FR-RL-03 | 2.9c, 3.1 | |
| FR-RL-04 | 0.5b, 2.9c, 3.1, 4.1 | Core: M-none + one mitigation |
| FR-RL-05 | 1.10, 2.9c, 2.10, 4.1 | Selection before lock; M3 uses locked values |
| FR-RL-06 | 1.3 | |
| FR-RL-07 | 1.6 | Applies to every training run, M1–M4 |
| FR-RL-08 | — | Stretch; not planned |
| FR-BASE-01 | 3.5 | |
| FR-BASE-02 | 4.2 | Extended |
| FR-BASE-03 | 3.3 | |
| FR-BASE-04 | 1.1–1.6, 3.2 | |
| FR-EVAL-01 | 3.5, 4.3 | Unseen arrangement is Extended |
| FR-EVAL-02 | 2.0, 3.5 | |
| FR-EVAL-03 | 1.6, 3.1–3.3, 3.6 | ≥ 5 training seeds; ≥ 5 completed is a proposed target; a shortfall is raised for review |
| FR-EVAL-04 | 0.3b, 2.7b, 3.5 | |
| FR-EVAL-05 | 1.8 | |
| FR-EVAL-06 | 0.3e, 2.0, 2.11, 3.7b | Recorder from M0; subset rule locked in M2; final videos in M3 |
| FR-EVAL-07 | 1.7 | |
| FR-EVAL-08 | 2.7, 3.5 | |
| FR-TRK-01 | 1.2 | |
| FR-TRK-02 | 1.2 | |
| FR-TRK-03 | 3.7 | |
| FR-TRK-04 | 1.4, 3.7 | |
| FR-TRK-05 | 0.8, 1.11, 2.12, 3.10, 4.5, 5.4 | |
| FR-TRK-06 | 5.5 | |
| NFR-PORT-01 | — | Design constraint; checked in code review |
| NFR-PORT-02 | 1.5 | |
| NFR-PORT-03 | 0.4 | |
| NFR-PORT-04 | 0.2a, 0.4, 1.5 | |
| NFR-REP-01 | 0.0b | |
| NFR-REP-02 | 5.3 | |
| NFR-REP-03 | 0.4b, 2.1, 5.3 | Cross-backend part is SHOULD |
| NFR-CMP-01 | — | Permission, no work item |
| NFR-CMP-02 | 0.7, 1.11, 3.8 | Every milestone report states compute |
| NFR-COD-01 | 0.0b, M2 exit checks | |
| NFR-COD-02 | 1.1 | |
| NFR-LIC-01 | 0.1, 5.1 | |
| FR-FR3-01 | F0.0, F0.4 | Includes the hand model |
| FR-FR3-02 | F0.1–F0.4 | |
| FR-FR3-03 | F0.5 | |
| FR-FR3-04 | F1.1 | |
| FR-FR3-05 | F1.5–F1.7 (incl. F1.6b, F1.6c) | Only if F1 includes a research comparison |
| FR-FR3-06 | F1.8 | |
| FR-FR3-07 | F1 entry criteria and completion checks | Scope (a) demonstration or (b) research comparison |
| Stretch tier (E6, FR-RL-08) | — | Not planned |
