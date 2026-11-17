# Test plan

Status: draft. This plan defines how each milestone check in [milestones.md](milestones.md) is verified. The [requirements](requirements.md) remain authoritative. Commands, tool versions and file paths marked **TBD** are fixed when the corresponding code and environment exist (see [setup.md](setup.md)); versions are unverified until recorded under "Tested configurations" there.

## 1. Conventions

**Test IDs:** `T<milestone>.<n>`, e.g. `T0.3`. The *Item* column links to the milestone work item.

**Levels:**

| Level | Meaning | Typical runner |
|---|---|---|
| U: unit | One function or class, small synthetic inputs, seconds | pytest, every push |
| I: integration | Several components against the real simulator, minutes | pytest (marked `integration`), every push where CPU time allows, otherwise nightly |
| S: smoke | End-to-end path runs at minimal size on a given platform | CI job per platform |
| P: protocol | Statistical or experimental procedure with a predeclared pass rule | Scripted run; result recorded in the milestone report |
| A: audit | Document or record check against a checklist | Checklist in the milestone report |
| R: reproducibility | Same input → same or statistically equivalent output | Scripted rerun |

**Gate classes:**
- **Core gate:** verifies an outcome that the requirements mandate (a SHALL requirement or a requirements §10 acceptance condition). Required for milestone acceptance.
- **Project check:** an implementation check chosen by this plan to support a requirement (e.g. CI, a specific episode count, an exact-equality regeneration). Its outcome is recorded and addressed, but its failure is not by itself a failed requirement unless the requirement's own outcome is also not demonstrated.
- **Extended:** required only if the corresponding Extended experiment runs; otherwise NOT RUN.
- **Non-gating:** demos and diagnostics; recorded but cannot block acceptance.

Some rows contain both: the Core part and the project-check part are labelled separately in the Gate column.

**Outcomes:** each executed test records exactly one of:
- **PASS:** pass criterion met.
- **FAIL:** executed, criterion not met. Recorded with the observed value; never re-run silently until it passes. A re-run after a fix is a new, separately recorded execution.
- **BLOCKED:** could not execute (missing artifact, platform limitation, upstream defect). Recorded with cause. A BLOCKED Core gate is not a pass; only the Linux CPU smoke test in M0 (NFR-PORT-03) has a defined BLOCKED route, which requires a recorded cause and approved deviation (requirements §10 M0). Any other BLOCKED Core gate leaves the milestone open.
- **NOT RUN:** not executed (e.g. Extended experiment not approved). Listed explicitly.

**Evidence:** every execution records: test ID, date, git commit, environment (OS, backend, key library versions from the run manifest), command, outcome and the path to raw output. Generated evidence lives under the ignored `outputs/` directory (layout TBD). CI logs or artifacts may record automated checks; milestone reports summarize results and link to durable shared artifacts when available. The storage and redistribution method remains to be selected.

**Fixtures (shared):**
- `fixtures/env_seeds.yaml`: small fixed set of nominal seeds for unit/integration tests (never the validation or test seeds).
- A small fixture bank generated from fixture seeds in a temporary directory by an automated fixture; its format and generation procedure are to be defined.
- Toy policies: a constant-action policy and a random-action policy with fixed seed, for harness tests that must not depend on π₀.
- Synthetic results tables with known ground truth, for metric and bootstrap tests.

**Data separation rule (applies to every test):** unit, integration and smoke tests use fixture seeds and fixture banks only. No test reads validation or test seeds or banks, except the protocol tests that are defined to do so, and no test anywhere reads the test bank before the `design-lock` tag exists.

---

## 2. M0 Verify

| ID | Item | Req | Level | Prerequisites / fixtures | Procedure | Pass criterion | Gate |
|---|---|---|---|---|---|---|---|
| T0.1 | 0.0b | NFR-REP-01 | S | Lock files committed | Create environment from lock files on a clean Linux machine or runner using the documented setup steps; run the unit tests **[TBD command]** | (a) Lock files and per-platform setup instructions exist and the environment builds by following them; (b) unit tests pass | (a) Core; (b) Project check |
| T0.2 | 0.0c | NFR-COD-01 | S | CI configured | Push a commit; CI runs lint, type check and unit tests | CI job succeeds; run linked in M0 report | Project check |
| T0.3 | 0.1 | FR-ART-02, NFR-LIC-01 | A | `ARTIFACTS.md` | For each artifact used, check presence of source URL, version or content hash, license as stated by the source, redistribution terms, date checked; recompute content hashes of downloaded files | Every field present for every artifact; recomputed hashes match; any unverifiable artifact marked unusable | Core |
| T0.4 | 0.2a | NFR-PORT-04 | S | Upstream code at pinned commit | Install on Linux x86 CPU; import upstream modules; create the Can environment | Install and environment creation succeed, or BLOCKED with cause | Core |
| T0.5 | 0.2b | FR-ART-03, FR-POL-04 | U | Checkpoint and its released config | Load the selected state-based checkpoint; assert the policy inputs contain no camera images and match obs keys and order, normalization stats shape, action dim, chunk length, history steps, denoising steps, horizon against the mapping table | All assertions hold | Core |
| T0.6 | 0.2c | FR-ENV-03, -05 | I | Pinned environment, fixture seeds | Run a random-action episode through the task interface; compare per-step success flag with the benchmark's own predicate; check episode ends at H | Flags identical at every step; episode length ≤ H, timeout recorded as failure | Core |
| T0.7 | 0.3b | FR-EVAL-04 (subset) | I | π₀ loaded, fixture seeds | Run 3–5 nominal episodes with the eval loop; validate JSONL output against schema | Valid JSONL with one record per episode; schema check passes | Project check |
| T0.8 | 0.3a, 0.3c | FR-POL-01, -02 | P | Path 1; protocol, including tolerance, committed **before** the run (its commit is an ancestor of the commit recording the first evaluation output) | Run ≥ 200 nominal episodes on fixed seeds; compute success rate and 95% CI by the protocol's method; classify the result as within or outside the predeclared tolerance of the released figure; hash π₀ | Protocol predeclared; result, CI and classification reported (agreement or disagreement); π₀ hash recorded. The test passes when the protocol is executed and reported as declared. A result outside tolerance is reported, not hidden, and the §6.1.1 decision table then applies (Path 2) | Core (Path 1), else NOT RUN |
| T0.9 | 0.3d | FR-POL-03 | P | Path 2 chosen (no usable checkpoint, or reproduction failed); verified demonstrations | Pretrain with the published configuration; evaluate by the FR-POL-01 protocol (fixed seeds, ≥ 200 episodes, 95% CI). Compare with a released figure only if one exists | Config provenance, π₀ hash, evaluation result with CI and compute recorded; comparison reported if applicable | Core (Path 2), else NOT RUN |
| T0.10 | 0.4 | NFR-PORT-03 | S | Lock files; π₀; the generated fixture bank | On Linux x86-64 CPU: load π₀, run a few episodes (project choice: 3), load and restore one fixture bank state. Repeat on macOS arm64 | Linux: completes without error, or BLOCKED with recorded cause and approved deviation (§10 M0). macOS: outcome recorded | Core (Linux); Non-gating (macOS arm64, SHOULD) |
| T0.11 | 0.4b | FR-ENV-04, NFR-REP-03 | I | Task interface v0; fixture seed; fixed action sequence | Run 50 steps; snapshot; run 100 steps (A). Restore snapshot; replay the same 100 actions (B). Compare qpos, qvel, object pose, gripper state, controller goal at every step | Exact equality on the same backend. On mismatch: FAIL with the list of differing state fields | Core |
| T0.12 | 0.5a | FR-DIS-04 (feasibility) | P | Scripted controller; threshold, sample size and CI method predeclared | Run scripted controller on the predeclared nominal seed sample; compute success rate and CI | Success rate and CI recorded against the predeclared threshold; the result informs the reference-controller decision (scripted or π₀, FR-DIS-04) | Project check |
| T0.13 | 0.6 | FR-NOV-01 | A | Prior-work review document | Check each cited work is read in full text and classified; check the novelty classification is stated | Classification recorded with evidence | Core |
| T0.14 | 0.8 | FR-TRK-05 | A | M0 report | Checklist: work done, measurements, costs, deviations, negative findings, go/review decision, tested backends | All sections present | Core |
| T0.15 | 0.3e | FR-EVAL-06 (groundwork) | I | Recorder v0; fixture seed; toy or π₀ policy | Record one short episode on each available platform with one configured observer camera, frame rate and resolution | Video file opens and decodes; frame count consistent with episode length and the configured frame interval (as recorded in metadata); frames not all identical; metadata record complete (episode ID, seed, policy hash, camera, fps, resolution, backend, frame count). Renderer unavailable on a platform: BLOCKED for that platform with cause | Project check |
| D0 | — | — | — | — | Video of π₀ on Can (fixture seeds) | — | Non-gating |

---

## 3. M1 Fine-tune

| ID | Item | Req | Level | Prerequisites / fixtures | Procedure | Pass criterion | Gate |
|---|---|---|---|---|---|---|---|
| T1.1 | 1.1 | FR-RL-01, NFR-PORT-03 | S | π₀, fixture seeds | Run training for a small fixed number of env steps (project choice, e.g. 1k) on the training backend; attempt CPU (SHOULD) | Training backend: runs to completion with loss and success logged. CPU: outcome recorded, limitation listed if BLOCKED (NFR-PORT-04) | Project check (training backend); Non-gating (CPU) |
| T1.2 | 1.2 | FR-TRK-01, -02 | U | — | Generate a manifest from a dummy run; validate against schema; disable optional trackers | All required fields present; local logger works with no external tracker | Core |
| T1.3 | 1.3 | FR-RL-06 | I | Short training run | Train with checkpoint interval c; kill and resume | (a) Checkpoints exist at the fixed env-step interval; (b) resumed env-step counter and optimizer state continue from the checkpoint | (a) Core; (b) Project check |
| T1.4 | 1.5 | NFR-PORT-02, -04 | U/S | — | Switch device in config between `cpu`, `cuda`, `mps`; run a forward pass | Device honoured; unavailable backends listed as unverified, not as passed | Core |
| T1.5 | 1.5b | FR-ENV-06 | I | Fixture seeds | Run identical seeded rollouts with 1 and N workers | (a) Worker count is configurable and both runs complete; (b) same per-episode results for the same seeds; throughput recorded | (a) Core; (b) Project check |
| T1.6 | 1.6 | FR-RL-07 | U | Simulated failing run | Seed runner with an injected crash in one seed | Failed run kept in the run table with status; replacement seed is new and recorded; no seed reused or overwritten (FR-RL-07, all milestones). In M1, the cap of 8 launches is enforced (§10 M1); later milestones apply their own documented seed policy | Core |
| T1.7 | 1.7 | FR-EVAL-07 | S | Any checkpoint | Run the documented evaluation command | Produces per-episode JSONL and summary with one command | Core |
| T1.8 | 1.8 | FR-EVAL-05 | U | Toy policy pair with known divergence | Compute action divergence | Matches analytic value within tolerance; zero for identical policies | Core |
| T1.9 | 1.6, 1.11 | §10 M1, FR-EVAL-03 | P/A | All E1 launches | Check: 5 launches; additional launches only if fewer than 5 complete, ≤ 8 total; CIs over completed seeds; launches vs completions reported; shortfall below 5 completions raised for review | All checks hold | Core |
| T1.10 | 1.10 | FR-RL-05, §7.4 | A | Budget proposal | Check that the budget derivation cites only validation-seed curves and compute | No test seed or test bank referenced | Core |
| D1 | — | — | — | — | Learning curve figure; videos of π₀ and a fine-tuned checkpoint on the same validation seeds | — | Non-gating |

---

## 4. M2 Bank + design lock

| ID | Item | Req | Level | Prerequisites / fixtures | Procedure | Pass criterion | Gate |
|---|---|---|---|---|---|---|---|
| T2.1 | 2.1 | FR-ENV-04, NFR-REP-03 | U/I | the generated fixture bank | Serialize → deserialize → restore each snapshot; step a fixed action sequence; compare with the original continuation. Check schema version field and that no policy internal state is stored | Exact equality on the same backend; schema valid; no policy-state fields | Core |
| T2.2 | 2.2 | FR-DIS-03 | U | Hand-built states | For each trigger D1/D2/D3: states that must fire, states that must not (boundary values of height, contact, timing) | All cases classified as specified | Core |
| T2.3 | 2.3 | FR-DIS-01, -02 | U | Hand-built states | Apply each disturbance at min, mid and max magnitude; inspect state immediately after application, before any physics step; then step a short fixed horizon | At application: the targeted quantity changes by the configured amount (gripper command opened; object pose offset; object velocity impulse) and declared invariants hold (e.g. robot joint state unchanged by an object perturbation; other objects untouched). After stepping: expected qualitative effect observed (e.g. D1 object loses finger contact). Physics-driven changes elsewhere are expected and not checked | Core |
| T2.4 | 2.4 | FR-DIS-04, -05, -06 | I | Reference controller, fixture seeds | Generate a small bank twice with the same seed; restore each snapshot from both runs | (a) Same snapshot IDs and split membership in both runs; train/validation splits disjoint; trigger rate and non-firing count logged. (b) Restored states equal (state arrays compared after deserialization; the bank hash is computed over a canonical serialization of state arrays, not raw file bytes) | (a) Core; (b) Project check |
| T2.5 | 2.5 | FR-DIS-07 | I | the generated fixture bank with known outcomes | Run the labeller with N attempts | Labels match the known outcomes; "unknown recoverability" used, never "unrecoverable" | Core |
| T2.6 | 2.6 | FR-DIS-10 | U | Each policy type; the generated fixture bank | Restore a state; initialise the policy by the rule | For each policy type: internal state reset; pending action chunk empty; history equals stored history; next action is a fresh replan. A policy type that cannot be initialised by the rule is not a pass: it is recorded as a limitation, and its disturbed results are reported from the FR-EVAL-08 end-to-end suite instead | Core |
| T2.7 | 2.7 | FR-EVAL-08 | I | Fixture seeds | Run the end-to-end suite with online disturbance | Trigger rate reported; per-episode records include disturbance type and parameters | Core |
| T2.8 | 2.7b | FR-EVAL-04 | U | Scripted trajectories with known numbers of re-grasps (0, 1, 2) | Run the re-grasp counter | Counts equal the known values | Core |
| T2.9 | 2.9b | §7.3 | U | Synthetic data with known effect | (a) Training seeds resampled independently per variant; (b) episode IDs resampled jointly across variants; (c) ≥ 10,000 replicates; (d) verdict function returns confirmed / refuted / inconclusive correctly on constructed CIs, including the negligible-effect and non-inferiority rules; (e) CI of a known difference covers the true value at roughly the nominal rate over repeated synthetic draws | (a)–(d) hold; (e) coverage recorded | (a)–(d) Core; (e) Project check |
| T2.10 | 2.9d | FR-ENV-07 | U | Config | Sample from the unseen-arrangement distribution | All samples fall outside the training range and inside the declared numeric range | Core |
| T2.11 | 2.8 | §10 M2 review triggers | P | Validation bank, π₀ | E0 on validation; compute disturbed success and verified-recoverable fraction | Numbers recorded; triggers evaluated against the thresholds; any resulting design change recorded before the lock | Core |
| T2.12 | 2.9c | FR-RL-05, FR-DIS-05 | A | Selection log; training, tuning and selection run manifests produced before the lock | Check every selection decision cites train/validation data only; check that none of these pre-lock manifests references test seeds or the test bank. Post-lock evaluation manifests (M3 onward) are out of scope of this check and legitimately reference test data | No test references in pre-lock records | Core |
| T2.13 | 2.10, 2.11 | FR-DIS-05 | A | Git history; test-bank generation run manifest | Check: (a) the `design-lock` tag exists; (b) the tagged commit is an ancestor of the commit that adds the test-bank manifest and hash; (c) the generation run manifest records the lock commit as its code version and a start time after the tag; (d) the lock contains selected values (not only search spaces) for every Core configuration. Filesystem timestamps alone are not accepted as evidence | All four hold | Core |
| T2.14 | 2.11 | FR-DIS-05 | U | Training and tuning code paths | Attempt to load the test bank from training/tuning entry points | Load refused by the access guard | Project check (supports FR-DIS-05; the requirement itself is verified by T2.12 and T3.1) |
| T2.15 | 2.12 | NFR-COD-01, §10 M2 | S | All M2 code | Run the full unit-test suite (locally and in CI) | (a) Unit tests for triggers, disturbances, bank save/restore, recoverability labelling, metrics and FR-DIS-10 exist and pass; (b) the suite passes in CI | (a) Core; (b) Project check |
| T2.16 | 2.0, 2.11 | FR-EVAL-06 | A | Lock record; post-lock bank manifest | Check the video subset rule and recorder settings are in the lock; check the concrete subset IDs in the post-lock manifest follow that rule | Rule locked; IDs reproducible from the rule | Core |
| D2 | — | — | — | — | Drop + recovery-attempt video from a validation or fixture bank state | — | Non-gating |

---

## 5. M3 Core

| ID | Item | Req | Level | Prerequisites / fixtures | Procedure | Pass criterion | Gate |
|---|---|---|---|---|---|---|---|
| T3.1 | 3.4 | FR-RL-05 | A | Run manifests, lock record | Compare each final run's config with the lock | PASS only if every run matches the lock, or every deviation is documented with its cause and assessed in the report (and the affected results are labelled). Any unexplained deviation is FAIL | Core |
| T3.2 | 3.1–3.3 | FR-EVAL-03, FR-RL-07, §7.4 | A | Run table | Check ≥ 5 seeds launched per configuration, equal env-step budget across RL variants, failed runs retained, failed-run rate computed | All hold; shortfall below 5 completed runs raised for review | Core |
| T3.3 | 3.5 | FR-EVAL-01, -02, -04, -08, FR-BASE-01 | P | Locked configs, test bank, test seeds | One evaluation per selected configuration and for π₀ on Nominal (≥ 200), Disturbed (all test-bank states) and the end-to-end suite | Evaluation log shows exactly one test evaluation per configuration; per-episode fields complete | Core |
| T3.4 | 3.6 | §7.1, §7.3 | P | Raw logs, bootstrap code at the locked commit | Generate result tables by script | (a) Tables are generated by script from raw per-episode logs with the locked bootstrap code; failed-run rate and per-seed values beside each CI; S_dist(verified) denominator shown. (b) A second run of the script yields identical tables | (a) Core; (b) Project check |
| T3.5 | 3.7 | FR-TRK-03, -04 | A | Figures | Check headline figure axes, CI bars, π₀ marker; learning curves per variant | All present | Core |
| T3.5b | 3.7b | FR-EVAL-06 | A + I | Videos, metadata, post-lock subset IDs | (a) Coverage: one video for every (policy, subset entry) pair, for π₀ and every final variant; metadata links each video to its episode record. (b) Each video opens and decodes, frame count consistent with its episode length and frame interval, frames not all identical. (c) Bank-based videos of the same subset entry share the same bank-state ID; end-to-end videos share the nominal seed only | (a) and the existence of readable videos in (b): Core. Decoding thresholds and the non-identical-frames check: Project check | Core / Project check |
| T3.6 | 3.8 | §7.4, NFR-CMP-02 | A | Cost logs, M1 projection | Compare total Core compute and each part with the projection | Trigger status recorded | Core |
| T3.7 | 3.9 | §7.3 | P | Result tables | Apply the verdict function to H1a, H1b, H2a, H2b | Verdicts recorded with CIs; code commit equals the locked bootstrap commit | Core |
| D3 | — | — | — | — | Headline plot and side-by-side videos from T3.5b | — | Non-gating |

---

## 6. M4 Extended

Each test is **Extended**: it runs only if the corresponding experiment is approved; otherwise its outcome is NOT RUN.

| ID | Item | Req | Level | Procedure | Pass criterion |
|---|---|---|---|---|---|
| T4.1 | 4.1 | FR-RL-04, -05 | U + A | Unit test for each new mitigation hook (e.g. frozen layers receive no gradient; regularizer is zero when π = π₀; auxiliary loss uses only original demonstrations); audit validation-only selection | Hooks behave as specified; no test references in selection |
| T4.2 | 4.2 | FR-BASE-02 | A | Check that the budget record commit precedes demonstration collection; demos come from train-bank states only; resources accounted separately | All hold |
| T4.3 | 4.3 | FR-EVAL-01, FR-DIS-09 | P | Evaluate on the locked unseen-arrangement distribution and held-out magnitudes | One evaluation per configuration; results with CIs |
| T4.4 | M4 tuning rule | FR-DIS-05 | A | Lock addendum committed before any Extended test evaluation | Ordering holds |
| T4.5 | 4.4 | §7.3 | P | H3 verdict, only if E4b ran | Verdict recorded or NOT RUN |

---

## 7a. FR3 track

F0 tests are not gates for the Core milestones. A BLOCKED F0 step is a valid outcome that feeds the F1 path decision. F1 tests apply once F1 starts. The research-comparison tests (TF1.5, TF1.6, and the verdict part of TF1.7) apply only if the recorded F1 scope is a research comparison (FR-FR3-07), and are otherwise NOT RUN.

| ID | Item | Req | Level | Procedure | Pass criterion | Gate |
|---|---|---|---|---|---|---|
| TF0.1 | F0.0, F0.4 | FR-FR3-01 | A | Check `ARTIFACTS.md` entries for the FR3 model and hand model | All FR-ART-02 fields present and verified | F0 |
| TF0.2 | F0.1 | FR-FR3-02a | I | Load the model; list joints, limits and actuators; render | Inspection table produced; video readable with metadata | F0 |
| TF0.3 | F0.2 | FR-FR3-02b | I | Command each configuration of a fixed set; record final joint error | Errors recorded; video readable. A tolerance, if set, is a project check | F0 (record); Project check (tolerance) |
| TF0.4 | F0.3 | FR-FR3-02c | U + I | (a) FK unit test against known poses (e.g. the `home` keyframe vs model site positions); (b) reach a fixed target set; record end-effector position error | (a) passes; (b) errors recorded; video readable | F0 (record); Project check (FK test, error tolerance) |
| TF0.5 | F0.4 | FR-FR3-02d | I | Attach or use the hand; run the fixed grasp-and-lift trials | Grasp success rate recorded; video readable; BLOCKED with cause if no hand can be attached | F0 |
| TF0.6 | F0.5 | FR-FR3-03 | A | Check the decision record compares paths (i)–(iii) with evidence, effort and compute estimates and risks | All elements present | F0 |
| TF1.1 | F1.1 | FR-FR3-04 | U + A | (a) Unit tests of the success predicate on hand-built states (all sorted / one misplaced / object dropped); (b) the task-definition commit precedes the first demonstration or policy commit | Both hold | F1 |
| TF1.2 | F1.2 | — | I | Load the sorting scene through the task interface; run a random-action episode; record video | Episode runs; video readable | Project check |
| TF1.3 | F1.3, F1.4 | FR-FR3-01 | A + I | Per path: (i)/(ii) demonstration provenance recorded and generator or retargeting success rate measured on a fixed seed set; (iii) existing checkpoint verified in `ARTIFACTS.md` with a compatibility mapping | Elements for the chosen path complete | F1 |
| TF1.4 | F1.4 | FR-FR3-07 | P | FR3 base-policy nominal evaluation on fixed FR3 evaluation seeds with 95% CI: stated episode count (demonstration) or ≥ 200 episodes (research comparison); π₀ hash | Result with CI recorded; BLOCKED with cause if no base policy was obtained | F1 |
| TF1.5 | F1.5, F1.6b, F1.6c | FR-FR3-05 | A | FR3 train/validation banks exist before the lock and no test bank exists before it; splits disjoint; FR3 lock tag is an ancestor of the commit adding the FR3 test-bank manifest | Ordering and separation hold | F1 (research comparison) |
| TF1.6 | F1.6, F1.6b | FR-FR3-05 | A | Pre-lock FR3 selection records reference FR3 validation data only; no Core-locked value used without FR3 re-selection | Holds | F1 (research comparison) |
| TF1.7 | F1.7 | FR-FR3-05, §7.3 | P | One test evaluation per selected configuration; §7.3 statistics; demonstration-only results labelled, not given verdicts | Holds | F1 |
| TF1.8 | F1.8 | FR-FR3-06 | A + I | Videos of each policy obtained, readable, metadata complete; before/after videos on the same fixed FR3 states when both a base and a fine-tuned policy exist | Coverage complete for the policies that exist; missing policies recorded as BLOCKED or negative outcomes | F1 |

---

## 7. M5 Report and reproducibility

| ID | Item | Req | Level | Procedure | Pass criterion | Gate |
|---|---|---|---|---|---|---|
| T5.1 | 5.1 | NFR-LIC-01 | A | For each released file, check redistribution terms in `ARTIFACTS.md` | All released files permitted | Core |
| T5.2 | 5.2 | FR-TRK-06 | R | From a clean checkout with documented setup, run the figure regeneration script on released logs | All figures and tables regenerate; reported numbers match within the predeclared numerical tolerance (bitwise identity not required) | Core |
| T5.3 | 5.3 | NFR-REP-02 | R | Rerun a predeclared subset (e.g. π₀ and one variant, E0 + one Core comparison) and compare with originals by the predeclared paired comparison and tolerance | The comparison and tolerance are declared before the rerun; the result (agreement or disagreement) is reported. Bitwise identity is not required; known nondeterminism is documented | Core |
| T5.4 | 5.3 | NFR-REP-03 | R | Replay bank states on a second backend; compare E0 | Differences measured and reported | Non-gating (SHOULD) |
| T5.5 | 5.5 | FR-TRK-06 | A | Deliverables audit against FR-TRK-06 and the traceability appendix | Every mandatory (Core) deliverable present and complete; optional work (Extended, Stretch) present or explicitly listed as not done. A missing or incomplete mandatory deliverable is FAIL | Core |

---

## 8. Requirements without a test

| Requirement | Reason |
|---|---|
| FR-ENV-01 (MuJoCo only) | Design constraint; checked in design review |
| NFR-PORT-01 (platform-independent Python) | Code-review checklist item |
| NFR-CMP-01 | Permission, not a behaviour |
| FR-RL-08, E6 | Stretch; not planned |
