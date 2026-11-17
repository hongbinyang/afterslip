# Afterslip development roadmap

Status: draft. The [requirements](requirements.md) (v1.1 draft) govern scope and acceptance. Task, artifacts, algorithm and mitigation remain candidates until the M0 checks. Results may be negative or inconclusive. This roadmap is the one-page overview; work items, decision points and completion checks are in [milestones.md](milestones.md), and verification procedures are in [test-plan.md](test-plan.md).

## Roadmap

| Milestone | Outcome | Depends on | Acceptance (requirements §10) |
|---|---|---|---|
| M0 Verify | Licensed, compatible artifacts and a reproducible starting policy π₀; candidate controllers and Core mitigation selected; environment and video recording working | — | Artifact register; decision-table path; reproduction or pretraining result; Linux CPU smoke test passed or BLOCKED with an approved deviation; novelty finding; M0 report |
| M1 Fine-tune | Nominal-only RL fine-tuning (E1) running end to end; first validation-based selection; shared RL step budget proposed from validation curves; compute projection | M0 | 5 seeds launched, failures kept, up to 8 launches; CIs over completed seeds; compute recorded; effect reported whatever its sign |
| M2 Bank + design lock | Disturbances and train/validation banks; recoverability labels; reconstruction tests; E0 on validation; **all validation-based tuning and selection for every Core configuration**; design lock with selected values; test bank and test video subset generated after the lock | M1 | Unit tests pass; trigger rate and verified-recoverable fraction reported; lock committed before the test bank exists |
| M3 Core | Locked configurations trained (no further tuning); one test evaluation per configuration; final evaluation videos; H1/H2 verdicts | M2 lock | Headline figure with CIs; H1 and H2 verdicts by §7.3 |
| M4 Extended | Approved Extended experiments only (E3b, E4b, E5), under the same validation-only selection and lock rules | M3 | M3 standard for each completed experiment |
| M5 Report | Reproducible deliverables and findings, including negative or inconclusive results | M3 (and M4, F1 if run) | All FR-TRK-06 deliverables present |

**FR3 track** (requirements §6.9), alongside Core:

| Phase | Outcome | Depends on | Acceptance |
|---|---|---|---|
| F0 FR3 feasibility | Franka Research 3 learned and demonstrated step by step (model, joint control, kinematics, hand and grasp), each with a video; F1 path decision with estimates | Environment (M0 0.0b); runs alongside M0–M2; not a gate for Core milestones; costs and schedule impact recorded and reviewed | Each step recorded or BLOCKED with cause; sources verified; path decision; F0 report |
| F1 FR3 sorting | Three-bin FR3 sorting scene; FR3 base policy by the chosen path; the Afterslip method applied within the chosen scope (demonstration or research comparison); videos | M3 accepted; F0 path decision; scope decision (FR-FR3-07) | Task defined before data; each policy obtained evaluated with CI (demonstration: stated episode count; research comparison: FR-FR3-05 protocol, ≥ 200 episodes); videos of each policy, before/after when both exist; blocked or negative outcomes recorded; F1 report |

## Cross-cutting controls

- Train and validation data only for design, tuning and selection; the test bank is generated and hashed after the M2 lock and evaluated once per selected configuration.
- Records for every run: episode IDs, per-seed results, failures, costs, artifact hashes, configuration and environment.
- Statistics: the hierarchical bootstrap of requirements §7.3 (shared episode IDs resampled jointly, independent training seeds resampled independently) unless a pairing is explicitly justified. The end-to-end disturbed suite is always reported alongside bank-based results.
- Equal RL environment-step budgets across compared variants; demonstration resources accounted separately.
- Videos with metadata from M0 onward as evidence and demonstration; the FR-EVAL-06 final videos are recorded on a fixed test video subset after the lock.

## Related documents

| Document | Content |
|---|---|
| [requirements.md](requirements.md) | What to build and how results are judged (authoritative) |
| [milestones.md](milestones.md) | Work items, decision points, deliverables and completion checks per milestone; requirements traceability |
| [test-plan.md](test-plan.md) | Test levels, gates, procedures, pass criteria and evidence per milestone |
| [setup.md](setup.md) | Platforms, technology stack, versions, package management, tested configurations |
| [explainer.md](explainer.md) | The project story, illustrated |
