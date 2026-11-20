# M0 nominal evaluation protocol (draft for review)

Status: **approved by the project lead on 2026-11-20** for the first M0 evaluation. Commit this protocol before the first policy evaluation. No policy result has been measured under it. This protocol covers Path 2 (FR-POL-03, T0.9); it does not claim reproduction of an image-based checkpoint.

## Purpose and separation

Measure the nominal Can success rate of the pretrained state-based policy before any fine-tuning. This is an M0 development evaluation, not the final M3 test suite. Use a dedicated nominal validation seed set; keep fixture seeds and final test seeds disjoint. Never use the final test bank or final test seeds for this protocol. Record any use of the M0 validation results for subsequent decisions.

## Procedure

1. Before evaluation, freeze and SHA-256 hash the exact checkpoint and normalization file, record their paths, training config, data and code revisions, and choose model or EMA weights. Use the final EMA checkpoint; any change requires a recorded decision before evaluation.
2. Evaluate 200 episodes with environment reset seeds 100000–100199 inclusive, one episode per seed, in order. Reservations: fixture seeds 0–999, M0 validation 100000–100199, and final test 200000–200199; these ranges are disjoint and the final test range is reserved until M3. Demonstrations are recorded states, not reset seeds. Apply the policy sampling seed (environment seed plus 1000000) to PyTorch only, never NumPy, whose global RNG sets object placements; record both seeds. Verify the harness respects this separation before evaluation. Use one environment per process to avoid interleaving the shared RNG.
3. Use the documented Can task configuration (Panda, OSC_POSE, 20 Hz, horizon 300) and the state observation ordering, normalization and action chunking specified by the checkpoint's config. Primary inference: one flow integration step, one observation, and all four predicted actions executed per chunk; clip normalized actions to [-1, 1] and unnormalize with the frozen statistics. Verify this mapping against the upstream implementation before the run. Report inference device, software versions and deviations; GPU kernels may be nondeterministic. Any other step count is secondary and cannot select the primary result. Stop an episode when the benchmark success predicate is true or the horizon is reached. A horizon without success is a failure. Stopping at first success preserves the any-step success metric, though episode lengths differ from an evaluator that continues.
4. Log one record per seed with success, steps and any error, plus a run manifest with hardware, backend, versions, checkpoint and normalization hashes, config, code and submodule revisions, seed list, start/end time and wall-clock. Do not silently replace failed episodes or seeds. If an infrastructure error prevents completion, retain it, report the number completed and the cause, and mark the protocol incomplete; agree on a recovery before restarting.
5. For completed episodes report successes and denominator, success proportion and a two-sided 95% Wilson binomial interval. The interval describes episode variability for this fixed checkpoint and evaluation distribution, not variation across training runs. Report all episode outcomes and failures; do not use a 95% interval as proof of equal performance.

## Decision rule to approve before the run

**Operational readiness threshold:** at least 90% success (180/200) on the fixed M0 validation sample, with the Wilson interval reported. This is a project choice for deciding whether the policy is a useful starting point, not a published reproduction target or a requirement in FR-POL-03. Below threshold, report the result and review the data conversion, training and compute before deciding whether to retry or revisit simulator compatibility. Reaching it does not prove disturbance robustness. Any retry, new seed set, or change in threshold must be recorded as a new decision before that run.

Path 2 has no verified released state-based Can figure to compare against; do not claim agreement with the image-policy result. If a genuinely comparable released result is identified later, define its comparison and tolerance before examining this protocol's outcomes.

## Evidence and open approvals

- Data: the verified Can demonstration source and conversion provenance in `ARTIFACTS.md`; train-only normalization and held-out demonstration loss are reported separately from nominal episode success.
- Report checkpoint hash, fixed seeds, full per-episode JSONL, proportion, interval, actual compute and deviations in `reports/M0.md` (T0.9). Keep failed runs and the original artifacts.
- **Before running:** obtain implement and verify seed separation and the inference mapping; commit the approved protocol. A protocol execution and its outcome are distinct from meeting the optional readiness threshold.
