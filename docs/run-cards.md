# Run cards

Long runs are started by hand on a local machine. Each card gives the exact commands, the expected duration, how to follow progress and stop safely, and what the run writes. Every behaviour below was checked against the code at the time of writing; the check date is given per card.

**Every run records itself** (`afterslip/runs.py`): its run folder gets

- `manifest.json` at the start: git commit and branch, whether the checkout was modified, the upstream submodule's commit, the command, the resolved configuration, device, Python and package versions, machine;
- `console.log`: everything the run prints, as it happens;
- `summary.json` at the end: `status` (`complete`, `interrupted` or `failed`, with the error), duration and the result; for a run that stopped early, the progress reached so far.

Nothing needs to be copied from the terminal: when a run has finished, say so in the room, and the results are read from the run folder.

## Pretraining the state-based Can policy (milestone 0.3d)

Checked 2026-11-22 on an Apple M2 Pro (macOS 26.6), with short runs of the same command; the full run took 50 min 48 s on 2026-11-20.

**Before the first run** (once per checkout):

```bash
git pull --ff-only
git submodule update --init
make sync
make data-can          # downloads 47 MB once, checks its sha256, regenerates the observations
make test              # expected: all pass, 1 xfailed
```

**Start** (from the repository root, in the `afterslip` environment):

```bash
make pretrain-can DEVICE=mps
```

`DEVICE=cpu` also works (about 4 times slower). The configuration is `configs/pretrain/can_reflow_mlp_state.yaml`; any value can be overridden for a trial run, e.g. `python -m afterslip.upstream.pretrain configs/pretrain/can_reflow_mlp_state.yaml device=mps train.n_epochs=5`.

**Expected duration:** about 50 minutes on Apple MPS and about 4 hours on CPU (setup.md §6.6).

**Progress:** a progress bar `n/2000 [elapsed<remaining, s/it]` and, for every batch, `epoch: e/2000=…%, steps: s, loss: …`. In the full run the mean loss went from 1.28 (first epoch) to 0.20 (epoch 100) and 0.127 (last epoch).

**Stop safely:** press Ctrl-C once. The run stops within seconds with `KeyboardInterrupt` (exit code 130) and leaves no processes behind; `summary.json` records `"status": "interrupted"`. The checkpoint of the last finished epoch stays at `checkpoint/last.pt`. An interrupted run is not resumed: start the command again for a fresh run in a new folder.

**Run folder** (all under the git-ignored `outputs/`; nothing is committed):

```
outputs/reinflow/log/robomimic/pretrain/can_pre_reflow_mlp_state_ta4_td100/<YYYY-MM-DD_HH-MM-SS>_42/
  manifest.json, console.log, summary.json   the run's own record
  checkpoint/state_<epoch>.pt                every 500 epochs and at the end (model, EMA, optimizer, scheduler)
  checkpoint/last.pt                         after every epoch; the final one is the policy to evaluate
  heldout_validation.json                    held-out demonstration loss (also in summary.json)
  architecture.log                           network summary
  render/                                    empty (in-training evaluation is off)
```

At the end the launcher prints the held-out report and the run folder. The held-out loss shows how well the policy imitates unseen demonstrations; **it is not a task success rate**. Runs started before 2026-11-22 have no `manifest.json`, `console.log` or `summary.json`.

## Evaluating a pretrained policy (milestone 0.3d, protocol in m0-evaluation-protocol.md)

Checked 2026-11-22 on an Apple M2 Pro with a throwaway 1-epoch checkpoint and trial runs on fixture seeds; the real policy has not been evaluated in any check.

**Start** (from the repository root, in the `afterslip` environment; `RUN` is the pretraining run folder):

```bash
make eval-can RUN=outputs/reinflow/log/robomimic/pretrain/can_pre_reflow_mlp_state_ta4_td100/<run folder>
```

For the policy pretrained on 2026-11-20, whose run predates run records, the expected normalization hash must be given (see below):

```bash
make eval-can RUN=outputs/reinflow/log/robomimic/pretrain/can_pre_reflow_mlp_state_ta4_td100/2026-11-20_16-13-03_42 \
  NORM_SHA=28ed3a099f45d596926f1af2e379cd388d94396371cd48e3006675a48118d46e
```

**Normalization check:** before any seed is used, the harness compares the sha256 of the normalization file it is about to use with the one the policy was trained with: recorded in the pretraining run's `summary.json` (runs since 0.3b) or given as `NORM_SHA`. A mismatch, or a protocol run with no expected hash, stops immediately; no run folder is created.

This runs the approved protocol exactly: 200 episodes on seeds 100000–100199, the checkpoint's final EMA weights, one flow integration step, four actions per chunk, CPU. `EPISODES=5` makes a **trial** instead: the first 5 fixture seeds (0–4), never validation seeds, and no pass/fail verdict.

**Expected duration:** at most about 8 minutes (an episode that fails runs the full 300 steps, about 2.2 s; successful episodes end earlier).

**Progress:** one line per episode, e.g. `episode 17/200 seed 100016: success in 143 steps; successes so far 15/17`.

**Stop safely:** Ctrl-C once. Checked: the run stops with exit code 130, no processes remain, `episodes.jsonl` keeps every finished episode, and `summary.json` records `"status": "interrupted"` with the episodes completed so far. By the protocol, an interrupted or incomplete run is reported as incomplete; agree on a recovery in the room before starting again.

**If an episode raises an error:** the run stops at that episode, keeps its record in `episodes.jsonl`, and `summary.json` records `"status": "failed"` with the error, the progress and no verdict (protocol step 4). Report it in the room; a recovery is agreed before any rerun.

**Run folder:** `outputs/eval/can-nominal/<YYYY-MM-DD_HH-MM-SS>/`, always new: a run refuses to start in a folder that already contains files, so earlier results are never overwritten.

```
manifest.json    the run's record, including the checkpoint and normalization file sha256
console.log      everything printed
episodes.jsonl   one line per seed (design.md §7): episode_id, seed, policy_seed, success, steps,
                 timeout, policy_hash (checkpoint sha256), git_commit, error, ...
summary.json     successes, rate, 95% Wilson interval, "passed" (at least 180 of 200), settings
```

**Normalization provenance for the 2026-11-20 policy (inferred, not recorded by that run):** the pretraining run predates run records, so it did not capture which normalization file it used; the expected hash above is an inference. Checked 2026-11-22: `outputs/reinflow/data/robomimic/can-state/normalization.npz` was written by `make data-can` at 16:12:13, 50 s before the run started, and not modified since; regenerating it from the pinned dataset gives identical arrays (the conversion is deterministic). Its sha256 is `28ed3a099f45d596926f1af2e379cd388d94396371cd48e3006675a48118d46e`, the `NORM_SHA` above; the evaluation enforces it and records the provenance as "given on the command line; not recorded by the pretraining run". Do not run `make data-can` again before this evaluation: the arrays would be identical, but the file and its hash would change.

At the end it prints, e.g., `Successes 187/200 = 93.5%, 95% Wilson interval 89.2%-96.2%: PASS`. Below 180 it prints `BELOW THRESHOLD (review per protocol)`: the protocol's review then starts; nothing is re-run or changed before that review.
