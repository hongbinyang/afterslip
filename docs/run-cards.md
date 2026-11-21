# Run cards

Long runs are started by hand on a local machine. Each card gives the exact commands, the expected duration, how to follow progress and stop safely, what the run writes, and what to report back. Every behaviour below was checked against the code at the time of writing; the check date is given per card.

## Pretraining the state-based Can policy (milestone 0.3d)

Checked 2026-11-20 on an Apple M2 Pro (macOS 26.6), with short runs of the same command.

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
make pretrain-can DEVICE=mps 2>&1 | tee outputs/pretrain-can.log
```

`DEVICE=cpu` also works (slower). The configuration is `configs/pretrain/can_reflow_mlp_state.yaml`; any value can be overridden for a trial run, e.g. `python -m afterslip.upstream.pretrain configs/pretrain/can_reflow_mlp_state.yaml device=mps train.n_epochs=5`.

**Expected duration:** about 1 hour on Apple MPS and about 4 hours on CPU (setup.md §6.6: measured over 20 epochs, so an estimate; checkpoint saves and the held-out scoring add a little).

**Progress:** a progress bar `n/2000 [elapsed<remaining, s/it]` and, for every batch, `epoch: e/2000=…%, steps: s, loss: …`. The loss starts near 1.3 and was about 0.35–0.42 after 20–25 epochs in the checks; it should keep falling slowly. The same lines go to `outputs/pretrain-can.log` through `tee`.

**Stop safely:** press Ctrl-C once. The run stops within seconds with `KeyboardInterrupt` (exit code 130) and leaves no processes behind. The checkpoint of the last finished epoch stays at `<run folder>/checkpoint/last.pt`; no held-out report is written. An interrupted run is not resumed: start the command again for a fresh run (the old folder stays and can be deleted).

**What it writes** (all under the git-ignored `outputs/`; nothing is committed):

```
outputs/reinflow/log/robomimic/pretrain/can_pre_reflow_mlp_state_ta4_td100/<YYYY-MM-DD_HH-MM-SS>_42/
  architecture.log            network summary
  render/                     empty (used by upstream only for in-training evaluation, which is off)
  checkpoint/state_<epoch>.pt every 500 epochs and at the end (model, EMA weights, optimizer)
  checkpoint/last.pt          after every epoch; the final one is the policy to evaluate
  heldout_validation.json     written at the end (see below)
```

**At the end** the launcher scores `checkpoint/last.pt` on the 20 held-out demos and prints, and saves as `heldout_validation.json`: the run folder (`logdir`), the checkpoint's `checkpoint_sha256`, and the held-out loss of the `model` and `ema` weights.

**Report back to the room:**

1. the printed JSON block (run folder, `checkpoint_sha256`, both held-out losses);
2. the last `epoch: 1999/2000 …, loss: …` line;
3. the elapsed time from the progress bar and the `DEVICE` used.

The checkpoint file itself stays on the machine; its sha256 identifies it for the evaluation (protocol: docs/m0-evaluation-protocol.md, final EMA weights).

**If something looks wrong** (an error, a loss that rises steadily or becomes `nan`, no progress for several minutes): press Ctrl-C, keep `outputs/pretrain-can.log`, and paste its last 30 lines into the room.
