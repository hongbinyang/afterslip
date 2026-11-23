"""M0 nominal evaluation of a pretrained state-based policy (docs/m0-evaluation-protocol.md).

Protocol (approved 2026-11-20): 200 episodes, reset seeds 100000-100199 in order, one episode
per seed; final EMA weights; one flow integration step; all 4 actions of a chunk executed;
normalized actions clipped to [-1, 1] and un-normalized with the frozen statistics; policy
noise from a torch generator seeded with (reset seed + 1000000), never NumPy's global RNG,
which places the objects; an episode ends at the benchmark's success or at the horizon (300).
Pass: at least 180 successes out of 200, reported with a 95% Wilson interval.

Every run writes its own record into its folder (afterslip.runs): manifest.json,
console.log, episodes.jsonl (one line per seed, written as it finishes) and summary.json.
An episode that raises stops the run: summary.json records "failed" with the progress so
far and no verdict (protocol step 4). A run never writes into a non-empty folder.

Before any seed is used, the normalization file must have the sha256 the policy was trained
with: recorded by the pretraining run, or (for runs before 0.3b) given with
--normalization-sha256; a protocol run without an expected hash, or with a mismatch, stops.

Normalization: the statistics come from the current data folder (`make data-can`) and their
sha256 is recorded. The conversion is deterministic, so a regeneration gives identical
statistics. For the pretraining run of 2026-11-20, which did not record its normalization
file, this was checked: identical arrays, and the file was written before the run started
and not changed since (docs/run-cards.md).

    python -m afterslip.evaluation.nominal select [--run <folder>] [--normalization-sha256 H]
    python -m afterslip.evaluation.nominal                  # evaluate the selected policy
    python -m afterslip.evaluation.nominal --run <folder>   # evaluate another run explicitly
    python -m afterslip.evaluation.nominal --episodes 5     # trial: fixture seeds, no verdict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from afterslip import seeds as seed_ranges
from afterslip.data.can_demos import observation_vector
from afterslip.runs import REPO_ROOT, RunRecord, code_revision

SEEDS = tuple(seed_ranges.M0_VALIDATION)
# Trial runs use fixture seeds, so the validation episodes stay unseen until the real run.
TRIAL_SEEDS = tuple(seed_ranges.FIXTURE)
POLICY_SEED_OFFSET = 1_000_000
PASS_SUCCESSES = 180
WEIGHTS = "ema"
INFERENCE_STEPS = 1
CONFIG = REPO_ROOT / "configs" / "pretrain" / "can_reflow_mlp_state.yaml"
DEFAULT_OUT = REPO_ROOT / "outputs" / "eval" / "can-nominal"
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Policy:
    """The pretrained flow policy with the protocol's inference settings."""

    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        import hydra.utils
        import torch

        from afterslip.upstream.pretrain import load_config

        self.cfg = load_config(CONFIG, [f"device={device}"])
        self.device = device
        self.model = hydra.utils.instantiate(self.cfg.model)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model.load_state_dict(state[WEIGHTS])
        self.model.eval()
        stats = np.load(self.cfg.normalization_path)
        self.obs_min, self.obs_max = stats["obs_min"], stats["obs_max"]
        self.act_min, self.act_max = stats["action_min"], stats["action_max"]
        self.horizon_steps = int(self.cfg.horizon_steps)
        self.action_dim = int(self.cfg.action_dim)

    def chunk(self, raw_obs: dict, generator) -> np.ndarray:
        """Actions (horizon_steps, action_dim) in environment units for one observation."""
        import torch

        state = observation_vector(raw_obs)
        # Same mapping as upstream's robomimic_lowdim wrapper; observations are not clipped.
        state = 2 * ((state - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)
        cond = {"state": torch.as_tensor(state, dtype=torch.float32).reshape(1, 1, -1)}
        z = torch.randn((1, self.horizon_steps, self.action_dim), generator=generator)
        with torch.no_grad():
            sample = self.model.sample(
                {k: v.to(self.device) for k, v in cond.items()},
                inference_steps=INFERENCE_STEPS,
                clip_intermediate_actions=True,
                z=z.to(self.device),
            )
        action = sample.trajectories[0].cpu().numpy().clip(-1, 1)
        return (action + 1) / 2 * (self.act_max - self.act_min) + self.act_min


class EpisodeError(RuntimeError):
    """An episode raised: the evaluation stops and is recorded as failed, without a verdict."""


@dataclass
class Episode:
    """One line of episodes.jsonl: the design §7 record plus the protocol's policy seed."""

    episode_id: str
    suite: str
    seed: int
    policy_seed: int
    bank_state_id: str | None  # None for nominal episodes
    policy_hash: str  # sha256 of the evaluated checkpoint
    success: bool
    steps: int
    timeout: bool  # the horizon was reached without success
    video: str | None
    git_commit: str | None
    error: str | None = None


def run_episode(task, policy: Policy, seed: int, suite: str, policy_hash: str, commit) -> Episode:
    import torch

    policy_seed = seed + POLICY_SEED_OFFSET
    generator = torch.Generator().manual_seed(policy_seed)  # never NumPy's global RNG

    def record(success: bool, timeout: bool, error: str | None = None) -> Episode:
        return Episode(
            f"{suite}-{seed}",
            suite,
            seed,
            policy_seed,
            None,
            policy_hash,
            success,
            steps,
            timeout,
            None,
            commit,
            error,
        )

    steps = 0
    try:
        raw = task.reset(seed=seed)
        while True:
            for action in policy.chunk(raw, generator):
                result = task.step(action)
                steps += 1
                raw = result.observation
                if result.success or result.timeout:
                    return record(bool(result.success), bool(result.timeout))
    except Exception as error:  # recorded, never silently replaced (protocol step 4)
        return record(False, False, f"{type(error).__name__}: {error}")


class ProvenanceError(RuntimeError):
    """The normalization file is not the one the policy was trained with (or is unknown)."""


def expected_normalization(run_dir: Path, given: str | None) -> tuple[str | None, str]:
    """The normalization sha256 the policy was trained with, and where that value comes from.

    Pretraining runs since 0.3b record it in their summary.json. Older runs did not; for those
    the expected value is given on the command line, and the source says it was inferred.
    """
    summary = Path(run_dir) / "summary.json"
    recorded = None
    if summary.is_file():
        recorded = (json.loads(summary.read_text()).get("result") or {}).get("normalization_sha256")
    if recorded and given and recorded != given:
        raise ProvenanceError(f"given hash {given} differs from the recorded one {recorded}")
    if recorded:
        return recorded, "recorded by the pretraining run"
    if given:
        return given, (
            "given on the command line; not recorded by the pretraining run "
            "(inferred from file times and a deterministic regeneration, docs/run-cards.md)"
        )
    return None, "unverified"


def finished_pretraining_runs(log_root: Path | None = None) -> list[Path]:
    """Finished pretraining runs of the state-based Can policy, oldest first.

    Finished means: `checkpoint/last.pt` exists and summary.json says "complete", or, for
    runs from before run records, the held-out report (written at the very end) exists.
    Run folders are named by start time, so sorting by name sorts by start.
    """
    import os

    from afterslip.upstream import configure_reinflow

    configure_reinflow()
    root = log_root or Path(os.environ["REINFLOW_LOG_DIR"]) / "robomimic" / "pretrain"
    finished = []
    for run in root.glob("can_pre_reflow_mlp_state_*/*/"):
        if not (run / "checkpoint" / "last.pt").is_file():
            continue
        summary = run / "summary.json"
        if summary.is_file():
            if json.loads(summary.read_text()).get("status") == "complete":
                finished.append(run)
        elif (run / "heldout_validation.json").is_file():
            finished.append(run)
    return sorted(finished, key=lambda run: run.name)


class SelectionError(RuntimeError):
    """The policy to select is not unambiguous; RUN must name it."""


SELECTION = DEFAULT_OUT / "selected-policy.json"  # local, git-ignored


def select_policy(
    run: Path | None = None,
    normalization_sha256: str | None = None,
    path: Path | None = None,
    replace: bool = False,
    checkpoint_sha256: str | None = None,
    log_root: Path | None = None,
) -> dict:
    """Record the policy the protocol evaluates: run folder and expected hashes.

    The protocol fixes the checkpoint before evaluation, so the evaluation never picks a
    run by itself: this explicit, checked record is its default. Without `run`, a selection
    is made only when exactly one finished pretraining run exists; with several, the run
    must be named. `checkpoint_sha256`, when given, must match the checkpoint. An existing
    selection is only replaced with `replace`.
    """
    from afterslip.upstream.pretrain import load_config

    path = path or SELECTION
    if path.is_file() and not replace:
        current = json.loads(path.read_text())
        raise FileExistsError(
            f"a policy is already selected: {current['run']} "
            f"(checkpoint {current['checkpoint_sha256']}); use --replace to change it"
        )
    if run is None:
        candidates = finished_pretraining_runs(log_root)
        if not candidates:
            raise FileNotFoundError("no finished pretraining run; run make pretrain-can")
        if len(candidates) > 1:
            listing = "\n".join(f"  {c}" for c in candidates)
            raise SelectionError(
                f"several finished pretraining runs; name one with RUN=:\n{listing}"
            )
        run = candidates[0]
    run = Path(run)
    checkpoint = run / "checkpoint" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} does not exist")
    actual_checkpoint = sha256(checkpoint)
    if checkpoint_sha256 is not None and actual_checkpoint != checkpoint_sha256:
        raise ProvenanceError(
            f"{checkpoint} has sha256 {actual_checkpoint}, expected {checkpoint_sha256}"
        )
    expected, source = expected_normalization(run, normalization_sha256)
    if expected is None:
        raise ProvenanceError(
            f"{run} does not record its normalization hash; select it with "
            "--normalization-sha256 (make select-policy NORM_SHA=...), see docs/run-cards.md"
        )
    actual = sha256(load_config(CONFIG).normalization_path)
    if actual != expected:
        raise ProvenanceError(f"normalization file has sha256 {actual}, expected {expected}")
    selection = {
        "run": str(run),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_checkpoint,
        "checkpoint_sha256_verified": checkpoint_sha256 is not None,
        "weights": WEIGHTS,
        "normalization_sha256": expected,
        "normalization_provenance": source,
        "selected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    print(json.dumps(selection, indent=2))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Selected for evaluation; recorded in {path}")
    return selection


def load_selection(path: Path | None = None) -> dict:
    path = path or SELECTION
    if not path.is_file():
        raise ProvenanceError(
            "no policy selected for evaluation; run make select-policy first "
            "(or give RUN=<pretraining run folder>), see docs/run-cards.md"
        )
    return json.loads(path.read_text())


def evaluate(
    run_dir: Path,
    out_dir: Path,
    episodes: int | None = None,
    device: str = "cpu",
    normalization_sha256: str | None = None,
    run_chosen_by: str = "given",
    expected_checkpoint: str | None = None,
):
    from afterslip.tasks.robosuite_can import CanTaskConfig, RobosuiteCanTask

    seeds = SEEDS if episodes is None else TRIAL_SEEDS[:episodes]
    is_protocol = episodes is None
    checkpoint = Path(run_dir) / "checkpoint" / "last.pt"
    policy = Policy(checkpoint, device=device)
    policy_hash = sha256(checkpoint)

    # Checked before any run folder is created or any seed is used.
    if expected_checkpoint is not None and policy_hash != expected_checkpoint:
        raise ProvenanceError(
            f"{checkpoint} has sha256 {policy_hash}, but the selected policy is "
            f"{expected_checkpoint}; the checkpoint changed since it was selected"
        )
    actual = sha256(policy.cfg.normalization_path)
    expected, source = expected_normalization(Path(run_dir), normalization_sha256)
    if expected is None and is_protocol:
        raise ProvenanceError(
            "the pretraining run does not record its normalization hash; pass the expected one "
            "(--normalization-sha256, make eval-can NORM_SHA=...), see docs/run-cards.md"
        )
    if expected is not None and actual != expected:
        raise ProvenanceError(
            f"normalization file {policy.cfg.normalization_path} has sha256 {actual}, "
            f"expected {expected} ({source}); it is not the file the policy was trained with"
        )
    settings = {
        "protocol": "docs/m0-evaluation-protocol.md",
        "protocol_run": is_protocol,
        "pretraining_run": str(run_dir),
        "pretraining_run_chosen_by": run_chosen_by,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": policy_hash,
        "normalization": str(policy.cfg.normalization_path),
        "normalization_sha256": actual,
        "normalization_expected_sha256": expected,
        "normalization_provenance": source,
        "weights": WEIGHTS,
        "inference_steps": INFERENCE_STEPS,
        "actions_per_chunk": policy.horizon_steps,
        "seeds": [seeds[0], seeds[-1]],
        "episodes_planned": len(seeds),
        "policy_seed_offset": POLICY_SEED_OFFSET,
    }
    with RunRecord(out_dir, kind="evaluation", config=settings, device=device) as record:
        task = RobosuiteCanTask(CanTaskConfig())
        suite = "m0-validation" if is_protocol else "trial"
        commit = code_revision()["commit"]
        results = []
        with open(out_dir / "episodes.jsonl", "x") as log:
            for i, seed in enumerate(seeds, 1):
                episode = run_episode(task, policy, seed, suite, policy_hash, commit)
                results.append(episode)
                log.write(json.dumps(asdict(episode)) + "\n")
                log.flush()
                done = sum(e.success for e in results)
                record.progress(
                    {
                        "episodes_completed": i,
                        "successes": done,
                        "last_seed": seed,
                        "protocol_complete": False,
                    }
                )
                print(
                    f"episode {i}/{len(seeds)} seed {seed}: "
                    f"{'success' if episode.success else 'failure'} in {episode.steps} steps"
                    f"{' (' + episode.error + ')' if episode.error else ''}; "
                    f"successes so far {done}/{i}",
                    flush=True,
                )
                if episode.error is not None:
                    # Protocol step 4: an infrastructure error makes the run incomplete. Keep
                    # the record, stop, and give no verdict; recovery is agreed before a rerun.
                    raise EpisodeError(f"seed {seed}: {episode.error}")
        successes = sum(e.success for e in results)
        low, high = wilson_interval(successes, len(results))
        result = {
            **settings,
            "episodes_completed": len(results),
            "errors": sum(e.error is not None for e in results),
            "successes": successes,
            "success_rate": successes / len(results),
            "wilson_95": [round(low, 4), round(high, 4)],
            "pass_threshold": f">= {PASS_SUCCESSES}/{len(SEEDS)}",
            "passed": (successes >= PASS_SUCCESSES) if is_protocol else None,
        }
        record.finish(result)
        verdict = (
            "trial run, no verdict"
            if not is_protocol
            else "PASS"
            if result["passed"]
            else "BELOW THRESHOLD (review per protocol)"
        )
        print(
            f"\nSuccesses {successes}/{len(results)} = {successes / len(results):.1%}, "
            f"95% Wilson interval {low:.1%}-{high:.1%}: {verdict}"
        )
        print(f"Run record: {out_dir} (manifest.json, console.log, episodes.jsonl, summary.json)")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    select = sub.add_parser("select", help="record which pretrained policy to evaluate")
    select.add_argument("--run", type=Path, default=None, help="needed if several runs finished")
    select.add_argument("--checkpoint-sha256", default=None, help="expected checkpoint hash")
    select.add_argument("--normalization-sha256", default=None, help="for runs before 0.3b")
    select.add_argument("--replace", action="store_true", help="replace an existing selection")
    evaluation = sub.add_parser("run", help="evaluate (default command)")
    for p in (parser, evaluation):
        p.add_argument("--run", type=Path, default=None, help="default: the selected policy")
        p.add_argument("--episodes", type=int, default=None, help="trial on N fixture seeds")
        p.add_argument("--device", default="cpu")
        p.add_argument("--out", type=Path, default=None)
        p.add_argument("--normalization-sha256", default=None, help="for runs before 0.3b")
    args = parser.parse_args(argv)
    if args.command == "select":
        select_policy(
            args.run,
            args.normalization_sha256,
            replace=args.replace,
            checkpoint_sha256=args.checkpoint_sha256,
        )
        return
    out = args.out or DEFAULT_OUT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run is not None:
        run, norm, ckpt, chosen_by = args.run, args.normalization_sha256, None, "given (--run)"
    else:
        chosen = load_selection()
        run, norm, ckpt = (
            Path(chosen["run"]),
            chosen["normalization_sha256"],
            chosen["checkpoint_sha256"],
        )
        chosen_by = f"selected policy ({SELECTION}, selected {chosen['selected_at']})"
    print(f"Evaluating pretraining run {run} ({chosen_by})", flush=True)
    evaluate(run, out, args.episodes, args.device, norm, chosen_by, expected_checkpoint=ckpt)


if __name__ == "__main__":
    main()
