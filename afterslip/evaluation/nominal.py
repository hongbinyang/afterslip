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

    python -m afterslip.evaluation.nominal --run <pretraining run folder>
    python -m afterslip.evaluation.nominal --run <folder> --episodes 5     # trial: fixture seeds
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


def evaluate(
    run_dir: Path,
    out_dir: Path,
    episodes: int | None = None,
    device: str = "cpu",
    normalization_sha256: str | None = None,
):
    from afterslip.tasks.robosuite_can import CanTaskConfig, RobosuiteCanTask

    seeds = SEEDS if episodes is None else TRIAL_SEEDS[:episodes]
    is_protocol = episodes is None
    checkpoint = Path(run_dir) / "checkpoint" / "last.pt"
    policy = Policy(checkpoint, device=device)
    policy_hash = sha256(checkpoint)

    # Checked before any run folder is created or any seed is used.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="pretraining run folder")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="trial on the first N fixture seeds, not the protocol",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--normalization-sha256",
        default=None,
        help="expected hash, for pretraining runs that did not record it",
    )
    args = parser.parse_args()
    out = args.out or DEFAULT_OUT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    evaluate(args.run, out, args.episodes, args.device, args.normalization_sha256)


if __name__ == "__main__":
    main()
