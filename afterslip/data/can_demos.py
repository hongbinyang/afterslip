"""Low-dimensional Can demonstrations for state-based pretraining (milestone 0.2b, Path 2).

Source: robomimic v1.5 proficient-human Can demonstrations (MIT; ARTIFACTS.md §3), recorded
with robosuite 1.5.1. Only the recorded simulator states and actions are used; every
observation is regenerated from the states in this project's robosuite 1.4.1 environment,
because robosuite 1.5.1 orders and defines the 14-D `object` observation differently
(docs/design.md §10).

Output, in the upstream training format (`third_party/ReinFlow`, StitchedSequenceDataset):

  train.npz          states (N, 23), actions (N, 7), rewards, terminals, traj_lengths,
                     all normalized to [-1, 1] with the train-split statistics
  val.npz            the same for robomimic's `valid` mask (held out from pretraining)
  normalization.npz  obs_min, obs_max, action_min, action_max (train split only)

Run: `make data-can` (or `python -m afterslip.data.can_demos --out-dir <dir>`).
Downloads and outputs live under the git-ignored `outputs/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from afterslip.tasks.robosuite_can import CanTaskConfig, RobosuiteCanTask

DATASET_URL = (
    "https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/"
    "v1.5/can/ph/low_dim_v15.hdf5"
)
DATASET_SHA256 = "3f2eb92e0a5025d0095e866ac16cc8092d6a762abe27dec90dbaff9027282962"
DATASET_BYTES = 46_889_752

# Policy observation: the upstream state-based Can configuration (obs_dim 23), in order.
OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")
OBS_DIM = 23
ACTION_DIM = 7

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "outputs" / "data" / "raw"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "reinflow" / "data" / "robomimic" / "can-state"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


DOWNLOAD_TIMEOUT_S = 60  # per connection attempt and per read; a stalled transfer fails


def download(raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Fetch the dataset once; any file whose hash differs from the pinned one is rejected.

    The transfer goes to a `.part` file that is removed on any failure (network error,
    timeout, wrong hash), so only a verified file ever appears under its final name.
    """
    path = raw_dir / "can_ph_low_dim_v15.hdf5"
    if path.is_file() and sha256(path) == DATASET_SHA256:
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=DOWNLOAD_TIMEOUT_S) as response:
            with open(partial, "wb") as out:
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
        found = sha256(partial)
        if found != DATASET_SHA256:
            raise RuntimeError(f"sha256 mismatch for {DATASET_URL}: {found} != {DATASET_SHA256}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def observation_vector(raw_obs: dict) -> np.ndarray:
    """The 23-D policy observation from a robosuite 1.4.1 observation dict."""
    parts = {**raw_obs, "object": raw_obs["object-state"]}
    return np.concatenate([np.asarray(parts[key], dtype=np.float64) for key in OBS_KEYS])


def regenerate_observations(env, states: np.ndarray) -> np.ndarray:
    """Observation at each recorded state, computed by our environment (no stepping)."""
    out = np.empty((len(states), OBS_DIM))
    for t, state in enumerate(states):
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        out[t] = observation_vector(env._get_observations(force_update=True))
    return out


@dataclass
class Split:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    traj_lengths: np.ndarray
    demos: list[str]


def load_split(path: Path, mask: str, env) -> Split:
    """Regenerated observations and recorded actions for the demos in a robomimic mask."""
    with h5py.File(path, "r") as f:
        names = sorted((n.decode() for n in f[f"mask/{mask}"][()]), key=lambda n: int(n[5:]))
        obs, actions, rewards, lengths = [], [], [], []
        for name in names:
            demo = f[f"data/{name}"]
            obs.append(regenerate_observations(env, demo["states"][()]))
            actions.append(demo["actions"][()])
            rewards.append(demo["rewards"][()])
            lengths.append(len(demo["actions"]))
    return Split(
        obs=np.concatenate(obs),
        actions=np.concatenate(actions),
        rewards=np.concatenate(rewards),
        traj_lengths=np.array(lengths),
        demos=names,
    )


def _normalize(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    # Same mapping as the upstream converter: [low, high] -> [-1, 1].
    return 2 * (x - low) / (high - low + 1e-6) - 1


def convert(raw_path: Path, out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    """Write train.npz, val.npz and normalization.npz; return a summary."""
    env = RobosuiteCanTask(CanTaskConfig()).benchmark_env
    env.reset()
    train = load_split(raw_path, "train", env)
    val = load_split(raw_path, "valid", env)
    # Statistics from the train split only, so nothing from held-out demos leaks in.
    stats = {
        "obs_min": train.obs.min(axis=0),
        "obs_max": train.obs.max(axis=0),
        "action_min": train.actions.min(axis=0),
        "action_max": train.actions.max(axis=0),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        np.savez_compressed(
            out_dir / f"{name}.npz",
            states=_normalize(split.obs, stats["obs_min"], stats["obs_max"]),
            actions=_normalize(split.actions, stats["action_min"], stats["action_max"]),
            rewards=split.rewards,
            terminals=np.zeros(len(split.obs), dtype=bool),
            traj_lengths=split.traj_lengths,
        )
    np.savez_compressed(out_dir / "normalization.npz", **stats)
    summary = {
        "source": DATASET_URL,
        "sha256": DATASET_SHA256,
        "obs_keys": list(OBS_KEYS),
        "train": {"demos": len(train.demos), "transitions": int(train.traj_lengths.sum())},
        "val": {"demos": len(val.demos), "transitions": int(val.traj_lengths.sum())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(convert(download(args.raw_dir), args.out_dir), indent=2))


if __name__ == "__main__":
    main()
