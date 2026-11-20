"""Can demonstrations regenerated in robosuite 1.4.1 (milestone 0.2b; docs/design.md §10).

Needs the downloaded dataset (`make data-can`); CI downloads it before the tests. The
conditions checked here were agreed for using robosuite 1.5.1 data in robosuite 1.4.1:
robot observations reproduced, observation order equal to the live environment, actions
aligned with observations, and the validation demos held out.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from afterslip.data import can_demos
from afterslip.tasks.robosuite_can import CanTaskConfig, RobosuiteCanTask

pytestmark = pytest.mark.integration

RAW = can_demos.DEFAULT_RAW_DIR / "can_ph_low_dim_v15.hdf5"
SAMPLE_DEMOS = ("demo_0", "demo_57", "demo_199")
# Slices of the 23-D observation that the dataset also records under the same definition.
ROBOT_SLICES = {
    "robot0_eef_pos": slice(0, 3),
    "robot0_eef_quat": slice(3, 7),
    "robot0_gripper_qpos": slice(7, 9),
}


@pytest.fixture(scope="module")
def raw():
    if not RAW.is_file():
        pytest.skip("dataset not downloaded; run `make data-can`")
    with h5py.File(RAW, "r") as f:
        yield f


@pytest.fixture(scope="module")
def env():
    task = RobosuiteCanTask(CanTaskConfig())
    task.reset(seed=0)
    return task.benchmark_env


def test_downloaded_file_is_the_pinned_one():
    if not RAW.is_file():
        pytest.skip("dataset not downloaded; run `make data-can`")
    assert can_demos.sha256(RAW) == can_demos.DATASET_SHA256


@pytest.mark.parametrize("name", SAMPLE_DEMOS)
def test_regenerated_robot_observations_match_the_recording(raw, env, name):
    demo = raw[f"data/{name}"]
    obs = can_demos.regenerate_observations(env, demo["states"][()])
    for key, part in ROBOT_SLICES.items():
        # The dataset stores observations in float32-level precision.
        np.testing.assert_allclose(obs[:, part], demo[f"obs/{key}"][()], atol=1e-6, err_msg=key)


def test_observation_order_equals_the_live_environment(env):
    # After a real step, the regenerated observation at the resulting state must equal the
    # observation the environment returned, element by element.
    env.reset()
    raw_obs, *_ = env.step(np.zeros(can_demos.ACTION_DIM))
    live = can_demos.observation_vector(raw_obs)
    regenerated = can_demos.regenerate_observations(env, env.sim.get_state().flatten()[None])[0]
    assert live.shape == (can_demos.OBS_DIM,)
    # robosuite builds a step's observation without a final forward pass, so body poses lag
    # the state by one physics substep (about 1e-5 here); regeneration recomputes them
    # exactly. A wrong order would show differences of order 0.1 to 1.
    np.testing.assert_allclose(regenerated, live, atol=1e-4)


@pytest.mark.parametrize("name", SAMPLE_DEMOS)
def test_actions_are_aligned_with_observations(raw, env, name):
    # Pair t is (observation at recorded state t, action taken from state t).
    demo = raw[f"data/{name}"]
    states, actions = demo["states"][()], demo["actions"][()]
    assert len(states) == len(actions) == demo.attrs["num_samples"]
    obs = can_demos.regenerate_observations(env, states)
    np.testing.assert_allclose(
        obs[:, ROBOT_SLICES["robot0_eef_pos"]], demo["obs/robot0_eef_pos"][()], atol=1e-6
    )
    # The next recorded observation is the one after the action: one step ahead, not equal.
    assert not np.allclose(obs[1:, :3], demo["obs/robot0_eef_pos"][()][:-1], atol=1e-6)


def test_conversion_holds_out_the_validation_demos(raw, tmp_path):
    summary = can_demos.convert(RAW, tmp_path)
    train_names = {n.decode() for n in raw["mask/train"][()]}
    valid_names = {n.decode() for n in raw["mask/valid"][()]}
    assert train_names.isdisjoint(valid_names)
    assert summary["train"]["demos"] == len(train_names)
    assert summary["val"]["demos"] == len(valid_names)

    train = np.load(tmp_path / "train.npz")
    norm = np.load(tmp_path / "normalization.npz")
    assert train["states"].shape == (train["traj_lengths"].sum(), can_demos.OBS_DIM)
    assert train["actions"].shape == (train["traj_lengths"].sum(), can_demos.ACTION_DIM)
    # Normalized with train statistics, so the train split lies within [-1, 1].
    assert train["states"].min() >= -1 and train["states"].max() <= 1
    assert norm["obs_min"].shape == norm["obs_max"].shape == (can_demos.OBS_DIM,)
