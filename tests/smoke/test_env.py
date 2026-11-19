"""Smoke tests for the simulation stack (docs/setup.md; test-plan T0.4, T0.15).

The stack is imported directly: a missing or broken dependency is a failure, not a skip.
Videos are written to pytest's temporary directory.

Run: pytest -m smoke
"""

from __future__ import annotations

import imageio
import numpy as np
import pytest
import robosuite

pytestmark = pytest.mark.smoke

SEED = 0
HORIZON = 300
CONTROL_FREQ = 20


@pytest.fixture
def can_env():
    """The robosuite PickPlaceCan task (the Robomimic Can task) with Panda + OSC_POSE."""
    ctrl = robosuite.load_controller_config(default_controller="OSC_POSE")
    env = robosuite.make(
        "PickPlaceCan",
        robots="Panda",
        controller_configs=ctrl,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
    )
    np.random.seed(SEED)  # robosuite's placement samplers use the global numpy RNG
    env.reset()
    yield env
    env.close()


def _random_actions(env, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    low, high = env.action_spec
    return [rng.uniform(low, high) * 0.3 for _ in range(n)]


def test_can_env_creates_and_steps(can_env):
    """The Can task can be created and stepped; the action space is 7-D (OSC pose + gripper)."""
    assert can_env.action_dim == 7
    for a in _random_actions(can_env, 10, np.random.default_rng(SEED)):
        obs, _reward, _done, _info = can_env.step(a)
    for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "Can_pos"):
        assert key in obs


def test_offscreen_video_is_readable(can_env, tmp_path):
    """Offscreen rendering produces a readable, non-blank video with one frame per step."""
    steps, res = 30, 128
    frames = []
    for a in _random_actions(can_env, steps, np.random.default_rng(SEED)):
        can_env.step(a)
        frames.append(can_env.sim.render(camera_name="agentview", width=res, height=res)[::-1])
    path = tmp_path / "can_random.mp4"
    imageio.mimsave(path, frames, fps=CONTROL_FREQ)

    decoded = np.stack(imageio.mimread(path, memtest=False)).astype(np.float32)
    assert decoded.shape[0] == steps

    # Non-blank: every frame has spatial structure, i.e. it is not a single uniform
    # colour. Measured per frame and per channel over the image plane (H, W).
    spatial_std = decoded.std(axis=(1, 2))  # shape (frames, channels)
    assert (spatial_std.max(axis=1) > 1.0).all(), "a frame is uniform: rendering likely blank"

    # Changing: the scene moves under random actions, so the first and last frames differ.
    # The threshold allows for lossy video compression noise.
    assert np.abs(decoded[-1] - decoded[0]).mean() > 0.5, "frames do not change over the episode"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,  # only the state mismatch is expected; any other error fails
    reason=(
        "Known failure (milestones 0.4b): restoring only sim.get_state() does not "
        "reproduce the continuation; cause not yet isolated. strict=True makes this "
        "test fail once fixed, so the marker is removed deliberately."
    ),
)
def test_sim_state_only_snapshot_round_trip(can_env):
    """Diagnostic: save sim state, run 100 actions, restore, replay; states should match exactly."""
    rng = np.random.default_rng(SEED)
    for a in _random_actions(can_env, 50, rng):
        can_env.step(a)
    snapshot = can_env.sim.get_state().flatten().copy()
    actions = _random_actions(can_env, 100, rng)

    traj_a = []
    for a in actions:
        can_env.step(a)
        traj_a.append(can_env.sim.get_state().flatten().copy())

    can_env.sim.set_state_from_flattened(snapshot)
    can_env.sim.forward()
    traj_b = []
    for a in actions:
        can_env.step(a)
        traj_b.append(can_env.sim.get_state().flatten().copy())

    np.testing.assert_array_equal(np.stack(traj_a), np.stack(traj_b))
