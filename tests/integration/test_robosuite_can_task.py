"""Done-definition for increment D1, task interface (docs/design.md §3; test-plan T0.6).

These tests fail until `RobosuiteCanTask` is implemented. They are the acceptance
checks for the hands-on task; do not weaken them to make them pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from afterslip.tasks.robosuite_can import CanTaskConfig, RobosuiteCanTask

pytestmark = pytest.mark.integration


@pytest.fixture
def task():
    return RobosuiteCanTask(CanTaskConfig())


def _random_actions(task, n, seed=0):
    rng = np.random.default_rng(seed)
    low, high = task.benchmark_env.action_spec
    return [rng.uniform(low, high) * 0.3 for _ in range(n)]


def test_action_dim_and_horizon(task):
    assert task.action_dim == 7
    assert task.horizon == CanTaskConfig().horizon


def test_observation_is_state_only(task):
    obs = task.reset(seed=0)
    assert {"robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "Can_pos"} <= set(obs)
    for key, value in obs.items():
        assert value.ndim <= 1, f"{key} looks like an image; policies are state-based (FR-POL-04)"


def test_same_seed_same_initial_state(task):
    a = task.reset(seed=3)
    b = task.reset(seed=3)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_different_seeds_place_the_can_differently(task):
    a = task.reset(seed=3)["Can_pos"]
    b = task.reset(seed=4)["Can_pos"]
    assert not np.array_equal(a, b)


def test_success_flag_is_the_benchmark_predicate(task):
    task.reset(seed=0)
    for action in _random_actions(task, 20):
        result = task.step(action)
        assert result.success == bool(task.benchmark_env._check_success())


def test_reaching_horizon_without_success_is_timeout(task):
    task.reset(seed=0)
    result = None
    for i, action in enumerate(_random_actions(task, task.horizon), start=1):
        result = task.step(action)
        if result.success:
            pytest.skip("random actions succeeded; timeout path not exercised for this seed")
        assert result.timeout == (i == task.horizon)
    assert result is not None and result.timeout and result.done


def test_render_returns_upright_rgb_frame(task):
    task.reset(seed=0)
    frame = task.render("agentview", 64, 48)
    assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8
    # [unverified on platforms other than macOS] upright orientation: the floor is
    # below the table, so the bottom rows should differ from the top rows.
    assert not np.array_equal(frame[0], frame[-1])


def test_rendering_does_not_change_the_trajectory(task):
    actions = _random_actions(task, 30)
    task.reset(seed=1)
    plain = [task.step(a).observation for a in actions]
    task.reset(seed=1)
    rendered = []
    for a in actions:
        rendered.append(task.step(a).observation)
        task.render("agentview", 64, 64)
    for x, y in zip(plain, rendered):
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])
