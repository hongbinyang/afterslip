"""Acceptance tests for snapshots, increment D2 (docs/design.md §4; test-plan T0.11).

Restore identity and replay identity, including hidden state; do not weaken them.

Scope: the in-memory snapshot on one backend (milestone 0.4b). On-disk serialization,
canonical hashing and banks are tested later with milestone 2.1 (test-plan T2.1).
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from afterslip.tasks.interface import SNAPSHOT_SCHEMA_VERSION, Snapshot
from afterslip.tasks.robosuite_can import CanTaskConfig, RobosuiteCanTask

pytestmark = [pytest.mark.integration]

INTERVENTION = {"type": "D1-drop", "params": {"magnitude": 0.5}, "step": 20}


@pytest.fixture
def task():
    return RobosuiteCanTask(CanTaskConfig())


def _actions(task, n, seed):
    rng = np.random.default_rng(seed)
    low, high = task.benchmark_env.action_spec
    return [rng.uniform(low, high) * 0.3 for _ in range(n)]


def _run(task, actions):
    """Step through `actions`; return the observations after each step."""
    return [task.step(a).observation for a in actions]


def _trace(task, actions):
    """Step through `actions`; record observable *and* hidden state after each step.

    Hidden state is read directly from the benchmark (not through get_state), so a
    divergence cannot hide behind matching observations (test-plan T0.11).
    """
    env = task.benchmark_env
    controller = env.robots[0].controller
    trace = []
    for a in actions:
        obs = task.step(a).observation
        trace.append(
            {
                "observation": obs,
                "qpos": env.sim.data.qpos.copy(),  # robot joints, gripper fingers, can pose
                "qvel": env.sim.data.qvel.copy(),
                "controller_goal_pos": np.array(controller.goal_pos, copy=True),
                "controller_goal_ori": np.array(controller.goal_ori, copy=True),
            }
        )
    return trace


def _assert_equal(a, b, path="snapshot"):
    """Exact, recursive equality for dicts, sequences, arrays and scalars."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f"{path}: keys differ"
        for key in a:
            _assert_equal(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _assert_snapshots_equal(a: Snapshot, b: Snapshot):
    for f in dataclasses.fields(Snapshot):
        _assert_equal(getattr(a, f.name), getattr(b, f.name), f.name)


# --- contents ------------------------------------------------------------------------


def test_snapshot_has_all_required_groups(task):
    task.reset(seed=0)
    _run(task, _actions(task, 5, seed=0))
    snap = task.get_state()
    assert isinstance(snap, Snapshot)
    for group in ("physics", "task", "controller", "rng"):
        assert getattr(snap, group), f"snapshot group '{group}' is empty"
    assert snap.metadata["schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert snap.metadata["seed"] == 0
    assert "intervention" in snap.metadata


def test_history_right_after_reset_has_no_actions(task):
    reset_obs = task.reset(seed=0)
    snap = task.get_state()
    assert len(snap.obs_history) == 1 and len(snap.action_history) == 0
    _assert_equal(snap.obs_history[-1], reset_obs, "obs_history[-1]")


def test_history_is_bounded_and_aligned():
    task = RobosuiteCanTask(CanTaskConfig(history_len=3))
    task.reset(seed=0)
    actions = _actions(task, 5, seed=0)
    observations = _run(task, actions[:2])
    snap = task.get_state()
    # Fewer steps than k: nothing is padded.
    assert len(snap.obs_history) == 3 and len(snap.action_history) == 2
    _assert_equal(snap.action_history[-1], actions[1], "action_history[-1]")
    _assert_equal(snap.obs_history[-1], observations[-1], "obs_history[-1]")
    observations += _run(task, actions[2:])
    snap = task.get_state()
    # More steps than k: only the last k are kept, oldest first.
    assert len(snap.obs_history) == 3 and len(snap.action_history) == 3
    _assert_equal(list(snap.action_history), actions[-3:], "action_history")
    _assert_equal(list(snap.obs_history), observations[-3:], "obs_history")


# --- intervention metadata -----------------------------------------------------------


def test_intervention_is_none_until_recorded_and_cleared_by_reset(task):
    task.reset(seed=0)
    assert task.get_state().metadata["intervention"] is None
    task.record_intervention(INTERVENTION)
    _assert_equal(task.get_state().metadata["intervention"], INTERVENTION, "intervention")
    task.reset(seed=1)
    assert task.get_state().metadata["intervention"] is None


def test_intervention_is_copied_and_restored(task):
    task.reset(seed=0)
    _run(task, _actions(task, 20, seed=0))
    record = copy.deepcopy(INTERVENTION)
    task.record_intervention(record)
    record["params"]["magnitude"] = 99.0  # caller mutates its own dict afterwards
    snap = task.get_state()
    _assert_equal(snap.metadata["intervention"], INTERVENTION, "intervention")
    task.reset(seed=1)
    task.set_state(snap)
    _assert_equal(task.get_state().metadata["intervention"], INTERVENTION, "intervention")


# --- copy and restore ----------------------------------------------------------------


def test_snapshot_is_an_independent_copy(task):
    task.reset(seed=0)
    _run(task, _actions(task, 5, seed=0))
    snap = task.get_state()
    saved = copy.deepcopy(snap)
    _run(task, _actions(task, 10, seed=1))
    _assert_snapshots_equal(snap, saved)


def test_restore_identity(task):
    task.reset(seed=0)
    _run(task, _actions(task, 20, seed=0))
    snap = task.get_state()
    _run(task, _actions(task, 30, seed=1))
    task.set_state(snap)
    _assert_snapshots_equal(task.get_state(), snap)


def test_set_state_returns_the_snapshot_observation(task):
    task.reset(seed=0)
    _run(task, _actions(task, 20, seed=0))
    snap = task.get_state()
    _run(task, _actions(task, 10, seed=1))
    obs = task.set_state(snap)
    _assert_equal(obs, snap.obs_history[-1], "observation")


# --- replay --------------------------------------------------------------------------


def test_replay_identity(task):
    """T0.11: 50 steps, snapshot, 100 steps (A); restore, same 100 actions (B); A == B.

    Compares observations and hidden state (qpos, qvel, controller goal) at every step.
    """
    task.reset(seed=0)
    _run(task, _actions(task, 50, seed=0))
    snap = task.get_state()
    actions = _actions(task, 100, seed=1)
    trajectory_a = _trace(task, actions)
    task.set_state(snap)
    trajectory_b = _trace(task, actions)
    _assert_equal(trajectory_a, trajectory_b, "trajectory")


def test_replay_identity_from_another_episode(task):
    """A snapshot restores correctly even after the env has moved on to a new episode."""
    task.reset(seed=0)
    _run(task, _actions(task, 50, seed=0))
    snap = task.get_state()
    actions = _actions(task, 50, seed=1)
    trajectory_a = _trace(task, actions)
    task.reset(seed=7)
    _run(task, _actions(task, 30, seed=2))
    task.set_state(snap)
    trajectory_b = _trace(task, actions)
    _assert_equal(trajectory_a, trajectory_b, "trajectory")
