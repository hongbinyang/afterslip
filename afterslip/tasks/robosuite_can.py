"""Task adapter for robosuite PickPlaceCan (the Robomimic Can task) with a Panda arm.

Implements `TaskInterface` (docs/design.md §3.3). Settings come from `CanTaskConfig`
and are candidates until matched to the pretrained policy's config (milestone 0.2b).

Notes:
  - robosuite places objects with NumPy's global RNG, so `reset` seeds it.
  - `success` is the benchmark's own predicate; `timeout` means the horizon was
    reached without success.
  - Observations keep low-dimensional entries only (FR-POL-04).
  - `MjSim.render` returns frames upside down (observed on macOS); `render` flips them.
  - Snapshots hold MuJoCo's full integration state plus the Python-side state robosuite
    keeps outside it (controller, gripper command, step counters, global NumPy RNG).
    Limitation: the global RNG is shared by every environment in the process.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
import robosuite as suite

from afterslip.tasks.interface import SNAPSHOT_SCHEMA_VERSION, Observation, Snapshot, StepResult


@dataclass(frozen=True)
class CanTaskConfig:
    """Candidate settings for the Can task (to be matched to the checkpoint at 0.2b)."""

    env_name: str = "PickPlaceCan"
    robot: str = "Panda"
    controller: str = "OSC_POSE"
    control_freq_hz: float = 20.0
    horizon: int = 300
    observer_camera: str = "agentview"  # candidate observer camera (design §5.2)
    history_len: int = 1  # k observations/actions kept for snapshots; from the policy config


class RobosuiteCanTask:
    """Robosuite PickPlaceCan behind the TaskInterface contract."""

    def __init__(self, config: CanTaskConfig = CanTaskConfig()) -> None:
        self.config = config
        self.horizon = config.horizon
        self.control_freq_hz = config.control_freq_hz
        controller_config = suite.load_controller_config(default_controller=config.controller)
        self._env = suite.make(
            env_name=config.env_name,
            robots=config.robot,
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            control_freq=int(config.control_freq_hz),
            horizon=config.horizon,
        )
        self.action_dim = self._env.action_dim
        self._seed: int | None = None
        self._intervention: dict[str, Any] | None = None
        self._obs_history: deque[Observation] = deque(maxlen=config.history_len)
        self._action_history: deque[np.ndarray] = deque(maxlen=config.history_len)

    @property
    def benchmark_env(self) -> Any:
        """The wrapped robosuite environment, for tests that compare with the benchmark."""
        return self._env

    def reset(self, seed: int) -> Observation:
        np.random.seed(seed)
        raw = self._env.reset()
        self._seed = seed
        self._intervention = None
        obs = self._state_obs(raw)
        self._obs_history.clear()
        self._action_history.clear()
        self._obs_history.append(obs)
        return {key: value.copy() for key, value in obs.items()}

    @staticmethod
    def _state_obs(raw: dict) -> Observation:
        """Keep low-dimensional state entries only (no images; FR-POL-04)."""
        return {key: np.asarray(value) for key, value in raw.items() if np.asarray(value).ndim <= 1}

    def step(self, action: np.ndarray) -> StepResult:
        raw, reward, _done, info = self._env.step(action)
        success = bool(self._env._check_success())
        timeout = (not success) and self._env.timestep >= self.horizon

        obs = self._state_obs(raw)
        self._action_history.append(np.array(action, copy=True))
        self._obs_history.append(obs)

        return StepResult(
            observation={key: value.copy() for key, value in obs.items()},
            reward=float(reward),
            success=success,
            timeout=timeout,
            info=info,
        )

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        frame = self._env.sim.render(camera_name=camera, width=width, height=height, depth=False)
        frame = np.ascontiguousarray(frame[::-1, :, :])
        return frame.astype(np.uint8)

    def camera_pose(self, camera: str) -> dict[str, Any]:
        cam_id = self._env.sim.model.camera_name2id(camera)
        return {
            "pos": self._env.sim.data.cam_xpos[cam_id].tolist(),
            "xmat": self._env.sim.data.cam_xmat[cam_id].tolist(),
        }

    # Everything mj_step reads: qpos, qvel, act, warm-start, controls, applied forces, time.
    _PHYSICS_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

    def record_intervention(self, metadata: dict[str, Any]) -> None:
        self._intervention = copy.deepcopy(metadata)

    def get_state(self) -> Snapshot:
        sim = self._env.sim
        model = sim.model._model
        data = sim.data._data
        physics = np.empty(mujoco.mj_stateSize(model, self._PHYSICS_SPEC))
        mujoco.mj_getState(model, data, physics, self._PHYSICS_SPEC)
        task_state = {
            "timestep": self._env.timestep,
            "cur_time": self._env.cur_time,
            "done": self._env.done,
        }
        robot = self._env.robots[0]
        controller_bundle: dict[str, Any] = {
            "gripper_action": robot.gripper.current_action.copy(),
            # Controller attributes persist across episodes, so all scalar/array ones are saved.
            "arm": {
                key: copy.deepcopy(value)
                for key, value in vars(robot.controller).items()
                if isinstance(value, (np.ndarray, float, int, bool))
            },
        }

        obs_history: tuple[Observation, ...] = tuple(
            copy.deepcopy(obs) for obs in self._obs_history
        )
        action_history: tuple[np.ndarray, ...] = tuple(
            np.asarray(act).copy() for act in self._action_history
        )
        metadata: dict[str, Any] = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "task": self.config.env_name,
            "seed": self._seed,
            "step": self._env.timestep,
            "intervention": copy.deepcopy(self._intervention),
        }

        return Snapshot(
            physics={"integration": physics},
            task=task_state,
            controller=controller_bundle,
            rng={"numpy_global": copy.deepcopy(np.random.get_state())},
            obs_history=obs_history,
            action_history=action_history,
            metadata=metadata,
        )

    def set_state(self, snapshot: Snapshot) -> Observation:
        sim = self._env.sim
        model = sim.model._model
        data = sim.data._data

        physics = snapshot.physics["integration"]

        # mj_forward recomputes derived quantities but overwrites the warm-start
        # (qacc_warmstart), so the state is set again afterwards.
        mujoco.mj_setState(model, data, physics.copy(), self._PHYSICS_SPEC)
        mujoco.mj_forward(model, data)
        mujoco.mj_setState(model, data, physics.copy(), self._PHYSICS_SPEC)

        task_state = snapshot.task
        step_val = int(task_state["timestep"])
        self._env.timestep = step_val
        self._env.cur_time = snapshot.task["cur_time"]
        self._env.done = bool(task_state["done"])

        robot = self._env.robots[0]
        robot.gripper.current_action = snapshot.controller["gripper_action"].copy()
        for key, value in snapshot.controller["arm"].items():
            setattr(robot.controller, key, copy.deepcopy(value))

        np.random.set_state(copy.deepcopy(snapshot.rng["numpy_global"]))

        self._obs_history.clear()
        for obs in snapshot.obs_history:
            self._obs_history.append(copy.deepcopy(obs))

        self._action_history.clear()
        for act in snapshot.action_history:
            self._action_history.append(np.asarray(act).copy())

        self._seed = snapshot.metadata["seed"]
        self._intervention = copy.deepcopy(snapshot.metadata["intervention"])

        return {key: value.copy() for key, value in snapshot.obs_history[-1].items()}
