"""Task adapter for robosuite PickPlaceCan (the Robomimic Can task) with a Panda arm.

Implements `TaskInterface` (docs/design.md §3.3). Settings come from `CanTaskConfig`
and are candidates until matched to the pretrained policy's config (milestone 0.2b).

Notes:
  - robosuite places objects with NumPy's global RNG, so `reset` seeds it.
  - `success` is the benchmark's own predicate; `timeout` means the horizon was
    reached without success.
  - Observations keep low-dimensional entries only (FR-POL-04).
  - `MjSim.render` returns frames upside down (observed on macOS); `render` flips them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import robosuite as suite

from afterslip.tasks.interface import Observation, StepResult


@dataclass(frozen=True)
class CanTaskConfig:
    """Candidate settings for the Can task (to be matched to the checkpoint at 0.2b)."""

    env_name: str = "PickPlaceCan"
    robot: str = "Panda"
    controller: str = "OSC_POSE"
    control_freq_hz: float = 20.0
    horizon: int = 300
    observer_camera: str = "agentview"  # candidate observer camera (design §5.2)


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

    @property
    def benchmark_env(self) -> Any:
        """The wrapped robosuite environment, for tests that compare with the benchmark."""
        return self._env

    def reset(self, seed: int) -> Observation:
        np.random.seed(seed)
        raw = self._env.reset()
        return self._state_obs(raw)

    @staticmethod
    def _state_obs(raw: dict) -> Observation:
        """Keep low-dimensional state entries only (no images; FR-POL-04)."""
        return {key: np.asarray(value) for key, value in raw.items() if np.asarray(value).ndim <= 1}

    def step(self, action: np.ndarray) -> StepResult:
        raw, reward, _done, info = self._env.step(action)
        success = bool(self._env._check_success())
        timeout = (not success) and self._env.timestep >= self.horizon
        return StepResult(
            observation=self._state_obs(raw),
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

    def get_state(self) -> Any:
        # TODO(D2): return a full snapshot of the environment (docs/design.md §4).
        # Implementation notes and pitfalls (increment D2):
        #   - FR-ENV-04 requires every group: simulator, task, controller internal state,
        #     RNG states, last-k observation/action history and intervention metadata.
        #   - robosuite's `sim.get_state()` holds only time, qpos and qvel; the smoke test
        #     `test_sim_state_only_snapshot_round_trip` shows that this alone does not
        #     reproduce the continuation. design.md §4.3 lists candidate components to
        #     investigate (controller goals, interpolators, counters, RNG, ...).
        #   - Never store policy-internal state in the snapshot.
        raise NotImplementedError("D2: snapshot (docs/design.md §4)")

    def set_state(self, snapshot: Any) -> Observation:
        # TODO(D2): restore a snapshot and return the observation (docs/design.md §4).
        # Two properties are tested separately:
        #   - restore identity: get_state() right after set_state(s) equals s;
        #   - replay identity: the same actions afterwards reproduce the original
        #     continuation exactly on the same backend.
        # After writing simulator state, recompute derived quantities (e.g. sim.forward()).
        raise NotImplementedError("D2: snapshot (docs/design.md §4)")
