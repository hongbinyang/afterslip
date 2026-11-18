"""Core task adapter: robosuite PickPlaceCan (the Robomimic Can task) with a Panda arm.

Implements `TaskInterface` (afterslip/tasks/interface.py, docs/design.md §3.3).

Settings come from `CanTaskConfig`. They must match the pretrained policy's released
config once it is verified (milestone 0.2b); until then the defaults are candidates.

Implementation notes and pitfalls (increment D1):
  - Create the robosuite env with `robosuite.make(...)`; the OSC_POSE controller config
    comes from `robosuite.load_controller_config(default_controller="OSC_POSE")`.
  - Seeding: robosuite's object placement samples from NumPy's *global* RNG. A seeded
    reset must therefore seed that global RNG before `env.reset()`. Check that two
    resets with the same seed give identical observations, and different seeds do not.
  - Success: use the benchmark's own predicate (robosuite env `_check_success()`), every
    step. Do not reimplement it.
  - Horizon: robosuite tracks `env.timestep` and `env.horizon`. Decide how `timeout` is
    computed so that reaching H without success is reported as timeout.
  - Observations: keep only low-dimensional entries (no images; FR-POL-04). Which keys
    the policy needs is fixed later by the checkpoint config; keep them all for now.
  - Rendering: `env.sim.render(camera_name=..., width=..., height=...)` returned frames
    upside down on macOS with robosuite 1.4.1 [unverified on other platforms]. Return
    upright frames. Rendering must not change the trajectory: the integration test
    compares a rendered and an unrendered rollout.
  - [unverified] Equivalence with the upstream training code's own environment wrapper
    (observation processing, action scaling, episode end) is checked at 0.2b/0.3c.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from afterslip.tasks.interface import Observation, StepResult


@dataclass(frozen=True)
class CanTaskConfig:
    """Candidate settings for the Can task (to be matched to the checkpoint at 0.2b)."""

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
        # TODO(D1): create the robosuite environment (offscreen renderer on, camera
        # observations off) and set self.action_dim from it.
        raise NotImplementedError("D1: create the robosuite environment")

    @property
    def benchmark_env(self) -> Any:
        """The wrapped robosuite environment, for tests that compare with the benchmark."""
        raise NotImplementedError("D1")

    def reset(self, seed: int) -> Observation:
        # TODO(D1): seed every random source the task uses, reset, return state observation.
        raise NotImplementedError("D1: seeded reset")

    def step(self, action: np.ndarray) -> StepResult:
        # TODO(D1): step the environment; fill success from the benchmark predicate and
        # timeout from the step count and horizon.
        raise NotImplementedError("D1: step")

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        # TODO(D1): offscreen render, return an upright uint8 RGB frame.
        raise NotImplementedError("D1: render")

    def get_state(self) -> Any:
        raise NotImplementedError("D2: snapshot (docs/design.md §4)")

    def set_state(self, snapshot: Any) -> Observation:
        raise NotImplementedError("D2: snapshot (docs/design.md §4)")
