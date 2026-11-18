"""The task interface: the only way research code touches a simulator (docs/design.md §3).

Training, evaluation, snapshots and recording talk to a `TaskInterface`, never to
robosuite or MuJoCo directly. A robot or task change (e.g. the FR3 track) is then a
new adapter, not a rewrite.

Key data structures
  Observation  dict of named low-dimensional arrays. State only: no camera images are
               policy inputs (requirement FR-POL-04).
  StepResult   what one control step returns, including the benchmark's own success flag.

Snapshots (`get_state` / `set_state`) are part of the contract but are designed and
implemented in increment D2 (design §4); D1 adapters may leave them unimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

try:  # Python 3.8+
    from typing import Protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol  # type: ignore

Observation = Dict[str, np.ndarray]


@dataclass(frozen=True)
class StepResult:
    """Result of one control step.

    observation  next observation (low-dimensional state only)
    reward       benchmark reward (sparse success reward, requirement FR-RL-02)
    success      the benchmark's own success predicate after this step (FR-ENV-05)
    timeout      True when the step count reached the horizon H without success
    info         adapter-specific extras; never needed by research code
    """

    observation: Observation
    reward: float
    success: bool
    timeout: bool
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Episode end. Candidate convention: end on success or timeout.

        [unverified] Whether the benchmark and the released checkpoint's evaluation end
        early on success or run to H is checked at milestone 0.2b; change this only
        together with the recorded config mapping.
        """
        return self.success or self.timeout


class TaskInterface(Protocol):
    """Contract every task adapter implements (design §3.2)."""

    horizon: int  # H, the task's standard horizon (FR-ENV-05)
    action_dim: int
    control_freq_hz: float

    def reset(self, seed: int) -> Observation:
        """Start a new episode. The same seed must give the same initial state."""
        ...

    def step(self, action: np.ndarray) -> StepResult:
        """Apply one action for one control step."""
        ...

    def render(self, camera: str, width: int, height: int) -> np.ndarray:
        """Return an upright RGB frame (H, W, 3) uint8 from a named camera.

        Must not change simulator state or any random number stream.
        """
        ...

    def get_state(self) -> Any:
        """Snapshot of the full environment state (increment D2, design §4)."""
        ...

    def set_state(self, snapshot: Any) -> Observation:
        """Restore a snapshot and return the observation (increment D2, design §4)."""
        ...
