"""Task interface: the single entry point from project code to a simulator (docs/design.md §3).

Training, evaluation, snapshots and recording use a `TaskInterface`, never robosuite or
MuJoCo directly, so a robot or task change means a new adapter.

  Observation  dict of named low-dimensional arrays; no camera images (FR-POL-04).
  StepResult   result of one control step, including the benchmark's success flag.
  Snapshot     full environment state at one moment (FR-ENV-04).

Snapshots (`get_state` / `set_state`) are designed in docs/design.md §4; in-memory
snapshots on one backend exist since increment D2, serialization follows in M2.
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
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Episode end. Candidate convention: end on success or timeout.

        [unverified] Whether the benchmark and the released checkpoint's evaluation end
        early on success or run to H is checked at milestone 0.2b; change this only
        together with the recorded config mapping.
        """
        return self.success or self.timeout


SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Snapshot:
    """Full environment state at one moment (FR-ENV-04, docs/design.md §4).

    Restoring a snapshot and replaying the same actions must reproduce the original
    continuation exactly on the same backend (NFR-REP-03). Policy-internal state is
    never stored here (FR-DIS-10).

    physics         simulator state arrays, keyed by name
    task            task bookkeeping: step counter, elapsed time, task flags
    controller      controller and gripper state that the simulator does not hold
    rng             random-number generator states used by the environment
    obs_history     last k observations, oldest first (k = config.history_len)
    action_history  last k actions, oldest first
    metadata        schema_version, task name, episode seed, step index, intervention

    History: only what actually happened in the current episode, never padded. After
    reset, obs_history holds the reset observation and action_history is empty; after n
    steps, len(obs_history) == min(k, n + 1) and len(action_history) == min(k, n).
    Alignment: action_history[-1] is the action whose step produced obs_history[-1].

    Intervention: metadata["intervention"] is None until a disturbance is recorded in the
    episode (`record_intervention`); afterwards it holds that metadata (type, parameters,
    step). It is cleared by reset and restored by set_state.

    Scope: in-memory snapshot, same backend (milestone 0.4b). On-disk serialization and
    hashing are added in M2 (milestone 2.1, test-plan T2.1).
    """

    physics: dict[str, np.ndarray]
    task: dict[str, Any]
    controller: dict[str, Any]
    rng: dict[str, Any]
    obs_history: tuple[Observation, ...]
    action_history: tuple[np.ndarray, ...]
    metadata: dict[str, Any]


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

    def camera_pose(self, camera: str) -> dict[str, Any]:
        """World pose of a named camera: {"pos": 3 floats, "xmat": 9 floats (row-major rotation)}.

        Recorded in every video's metadata (design §5.2).
        """
        ...

    def record_intervention(self, metadata: dict[str, Any]) -> None:
        """Attach disturbance metadata to the current episode (FR-ENV-04, FR-DIS-04).

        Called by the disturbance code right after a disturbance is applied; the next
        `get_state` stores it in metadata["intervention"].
        """
        ...

    def get_state(self) -> Snapshot:
        """Snapshot of the full environment state (docs/design.md §4).

        The snapshot is an independent copy: stepping the environment afterwards must not
        change it.
        """
        ...

    def set_state(self, snapshot: Snapshot) -> Observation:
        """Restore a snapshot and return its observation (docs/design.md §4).

        Restore identity: `get_state()` right after `set_state(s)` equals `s`.
        Replay identity: the same actions afterwards reproduce the original continuation.
        """
        ...
