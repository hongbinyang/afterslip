"""Observer recorder: one fixed third-person camera -> MP4 + JSON metadata sidecar.

Design: docs/design.md §5. The observer camera only records; the policy never sees
its images (FR-POL-04). Requirement served: FR-EVAL-06 (groundwork, milestone 0.3e).

Usage
    recorder = ObserverRecorder(camera="agentview", width=256, height=256, fps=20,
                                out_dir=Path("outputs/<run>/videos"))
    recorder.start(EpisodeMeta(episode_id="ep0", seed=0, suite="nominal"))
    for each control step:
        task.step(action)
        recorder.capture(task)
    record = recorder.finish()      # writes <episode_id>.mp4 and <episode_id>.json

Key data structures
  EpisodeMeta  what the caller knows about the episode (IDs, seed, suite, policy hash)
  VideoRecord  what was written (paths, frame count) plus all metadata fields

Pitfalls
  - One frame per control step: fps equals the control frequency (20 Hz for Can).
  - Frames come upright from `TaskInterface.render`; the recorder never flips them.
  - Write only under the given out_dir (inside outputs/ or a test temp dir).
  - Encode with imageio + imageio-ffmpeg (H.264 MP4) [candidate tooling, setup.md §2.1].
  - Record the rendering backend actually used (e.g. from MUJOCO_GL, or "default").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from afterslip.tasks.interface import TaskInterface

# Metadata keys every sidecar must contain (design §5.2; test-plan T0.15).
REQUIRED_METADATA_KEYS = (
    "episode_id",
    "bank_state_id",
    "seed",
    "suite",
    "policy_hash",
    "camera",
    "fps",
    "width",
    "height",
    "backend",
    "frame_count",
    "episode_steps",
    "git_commit",
)


@dataclass(frozen=True)
class EpisodeMeta:
    episode_id: str
    seed: int
    suite: str  # e.g. "nominal", "disturbed", "fixture"
    bank_state_id: Optional[str] = None
    policy_hash: Optional[str] = None


@dataclass(frozen=True)
class VideoRecord:
    video_path: Path
    metadata_path: Path
    frame_count: int


class ObserverRecorder:
    def __init__(self, camera: str, width: int, height: int, fps: float, out_dir: Path) -> None:
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps
        self.out_dir = Path(out_dir)
        self._frames: List[np.ndarray] = []
        self._meta: Optional[EpisodeMeta] = None

    def start(self, meta: EpisodeMeta) -> None:
        """Begin a new video; discard any frames from a previous episode."""
        # TODO(D1)
        raise NotImplementedError("D1: start")

    def capture(self, task: TaskInterface) -> None:
        """Render one frame from the observer camera and keep it."""
        # TODO(D1)
        raise NotImplementedError("D1: capture")

    def finish(self) -> VideoRecord:
        """Encode the MP4, write the JSON sidecar with REQUIRED_METADATA_KEYS, return the record."""
        # TODO(D1): episode_steps equals the number of captured frames (one per step);
        # git_commit can come from `git rev-parse HEAD` or be null outside a checkout.
        raise NotImplementedError("D1: finish")
