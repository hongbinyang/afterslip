"""Observer recorder: one fixed third-person camera to MP4 plus a JSON metadata sidecar.

See docs/design.md §5. The policy never sees the observer camera (FR-POL-04).

Usage:
    recorder = ObserverRecorder(camera="agentview", width=256, height=256, fps=20,
                                out_dir=Path("outputs/<run>/videos"))
    recorder.start(EpisodeMeta(episode_id="ep0", seed=0, suite="nominal"))
    for each control step:
        task.step(action)
        recorder.capture(task)
    record = recorder.finish()  # writes <episode_id>.mp4 and <episode_id>.json

One frame is captured per control step, so fps equals the control frequency.
Files are written only under `out_dir`.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio
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
    "camera_pose",
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
    bank_state_id: str | None = None
    policy_hash: str | None = None


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
        self._frames: list[np.ndarray] = []
        self._meta: EpisodeMeta | None = None
        self._camera_pose: dict | None = None

    def start(self, meta: EpisodeMeta) -> None:
        """Begin a new video; discard any frames from a previous episode."""
        self._meta = meta
        self._frames.clear()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, task: TaskInterface) -> None:
        """Render one frame from the observer camera and keep it."""
        frame = task.render(
            camera=self.camera,
            width=self.width,
            height=self.height,
        )
        self._frames.append(frame)
        self._camera_pose = task.camera_pose(self.camera)

    def finish(self) -> VideoRecord:
        """Encode the MP4, write the JSON sidecar with REQUIRED_METADATA_KEYS, return the record."""
        if self._meta is None:
            raise RuntimeError("finish() called before start()")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        video_path = self.out_dir / f"{self._meta.episode_id}.mp4"
        metadata_path = self.out_dir / f"{self._meta.episode_id}.json"

        imageio.mimsave(video_path, self._frames, fps=self.fps)

        metadata = {
            "episode_id": self._meta.episode_id,
            "bank_state_id": self._meta.bank_state_id,
            "seed": self._meta.seed,
            "suite": self._meta.suite,
            "policy_hash": self._meta.policy_hash,
            "camera": self.camera,
            "camera_pose": self._camera_pose,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "backend": os.environ.get("MUJOCO_GL", "default"),
            "frame_count": len(self._frames),
            "episode_steps": len(self._frames),  # one frame per control step
            "git_commit": self._git_commit(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        return VideoRecord(
            video_path=video_path, metadata_path=metadata_path, frame_count=len(self._frames)
        )

    @staticmethod
    def _git_commit() -> str | None:
        """Current commit of the checkout, or None outside a git checkout."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()
