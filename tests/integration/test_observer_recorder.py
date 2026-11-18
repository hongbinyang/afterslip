"""Done-definition for increment D1, observer recorder (docs/design.md §5; test-plan T0.15).

Fails until `ObserverRecorder` and `RobosuiteCanTask` are implemented.
"""

from __future__ import annotations

import json

import imageio
import numpy as np
import pytest

from afterslip.recording.recorder import REQUIRED_METADATA_KEYS, EpisodeMeta, ObserverRecorder
from afterslip.tasks.robosuite_can import CanTaskConfig, RobosuiteCanTask

pytestmark = pytest.mark.integration

STEPS = 20


def test_recorder_writes_readable_video_and_metadata(tmp_path):
    task = RobosuiteCanTask(CanTaskConfig())
    recorder = ObserverRecorder(camera="agentview", width=128, height=128, fps=20, out_dir=tmp_path)

    task.reset(seed=0)
    recorder.start(EpisodeMeta(episode_id="fixture-0", seed=0, suite="fixture"))
    rng = np.random.default_rng(0)
    low, high = task.benchmark_env.action_spec
    for _ in range(STEPS):
        task.step(rng.uniform(low, high) * 0.3)
        recorder.capture(task)
    record = recorder.finish()

    # Files are written only under out_dir.
    assert record.video_path.parent == tmp_path and record.metadata_path.parent == tmp_path

    frames = np.stack(imageio.mimread(record.video_path, memtest=False)).astype(np.float32)
    assert frames.shape[0] == STEPS == record.frame_count
    assert (frames.std(axis=(1, 2)).max(axis=1) > 1.0).all(), "a frame is blank"
    assert np.abs(frames[-1] - frames[0]).mean() > 0.5, "the scene does not change"

    meta = json.loads(record.metadata_path.read_text())
    assert set(REQUIRED_METADATA_KEYS) <= set(meta)
    assert meta["camera"] == "agentview" and meta["fps"] == 20
    assert meta["frame_count"] == meta["episode_steps"] == STEPS
