"""Self-recording runs (afterslip.runs): manifest, console log and summary."""

from __future__ import annotations

import json

import pytest

from afterslip.evaluation.nominal import wilson_interval
from afterslip.runs import RunRecord

pytestmark = pytest.mark.unit


def _read(folder, name):
    return json.loads((folder / name).read_text())


def test_complete_run_writes_manifest_console_and_summary(tmp_path):
    with RunRecord(tmp_path, kind="test", config={"a": 1}, device="cpu") as record:
        print("hello from the run")
        record.finish({"value": 3})
    manifest = _read(tmp_path, "manifest.json")
    assert manifest["kind"] == "test" and manifest["config"] == {"a": 1}
    assert {"commit", "modified", "reinflow_commit"} <= set(manifest["code"])
    assert manifest["versions"]["python"]
    assert "hello from the run" in (tmp_path / "console.log").read_text()
    summary = _read(tmp_path, "summary.json")
    assert summary["status"] == "complete" and summary["result"] == {"value": 3}


def test_interrupted_run_records_its_status_and_progress(tmp_path):
    with pytest.raises(KeyboardInterrupt), RunRecord(tmp_path, kind="test") as record:
        record.progress({"episodes_completed": 4})
        raise KeyboardInterrupt
    summary = _read(tmp_path, "summary.json")
    assert summary["status"] == "interrupted"
    assert summary["progress"] == {"episodes_completed": 4} and summary["result"] is None


def test_failed_run_records_the_error(tmp_path):
    with pytest.raises(ValueError), RunRecord(tmp_path, kind="test"):
        raise ValueError("boom")
    summary = _read(tmp_path, "summary.json")
    assert summary["status"] == "failed" and "boom" in summary["error"]


def test_a_run_never_writes_into_a_folder_with_files(tmp_path):
    earlier = tmp_path / "summary.json"
    earlier.write_text('{"status": "complete"}')
    with pytest.raises(FileExistsError), RunRecord(tmp_path, kind="test"):
        pass
    assert earlier.read_text() == '{"status": "complete"}'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["summary.json"]


@pytest.mark.parametrize(
    "successes,n,expected",
    [(180, 200, (0.8506, 0.9343)), (0, 3, (0.0, 0.5615)), (200, 200, (0.9812, 1.0))],
)
def test_wilson_interval_matches_known_values(successes, n, expected):
    low, high = wilson_interval(successes, n)
    assert (round(low, 4), round(high, 4)) == expected
