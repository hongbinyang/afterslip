"""Finished pretraining runs, the candidates for selecting the policy to evaluate."""

from __future__ import annotations

import json

import pytest

from afterslip.evaluation.nominal import finished_pretraining_runs

pytestmark = pytest.mark.unit

NAME = "can_pre_reflow_mlp_state_ta4_td100"


def _run(root, stamp, *, checkpoint=True, status=None, heldout=False):
    run = root / NAME / stamp
    (run / "checkpoint").mkdir(parents=True)
    if checkpoint:
        (run / "checkpoint" / "last.pt").write_bytes(b"x")
    if status:
        (run / "summary.json").write_text(json.dumps({"status": status}))
    if heldout:
        (run / "heldout_validation.json").write_text("{}")
    return run


def test_only_finished_runs_count(tmp_path):
    older = _run(tmp_path, "2026-11-20_16-13-03_42", heldout=True)  # before run records
    _run(tmp_path, "2026-11-21_09-00-00_42", status="interrupted")
    _run(tmp_path, "2026-11-21_09-30-00_42", status="failed")
    _run(tmp_path, "2026-11-21_10-00-00_42", status="complete", checkpoint=False)
    _run(tmp_path, "2026-11-21_11-00-00_42")  # still running: no summary, no report
    assert finished_pretraining_runs(tmp_path) == [older]
    newer = _run(tmp_path, "2026-11-22_08-00-00_42", status="complete")
    assert finished_pretraining_runs(tmp_path) == [older, newer]
