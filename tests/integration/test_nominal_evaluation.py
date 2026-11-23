"""Nominal evaluation harness (milestone 0.3b; docs/m0-evaluation-protocol.md).

Checks the inference mapping against the training data, the separation of policy and
scene randomness, and a short end-to-end trial with a tiny checkpoint trained here. The
real policy is never evaluated in tests, and trials never use the validation seeds.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from afterslip.data import can_demos
from afterslip.evaluation import nominal
from afterslip.tasks.robosuite_can import CanTaskConfig, RobosuiteCanTask
from afterslip.upstream.pretrain import REPO_CONFIGS, load_config, run

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """A 1-epoch checkpoint on 2 demos: exercises the plumbing, not the policy's quality."""
    if not (can_demos.DEFAULT_OUT_DIR / "train.npz").is_file():
        pytest.skip("data not prepared; run `make data-can`")
    folder = tmp_path_factory.mktemp("tiny-pretrain")
    overrides = [
        "device=cpu",
        "batch_size=64",
        "train.n_epochs=1",
        "train_dataset.max_n_episodes=2",
        f"logdir={folder}",
        "val_dataset_path=null",
    ]
    run(load_config(REPO_CONFIGS / "pretrain" / "can_reflow_mlp_state.yaml", overrides))
    return folder


@pytest.fixture(scope="module")
def policy(tiny_run):
    return nominal.Policy(tiny_run / "checkpoint" / "last.pt", device="cpu")


def test_observation_mapping_matches_the_training_data(policy):
    # The policy normalizes a live observation exactly as the training data was normalized.
    raw = can_demos.DEFAULT_RAW_DIR / "can_ph_low_dim_v15.hdf5"
    train = np.load(can_demos.DEFAULT_OUT_DIR / "train.npz")
    env = RobosuiteCanTask(CanTaskConfig()).benchmark_env
    env.reset()
    with h5py.File(raw, "r") as f:
        first_train_demo = sorted(
            (n.decode() for n in f["mask/train"][()]), key=lambda n: int(n[5:])
        )[0]
        state0 = f[f"data/{first_train_demo}/states"][0]
    env.sim.set_state_from_flattened(state0)
    env.sim.forward()
    vector = can_demos.observation_vector(env._get_observations(force_update=True))
    normalized = 2 * ((vector - policy.obs_min) / (policy.obs_max - policy.obs_min + 1e-6) - 0.5)
    np.testing.assert_allclose(normalized, train["states"][0], atol=1e-6)


def test_action_mapping_inverts_the_training_normalization(policy):
    train = np.load(can_demos.DEFAULT_OUT_DIR / "train.npz")
    stats = np.load(can_demos.DEFAULT_OUT_DIR / "normalization.npz")
    normalized = train["actions"][:50]
    recovered = (normalized + 1) / 2 * (policy.act_max - policy.act_min) + policy.act_min
    original = (normalized + 1) / 2 * (stats["action_max"] - stats["action_min"] + 1e-6) + stats[
        "action_min"
    ]
    np.testing.assert_allclose(recovered, original, atol=1e-5)  # upstream omits the 1e-6


def test_policy_noise_is_seeded_and_never_touches_numpy(policy):
    task = RobosuiteCanTask(CanTaskConfig())
    raw = task.reset(seed=0)
    numpy_state = np.random.get_state()
    a = policy.chunk(raw, torch.Generator().manual_seed(1_000_000))
    b = policy.chunk(raw, torch.Generator().manual_seed(1_000_000))
    c = policy.chunk(raw, torch.Generator().manual_seed(1_000_001))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (4, 7)
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1])


def test_trial_run_writes_the_full_record(tiny_run, tmp_path):
    result = nominal.evaluate(tiny_run, tmp_path, episodes=2, device="cpu")
    lines = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert [e["seed"] for e in lines] == [0, 1]  # fixture seeds, never validation seeds
    assert all(e["policy_seed"] == e["seed"] + nominal.POLICY_SEED_OFFSET for e in lines)
    design_fields = {
        "episode_id",
        "suite",
        "seed",
        "bank_state_id",
        "policy_hash",
        "success",
        "steps",
        "timeout",
        "video",
        "git_commit",
    }  # docs/design.md §7
    assert all(set(e) == design_fields | {"policy_seed", "error"} for e in lines)
    assert all(e["suite"] == "trial" and e["bank_state_id"] is None for e in lines)
    assert all(e["policy_hash"] == result["checkpoint_sha256"] for e in lines)
    assert all(e["timeout"] == (not e["success"] and e["steps"] == 300) for e in lines)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["result"] == result
    assert result["protocol_run"] is False and result["passed"] is None
    assert result["episodes_completed"] == 2 and len(result["wilson_95"]) == 2
    assert result["inference_steps"] == 1 and result["weights"] == "ema"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["kind"] == "evaluation" and manifest["config"]["checkpoint_sha256"]
    assert "episode 2/2" in (tmp_path / "console.log").read_text()


def test_an_episode_error_stops_the_run_without_a_verdict(tiny_run, tmp_path, monkeypatch):
    calls = {"n": 0}
    real_chunk = nominal.Policy.chunk

    def failing_chunk(self, raw, generator):
        calls["n"] += 1
        if calls["n"] > 3:  # the first episode needs more than 3 chunks, so it fails there
            raise RuntimeError("injected failure")
        return real_chunk(self, raw, generator)

    monkeypatch.setattr(nominal.Policy, "chunk", failing_chunk)
    with pytest.raises(nominal.EpisodeError):
        nominal.evaluate(tiny_run, tmp_path, episodes=3, device="cpu")
    lines = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert len(lines) == 1 and "injected failure" in lines[0]["error"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "failed" and "injected failure" in summary["error"]
    assert summary["result"] is None  # no rate, no verdict
    assert summary["progress"]["episodes_completed"] == 1


def test_evaluation_refuses_an_existing_run_folder(tiny_run, tmp_path):
    (tmp_path / "episodes.jsonl").write_text("earlier evidence\n")
    with pytest.raises(FileExistsError):
        nominal.evaluate(tiny_run, tmp_path, episodes=1, device="cpu")
    assert (tmp_path / "episodes.jsonl").read_text() == "earlier evidence\n"


def _older_run(tiny_run, tmp_path):
    """A copy of the run without its record, like pretraining runs from before 0.3b."""
    import shutil

    older = tmp_path / "older-run"
    shutil.copytree(tiny_run / "checkpoint", older / "checkpoint")
    return older


def test_pretraining_records_its_normalization_hash(tiny_run):
    summary = json.loads((tiny_run / "summary.json").read_text())
    stats = can_demos.DEFAULT_OUT_DIR / "normalization.npz"
    assert summary["result"]["normalization_sha256"] == nominal.sha256(stats)


def test_a_wrong_normalization_hash_stops_before_any_seed(tiny_run, tmp_path):
    out = tmp_path / "eval"
    with pytest.raises(nominal.ProvenanceError):
        nominal.evaluate(tiny_run, out, episodes=1, normalization_sha256="0" * 64)
    assert not out.exists()  # no folder, no episode


def test_protocol_run_without_a_known_hash_stops_before_any_seed(tiny_run, tmp_path):
    older = _older_run(tiny_run, tmp_path)
    out = tmp_path / "eval"
    with pytest.raises(nominal.ProvenanceError, match="does not record"):
        nominal.evaluate(older, out)  # protocol run: 200 validation seeds, never reached
    with pytest.raises(nominal.ProvenanceError, match="not the file"):
        nominal.evaluate(older, out, normalization_sha256="0" * 64)
    assert not out.exists()


def test_a_given_hash_is_checked_and_labelled_as_inferred(tiny_run, tmp_path):
    older = _older_run(tiny_run, tmp_path)
    stats = can_demos.DEFAULT_OUT_DIR / "normalization.npz"
    out = tmp_path / "eval"
    result = nominal.evaluate(older, out, episodes=1, normalization_sha256=nominal.sha256(stats))
    assert result["normalization_expected_sha256"] == result["normalization_sha256"]
    assert result["normalization_provenance"].startswith("given on the command line")
    manifest = json.loads((out / "manifest.json").read_text())
    assert "not recorded by the pretraining run" in manifest["config"]["normalization_provenance"]


@pytest.fixture
def selection_file(tmp_path, monkeypatch):
    path = tmp_path / "selected-policy.json"
    monkeypatch.setattr(nominal, "SELECTION", path)
    return path


def test_selecting_a_policy_records_its_hashes_once(tiny_run, selection_file):
    chosen = nominal.select_policy(tiny_run)
    saved = json.loads(selection_file.read_text())
    assert saved == chosen and saved["run"] == str(tiny_run)
    assert saved["checkpoint_sha256"] == nominal.sha256(tiny_run / "checkpoint" / "last.pt")
    assert saved["normalization_provenance"] == "recorded by the pretraining run"
    with pytest.raises(FileExistsError, match="already selected"):
        nominal.select_policy(tiny_run)
    assert nominal.select_policy(tiny_run, replace=True)["run"] == str(tiny_run)


def test_selecting_an_older_run_needs_the_normalization_hash(tiny_run, tmp_path, selection_file):
    older = _older_run(tiny_run, tmp_path)
    with pytest.raises(nominal.ProvenanceError, match="does not record"):
        nominal.select_policy(older)
    assert not selection_file.exists()
    stats = can_demos.DEFAULT_OUT_DIR / "normalization.npz"
    chosen = nominal.select_policy(older, normalization_sha256=nominal.sha256(stats))
    assert chosen["normalization_provenance"].startswith("given on the command line")


def test_evaluation_without_a_selection_stops(selection_file):
    with pytest.raises(nominal.ProvenanceError, match="no policy selected"):
        nominal.main(["--episodes", "1"])


def test_a_checkpoint_changed_after_selection_stops_before_any_seed(tiny_run, tmp_path):
    out = tmp_path / "eval"
    with pytest.raises(nominal.ProvenanceError, match="changed since it was selected"):
        nominal.evaluate(tiny_run, out, episodes=1, expected_checkpoint="0" * 64)
    assert not out.exists()


def test_the_default_command_evaluates_the_selected_policy(tiny_run, tmp_path, selection_file):
    nominal.select_policy(tiny_run)
    out = tmp_path / "eval"
    nominal.main(["--episodes", "1", "--out", str(out)])  # trial: fixture seed 0 only
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"]["pretraining_run"] == str(tiny_run)
    assert manifest["config"]["pretraining_run_chosen_by"].startswith("selected policy")


def _finished_copy(tiny_run, root, stamp):
    import shutil

    run = root / "can_pre_reflow_mlp_state_ta4_td100" / stamp
    shutil.copytree(tiny_run, run)
    return run


def test_selection_without_run_needs_exactly_one_finished_run(tiny_run, tmp_path, selection_file):
    root = tmp_path / "pretrain"
    only = _finished_copy(tiny_run, root, "2026-11-20_16-13-03_42")
    assert nominal.select_policy(log_root=root)["run"] == str(only)
    _finished_copy(tiny_run, root, "2026-11-22_08-00-00_42")
    with pytest.raises(nominal.SelectionError, match="name one with RUN="):
        nominal.select_policy(log_root=root, replace=True)
    assert json.loads(selection_file.read_text())["run"] == str(only)  # unchanged


def test_selection_verifies_an_expected_checkpoint_hash(tiny_run, selection_file):
    with pytest.raises(nominal.ProvenanceError, match="expected"):
        nominal.select_policy(tiny_run, checkpoint_sha256="0" * 64)
    assert not selection_file.exists()
    good = nominal.sha256(tiny_run / "checkpoint" / "last.pt")
    chosen = nominal.select_policy(tiny_run, checkpoint_sha256=good)
    assert chosen["checkpoint_sha256_verified"] is True
