"""State-based pretraining config mapping (milestone 0.2b; FR-ART-03; docs/design.md §10).

Each value of `configs/pretrain/can_reflow_mlp_state.yaml` is checked against its source:
our task and data (`CanTaskConfig`, `afterslip.data.can_demos`) and the upstream Can
configs it was derived from. A short CPU run checks that upstream pretraining runs on it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from afterslip.data import can_demos
from afterslip.tasks.robosuite_can import CanTaskConfig
from afterslip.upstream import REINFLOW_DIR
from afterslip.upstream.pretrain import REPO_CONFIGS, load_config, run

pytestmark = pytest.mark.integration

CONFIG = REPO_CONFIGS / "pretrain" / "can_reflow_mlp_state.yaml"
UPSTREAM = REINFLOW_DIR / "cfg" / "robomimic"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


def _upstream(relative: str):
    return OmegaConf.load(UPSTREAM / relative)


def test_observation_and_action_match_task_and_data(cfg):
    assert cfg.obs_dim == can_demos.OBS_DIM == 23
    assert cfg.action_dim == can_demos.ACTION_DIM == 7
    assert cfg.model.network.cond_dim == cfg.obs_dim * cfg.cond_steps
    # Actions and observations are normalized to [-1, 1] by the data pipeline.
    assert (cfg.model.act_min, cfg.model.act_max) == (-1, 1)


def test_chunking_and_history_match_the_upstream_can_configs(cfg):
    image_reflow = _upstream("pretrain/can/pre_reflow_mlp_img.yaml")
    state_diffusion = _upstream("pretrain/can/pre_diffusion_mlp.yaml")
    finetune = _upstream("finetune/can/ft_ppo_reflow_mlp_img.yaml")
    assert cfg.horizon_steps == image_reflow.horizon_steps == state_diffusion.horizon_steps
    assert cfg.horizon_steps == finetune.act_steps  # the whole chunk is executed
    assert cfg.cond_steps == image_reflow.cond_steps == state_diffusion.cond_steps == 1
    assert state_diffusion.obs_dim == cfg.obs_dim  # upstream's own state dimension for Can


def test_model_matches_the_upstream_state_reflow_structure(cfg):
    transport = _upstream("pretrain/transport/pre_reflow_mlp.yaml")
    image_reflow = _upstream("pretrain/can/pre_reflow_mlp_img.yaml")
    assert cfg.model._target_ == transport.model._target_ == "model.flow.reflow.ReFlow"
    assert cfg.model.network._target_ == transport.model.network._target_
    # Width and time embedding follow the Can image config, not the larger Transport task.
    assert list(cfg.model.network.mlp_dims) == list(image_reflow.model.network.mlp_dims)
    assert cfg.model.network.time_dim == image_reflow.model.network.time_dim
    assert cfg.denoising_steps == image_reflow.denoising_steps
    for key in ("learning_rate", "weight_decay", "n_epochs"):
        assert cfg.train[key] == image_reflow.train[key], key


def test_environment_settings_match_the_task(cfg):
    meta = json.loads((UPSTREAM / "env_meta" / "can.json").read_text())
    finetune = _upstream("finetune/can/ft_ppo_reflow_mlp_img.yaml")
    task = CanTaskConfig()
    assert meta["env_name"] == task.env_name
    assert meta["env_kwargs"]["robots"] == [task.robot]
    assert meta["env_kwargs"]["control_freq"] == task.control_freq_hz
    assert meta["env_kwargs"]["controller_configs"]["type"] == task.controller
    assert finetune.env.max_episode_steps == task.horizon


def test_paths_point_to_the_regenerated_data(cfg):
    out = can_demos.DEFAULT_OUT_DIR
    assert cfg.train_dataset_path == str(out / "train.npz")
    assert cfg.normalization_path == str(out / "normalization.npz")
    assert cfg.val_dataset_path == str(out / "val.npz")  # held-out, never trained on
    assert cfg.wandb is None and cfg.test_in_mujoco is False


def test_default_run_directory_resolves_once(cfg):
    # The default logdir contains a ${now:...} timestamp; it must resolve without the
    # upstream launcher and stay fixed, so training and held-out scoring share one folder.
    import os

    assert "${" not in cfg.logdir
    assert cfg.logdir.startswith(os.environ["REINFLOW_LOG_DIR"])
    assert cfg.logdir.endswith(f"_{cfg.seed}")
    # Fixed as a plain string in this config: every later read gives the same folder, even
    # after the clock moves on (a fresh load_config may resolve a new timestamp).
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(raw, dict) and raw["logdir"] == cfg.logdir


def test_normalization_file_matches_the_config(cfg):
    path = can_demos.DEFAULT_OUT_DIR / "normalization.npz"
    if not path.is_file():
        pytest.skip("data not prepared; run `make data-can`")
    stats = np.load(path)
    assert stats["obs_min"].shape == stats["obs_max"].shape == (cfg.obs_dim,)
    assert stats["action_min"].shape == (cfg.action_dim,)


def test_upstream_pretraining_runs_on_the_config(tmp_path):
    if not (can_demos.DEFAULT_OUT_DIR / "train.npz").is_file():
        pytest.skip("data not prepared; run `make data-can`")
    overrides = [
        "device=cpu",
        "batch_size=64",
        "train.n_epochs=1",
        "train.save_model_freq=1",
        "train_dataset.max_n_episodes=2",
        f"logdir={tmp_path}",
    ]
    report = run(load_config(CONFIG, overrides))
    assert (tmp_path / "checkpoint" / "last.pt").is_file()

    # The final checkpoint is scored on every chunk of the held-out file, and only there.
    assert report is not None and (tmp_path / "heldout_validation.json").is_file()
    val = np.load(can_demos.DEFAULT_OUT_DIR / "val.npz")
    horizon = load_config(CONFIG).horizon_steps
    expected = int(sum(n - horizon + 1 for n in val["traj_lengths"]))
    # The report names the run folder and the exact checkpoint that was scored.
    import hashlib

    assert report["logdir"] == str(tmp_path)
    last = tmp_path / "checkpoint" / "last.pt"
    assert report["checkpoint_sha256"] == hashlib.sha256(last.read_bytes()).hexdigest()
    for weights in ("model", "ema"):
        assert report[weights]["dataset"].endswith("val.npz")
        assert report[weights]["samples"] == expected
        assert np.isfinite(report[weights]["loss"])
