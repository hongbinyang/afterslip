"""The pinned upstream code installs and runs the Can task (milestone 0.2a; test-plan T0.4).

Imports the ReinFlow modules used for flow-policy pretraining, fine-tuning and evaluation,
and builds the state-based Can environment through ReinFlow's own environment factory,
with its own environment metadata (`cfg/robomimic/env_meta/can.json`).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from afterslip.upstream import configure_reinflow

pytestmark = pytest.mark.integration

REINFLOW_DIR = configure_reinflow()

# State observation of the upstream state-based Can configs (obs_dim 23).
LOW_DIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
OBS_DIM = 23
ACTION_DIM = 7


@pytest.mark.parametrize(
    "module",
    [
        "model.flow.reflow",
        "agent.pretrain.train_reflow_agent",
        "agent.finetune.reinflow.train_ppo_flow_agent",
        "agent.eval.eval_reflow_agent",
        "env.gym_utils",
    ],
)
def test_upstream_module_imports(module):
    importlib.import_module(module)


def test_can_environment_through_upstream_factory():
    from env.gym_utils import make_async
    from omegaconf import OmegaConf

    wrappers = OmegaConf.create(
        {
            "robomimic_lowdim": {"normalization_path": None, "low_dim_keys": LOW_DIM_KEYS},
            "multi_step": {"n_obs_steps": 1, "n_action_steps": 1, "max_episode_steps": 300},
        }
    )
    env = make_async(
        "can",
        num_envs=1,
        asynchronous=False,
        wrappers=wrappers,
        robomimic_env_cfg_path=str(REINFLOW_DIR / "cfg/robomimic/env_meta/can.json"),
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        max_episode_steps=300,
    )
    try:
        env.seed([0])
        obs = env.reset()
        assert obs["state"].shape == (1, 1, OBS_DIM)
        # Upstream's vector env returns (obs, reward, terminated, truncated, info).
        obs, reward, terminated, truncated, _info = env.step(np.zeros((1, 1, ACTION_DIM)))
        assert obs["state"].shape == (1, 1, OBS_DIM)
        assert np.isfinite(obs["state"]).all()
        assert reward.shape == terminated.shape == truncated.shape == (1,)
    finally:
        env.close()
