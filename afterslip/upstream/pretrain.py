"""Run upstream pretraining with a project config, then score it on held-out demos (0.2b).

Upstream's own launcher (`third_party/ReinFlow/script/run.py`) imports d4rl, which is
not installed (docs/setup.md §2.2), so this module does the same job: load the config,
apply `key=value` overrides, instantiate the agent named by `_target_`, and run it.

Held-out validation: upstream's trainer only validates on a random split of its own
training file (`train.train_split`), and that code path is broken at the pinned commit
(`with torch.no_grad:` without a call). So after training, this module computes the flow
loss of the final checkpoint on `val_dataset_path` (robomimic's `valid` demos, never seen
in training or normalization) and writes it to `<logdir>/heldout_validation.json`.

    python -m afterslip.upstream.pretrain configs/pretrain/can_reflow_mlp_state.yaml device=cpu
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from afterslip.upstream import REPO_ROOT, configure_reinflow

REPO_CONFIGS = REPO_ROOT / "configs"


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """The config with overrides applied; interpolations resolve when values are read."""
    configure_reinflow()
    import hydra.core.utils  # noqa: F401  (registers Hydra's `now` resolver)

    # Resolvers that upstream's launcher registers.
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    assert isinstance(cfg, DictConfig)
    return cfg


def heldout_loss(
    cfg: DictConfig, checkpoint: Path, weights: str = "ema", passes: int = 5, seed: int = 0
) -> dict:
    """Mean flow-matching loss of a checkpoint on the held-out demos.

    The loss samples a random time and noise per example, so it is averaged over `passes`
    sweeps with a fixed seed; the same seed makes checkpoints comparable.
    """
    import hydra.utils
    import torch
    from agent.pretrain import utils

    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(checkpoint, map_location=cfg.device, weights_only=True)
    model.load_state_dict(state[weights])
    model.eval()
    dataset = hydra.utils.instantiate(
        cfg.train_dataset, dataset_path=cfg.val_dataset_path, max_n_episodes=-1
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    torch.manual_seed(seed)
    total, count = 0.0, 0
    with torch.no_grad():
        for _ in range(passes):
            for batch in loader:
                actions, cond = utils.batch_to_device(batch, device=cfg.device)
                (xt, t), v = model.generate_target(actions)
                total += model.loss(xt, t, cond, v).item() * len(actions)
                count += len(actions)
    return {
        "checkpoint": str(checkpoint),
        "weights": weights,
        "dataset": str(cfg.val_dataset_path),
        "samples": len(dataset),
        "passes": passes,
        "seed": seed,
        "loss": total / count,
    }


def run(cfg: DictConfig) -> dict | None:
    """Train; then, if the config names held-out data, score the final checkpoint on it."""
    import functools

    import hydra.utils
    from agent.pretrain import train_agent, utils

    # Upstream's training loop calls batch_to_device(batch) without a device, and that
    # helper defaults to "cuda:0", so CPU and MPS runs fail. Bind the configured device;
    # the upstream code itself stays unmodified (docs/setup.md §2.2).
    train_agent.batch_to_device = functools.partial(utils.batch_to_device, device=cfg.device)

    agent_class = hydra.utils.get_class(cfg._target_)
    agent_class(cfg).run()

    if cfg.get("val_dataset_path") is None:
        return None
    last = Path(cfg.logdir) / "checkpoint" / "last.pt"
    report = {weights: heldout_loss(cfg, last, weights) for weights in ("model", "ema")}
    (Path(cfg.logdir) / "heldout_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit(__doc__)
    report = run(load_config(args[0], args[1:]))
    if report is not None:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
