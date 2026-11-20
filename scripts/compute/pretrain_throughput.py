"""Measure state-based pretraining throughput on one device (milestone 0.7).

Runs the project pretraining config on the full regenerated Can data for a few epochs and
prints the wall-clock time. Needs `make data-can`. Results: docs/setup.md §6.6.

    python scripts/compute/pretrain_throughput.py cpu 20
    python scripts/compute/pretrain_throughput.py mps 20
"""

from __future__ import annotations

import json
import sys
import tempfile
import time


def main() -> None:
    from afterslip.upstream.pretrain import REPO_CONFIGS, load_config, run

    device, epochs = sys.argv[1], int(sys.argv[2])
    overrides = [
        f"device={device}",
        f"train.n_epochs={epochs}",
        "train.save_model_freq=100000",  # no intermediate checkpoints
        f"logdir={tempfile.mkdtemp()}",
        "val_dataset_path=null",  # time training only
    ]
    cfg = load_config(REPO_CONFIGS / "pretrain" / "can_reflow_mlp_state.yaml", overrides)
    start = time.perf_counter()
    run(cfg)
    seconds = time.perf_counter() - start
    print(json.dumps({"device": device, "epochs": epochs, "seconds": round(seconds, 2)}))


# Required: on macOS the data-loader workers start new Python processes that re-import this
# file; without the guard each worker would start another training run and the run hangs.
if __name__ == "__main__":
    main()
