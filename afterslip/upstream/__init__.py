"""Access to the pinned upstream reference code (ReinFlow, `third_party/ReinFlow`).

ReinFlow is installed editable without its dependencies (`make install`) and imports its
own top-level packages (`agent`, `model`, `env`, `util`). Its modules read three
environment variables at import time; `configure_reinflow()` sets them and must run before
any ReinFlow import:

  REINFLOW_DIR       the submodule checkout; upstream checks it equals the code location
  REINFLOW_DATA_DIR  downloaded datasets and normalization files (default: outputs/reinflow/data)
  REINFLOW_LOG_DIR   checkpoints and logs (default: outputs/reinflow/log)

Data and log defaults live under the git-ignored `outputs/`; values already set in the
environment are kept. The upstream training code imports wandb at module level; wandb is
switched off (`WANDB_MODE=disabled`, no error reporting) unless the caller enables it.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REINFLOW_DIR = REPO_ROOT / "third_party" / "ReinFlow"


def configure_reinflow() -> Path:
    """Set the variables ReinFlow expects; return the ReinFlow checkout."""
    if not (REINFLOW_DIR / "pyproject.toml").is_file():
        raise RuntimeError(f"{REINFLOW_DIR} is missing; run: git submodule update --init")
    os.environ["REINFLOW_DIR"] = str(REINFLOW_DIR)
    os.environ.setdefault("REINFLOW_DATA_DIR", str(REPO_ROOT / "outputs" / "reinflow" / "data"))
    os.environ.setdefault("REINFLOW_LOG_DIR", str(REPO_ROOT / "outputs" / "reinflow" / "log"))
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_ERROR_REPORTING", "false")
    return REINFLOW_DIR
