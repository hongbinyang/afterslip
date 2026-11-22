"""Self-recording runs: every long run writes its own record into its run folder (FR-TRK-01).

    manifest.json  at the start: code revision, command, config, device, versions, machine
    console.log    everything the run prints (stdout and stderr), as it happens
    summary.json   at the end: status ("complete", "interrupted" or "failed"), result, duration

Usage:

    with RunRecord(folder, kind="evaluation", config=cfg_dict, device="mps") as record:
        ...                          # the run
        record.finish(result_dict)   # what summary.json reports as the result

If the block ends without `finish` (Ctrl-C or an error), summary.json still records the
status, the error and the last `record.progress(...)`, and the exception continues to
propagate.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "numpy", "mujoco", "robosuite", "robomimic")


def _git(*args: str, cwd: Path = REPO_ROOT) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def code_revision() -> dict[str, Any]:
    """Commit of the checkout and of the upstream submodule, and whether either was modified."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "modified": None if status is None else bool(status),
        "reinflow_commit": _git("rev-parse", "HEAD", cwd=REPO_ROOT / "third_party" / "ReinFlow"),
    }


def versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


class _Tee:
    """Writes to the original stream and to the run's console log."""

    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self._stream, self._log = stream, log

    def write(self, text: str) -> int:
        self._log.write(text)
        self._log.flush()
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class RunRecord:
    def __init__(
        self, folder: str | Path, kind: str, config: dict | None = None, device: str | None = None
    ) -> None:
        self.folder = Path(folder)
        self.kind = kind
        self.config = config
        self.device = device
        self._result: dict | None = None
        self._progress: dict | None = None
        self._start = 0.0

    def __enter__(self) -> RunRecord:
        # Never write into an existing run: earlier evidence must survive. An existing
        # empty folder is accepted; anything in it stops the run before any file is written.
        if self.folder.exists() and any(self.folder.iterdir()):
            raise FileExistsError(f"{self.folder} is not empty; choose a new run folder")
        self.folder.mkdir(parents=True, exist_ok=True)
        self._start = time.monotonic()
        manifest = {
            "kind": self.kind,
            "started": _now(),
            "command": sys.argv,
            "code": code_revision(),
            "config": self.config,
            "device": self.device,
            "versions": versions(),
            "machine": {"platform": platform.platform(), "processor": platform.processor()},
        }
        self._write("manifest.json", manifest)
        self._log = open(self.folder / "console.log", "a", buffering=1)
        self._streams = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(sys.stdout, self._log)  # type: ignore[assignment]
        sys.stderr = _Tee(sys.stderr, self._log)  # type: ignore[assignment]
        return self

    def progress(self, partial: dict) -> None:
        """What has been done so far; kept in summary.json if the run stops early."""
        self._progress = partial

    def finish(self, result: dict) -> None:
        self._result = result

    def __exit__(self, exc_type, exc, tb) -> None:
        sys.stdout, sys.stderr = self._streams
        self._log.close()
        if exc_type is None and self._result is not None:
            status, error = "complete", None
        elif exc_type is KeyboardInterrupt:
            status, error = "interrupted", "stopped by the user (Ctrl-C)"
        elif exc_type is None:
            status, error = "failed", "the run ended without a result"
        else:
            status, error = "failed", f"{exc_type.__name__}: {exc}"
        self._write(
            "summary.json",
            {
                "kind": self.kind,
                "status": status,
                "error": error,
                "finished": _now(),
                "duration_s": round(time.monotonic() - self._start, 1),
                "result": self._result,
                "progress": None if status == "complete" else self._progress,
            },
        )

    def _write(self, name: str, data: dict) -> None:
        (self.folder / name).write_text(json.dumps(data, indent=2, default=str) + "\n")
