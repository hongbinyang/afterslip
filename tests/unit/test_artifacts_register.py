"""The artifact register lists every locked Python package (ARTIFACTS.md Appendix A; T0.3).

A lock change without the matching register update fails here, so both land in one commit.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
LOCKS = ("requirements-linux.lock", "requirements-macos.lock")


def _normalize(name: str) -> str:
    """PEP 503 name normalization: case-insensitive, runs of -, _ and . are equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_packages() -> set[tuple[str, str]]:
    """(name, version) pairs; a package installed from git is pinned by its commit."""
    by_version = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)==([^\s;\\]+)", re.MULTILINE)
    by_commit = re.compile(
        r"^([A-Za-z0-9][A-Za-z0-9_.\-]*) @ git\+\S+@([0-9a-f]{40})$", re.MULTILINE
    )
    found = set()
    for lock in LOCKS:
        text = (ROOT / lock).read_text()
        for name, version in by_version.findall(text) + by_commit.findall(text):
            found.add((_normalize(name), version))
    return found


def _registered_packages() -> set[tuple[str, str]]:
    text = (ROOT / "ARTIFACTS.md").read_text()
    appendix = text.split("## Appendix A.", 1)[1]
    rows = re.findall(r"^\| ([^|\s]+) \| ([^|\s]+) \|", appendix, re.MULTILINE)
    return {(_normalize(name), version) for name, version in rows if name != "Package"}


def test_every_locked_package_is_registered():
    missing = _locked_packages() - _registered_packages()
    assert not missing, f"add to ARTIFACTS.md Appendix A: {sorted(missing)}"


def test_register_lists_no_package_outside_the_locks():
    stale = _registered_packages() - _locked_packages()
    assert not stale, f"remove or update in ARTIFACTS.md Appendix A: {sorted(stale)}"
