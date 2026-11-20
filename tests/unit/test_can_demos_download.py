"""The demonstration download accepts only the pinned file and never hangs or leaves debris."""

from __future__ import annotations

import io
import socket

import pytest

from afterslip.data import can_demos

pytestmark = pytest.mark.unit


def _serve(payload: bytes):
    def fake_urlopen(url, timeout):
        assert timeout == can_demos.DOWNLOAD_TIMEOUT_S  # every download is bounded
        return io.BytesIO(payload)

    return fake_urlopen


def test_a_file_with_another_hash_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(can_demos.urllib.request, "urlopen", _serve(b"not the dataset"))
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        can_demos.download(tmp_path)
    # Nothing is left behind that a later run could mistake for the dataset.
    assert list(tmp_path.iterdir()) == []


def test_a_stalled_transfer_fails_and_cleans_up(tmp_path, monkeypatch):
    class Stalled(io.BytesIO):
        def read(self, size=-1):
            raise socket.timeout("timed out")

    monkeypatch.setattr(can_demos.urllib.request, "urlopen", lambda url, timeout: Stalled())
    with pytest.raises(socket.timeout):
        can_demos.download(tmp_path)
    assert list(tmp_path.iterdir()) == []
