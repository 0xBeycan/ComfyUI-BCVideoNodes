"""The model download: a cut connection is resumed where it stopped, and when it cannot
finish it fails loudly, never leaving a truncated model under the model's name.

Needs ComfyUI (folder_paths, comfy.utils) and is skipped where it is not importable; on a
machine with ComfyUI run with it on the path:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/models/test_download.py
"""
import http.server
import re
import threading

import pytest

pytest.importorskip("folder_paths")

from bcvideonodes.models.common import download as dl  # noqa: E402

BODY = bytes(range(256)) * (3 << 12)   # 3 MiB, position-dependent so a misplaced resume shows


def serve(cut=None, ranges=True, full_on=None):
    """A local server for BODY. `cut`: bytes sent per connection before it closes early;
    `ranges`: whether it honours a Range header (206 from that offset); `full_on`: the
    connection number (from 1) that is sent whole regardless of `cut`."""
    connections = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            connections.append(self.headers.get("Range"))
            start = 0
            match = re.fullmatch(r"bytes=(\d+)-", self.headers.get("Range") or "")
            if ranges and match:
                start = int(match.group(1))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(BODY) - start))
            self.end_headers()
            whole = full_on is not None and len(connections) == full_on
            self.wfile.write(BODY[start:] if whole or cut is None else BODY[start:start + cut])
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/model.safetensors", connections


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(dl, "RETRY_WAIT", 0.0)


def test_a_complete_download_is_kept(tmp_path):
    server, url, _ = serve()
    try:
        path = tmp_path / "model.safetensors"
        dl.download(url, str(path))
        assert path.read_bytes() == BODY
        assert not (tmp_path / "model.safetensors.part").exists()
    finally:
        server.shutdown()


def test_a_cut_download_resumes_where_it_stopped(tmp_path):
    server, url, connections = serve(cut=len(BODY) // 3 + 1)
    try:
        path = tmp_path / "model.safetensors"
        dl.download(url, str(path))
        assert path.read_bytes() == BODY
        assert connections[0] is None and connections[1] == f"bytes={len(BODY) // 3 + 1}-"
        assert len(connections) == 3
    finally:
        server.shutdown()


def test_a_server_that_ignores_the_range_starts_over(tmp_path):
    # first connection cut, second answered 200 with the whole file: the .part is rewritten,
    # not appended to
    server, url, connections = serve(cut=len(BODY) // 2, ranges=False, full_on=2)
    try:
        path = tmp_path / "model.safetensors"
        dl.download(url, str(path))
        assert path.read_bytes() == BODY
        assert connections[1] is not None   # it did ask for the rest
    finally:
        server.shutdown()


def test_a_download_that_never_finishes_says_how_to_get_the_file_and_keeps_the_part(tmp_path):
    server, url, connections = serve(cut=1 << 20, ranges=False)
    try:
        path = tmp_path / "model.safetensors"
        with pytest.raises(RuntimeError) as failure:
            dl.download(url, str(path))
        message = str(failure.value)
        assert "closed after" in message and url in message and str(tmp_path) in message
        assert "by hand" in message
        assert len(connections) == dl.ATTEMPTS
        assert not path.exists()
        part = tmp_path / "model.safetensors.part"
        assert part.read_bytes() == BODY[:1 << 20]
        assert str(part) in message
    finally:
        server.shutdown()


def test_the_part_a_failed_run_kept_is_continued_by_the_next(tmp_path):
    path = tmp_path / "model.safetensors"
    (tmp_path / "model.safetensors.part").write_bytes(BODY[:1000])
    server, url, connections = serve()
    try:
        dl.download(url, str(path))
        assert path.read_bytes() == BODY
        assert connections == ["bytes=1000-"]
    finally:
        server.shutdown()


def test_a_refused_request_is_not_retried_and_leaves_no_model(tmp_path):
    class Missing(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_error(404)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Missing)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    path = tmp_path / "model.safetensors"
    (tmp_path / "model.safetensors.part").write_bytes(BODY[:1000])   # left by an earlier run
    try:
        with pytest.raises(RuntimeError, match="by hand"):
            dl.download(f"http://127.0.0.1:{server.server_address[1]}/model.safetensors", str(path))
        assert not path.exists()
    finally:
        server.shutdown()
