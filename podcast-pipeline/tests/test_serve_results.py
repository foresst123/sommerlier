import http.client
import importlib.util
import threading
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("serve_results", Path(__file__).resolve().parents[1] / "tools" / "serve_results.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)

DATA = bytes(range(256)) * 4000            # 1,024,000 bytes: more than one 1 MiB chunk is not needed, ranges are


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "out"
    (root / "lanes").mkdir(parents=True)
    (root / "lanes" / "a_lanes.html").write_text("<p>hi</p>", encoding="utf-8")
    (root / "none").mkdir()
    (root / "none" / "x.wav").write_bytes(DATA)
    (root / "empty.wav").write_bytes(b"")
    (root / "ten.wav").write_bytes(b"0123456789")
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "orig.wav").write_bytes(b"ORIGINAL-AUDIO")
    (tmp_path / "secret.txt").write_text("secret")
    (root / "link.wav").symlink_to(audio / "orig.wav")      # like _resident/input_*: a link into the tree
    srv = sr.make_server(root, audio)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def get(port, path, method="GET", headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, headers=headers or {})
    r = c.getresponse()
    body = r.read()
    out = (r.status, dict((k.lower(), v) for k, v in r.getheaders()), body)
    c.close()
    return out


def test_whole_file_has_accept_ranges_and_audio_type(server):
    status, h, body = get(server, "/none/x.wav")
    assert status == 200 and body == DATA
    assert h["accept-ranges"] == "bytes" and h["content-type"] == "audio/wav" and h["content-length"] == str(len(DATA))


def test_range_returns_only_that_part(server):
    status, h, body = get(server, "/none/x.wav", headers={"Range": "bytes=1000-1999"})
    assert status == 206 and body == DATA[1000:2000]
    assert h["content-range"] == f"bytes 1000-1999/{len(DATA)}" and h["content-length"] == "1000"
    status, h, body = get(server, "/ten.wav", headers={"Range": "bytes=7-"})
    assert status == 206 and body == b"789" and h["content-range"] == "bytes 7-9/10"
    status, h, body = get(server, "/ten.wav", headers={"Range": "bytes=-4"})
    assert status == 206 and body == b"6789"
    status, h, body = get(server, "/ten.wav", headers={"Range": "bytes=5-500"})        # end past the file is clamped
    assert status == 206 and body == b"56789" and h["content-range"] == "bytes 5-9/10"


def test_unsatisfiable_range_is_416_and_odd_headers_are_ignored(server):
    status, h, _ = get(server, "/ten.wav", headers={"Range": "bytes=10-20"})
    assert status == 416 and h["content-range"] == "bytes */10"
    assert get(server, "/ten.wav", headers={"Range": "bytes=5-2"})[0] == 416
    assert get(server, "/ten.wav", headers={"Range": "bytes=-0"})[0] == 416
    for odd in ("bytes=0-1,4-5", "items=0-3", "bytes=abc", "bytes=-"):
        status, _, body = get(server, "/ten.wav", headers={"Range": odd})
        assert status == 200 and body == b"0123456789", odd                              # whole file, as RFC 9110 allows


def test_empty_file_and_head(server):
    assert get(server, "/empty.wav")[::2] == (200, b"")
    status, h, body = get(server, "/ten.wav", method="HEAD")
    assert status == 200 and body == b"" and h["content-length"] == "10"
    status, h, body = get(server, "/ten.wav", method="HEAD", headers={"Range": "bytes=2-3"})
    assert status == 206 and body == b"" and h["content-length"] == "2"


def test_directory_listing_and_redirect(server):
    status, h, body = get(server, "/lanes")
    assert status == 301 and h["location"] == "/lanes/"
    status, h, body = get(server, "/lanes/")
    assert status == 200 and b"a_lanes.html" in body and h["content-type"].startswith("text/html")
    status, h, body = get(server, "/lanes/a_lanes.html")
    assert body == b"<p>hi</p>" and "utf-8" in h["content-type"]


def test_path_traversal_is_refused(server):
    for bad in ("/../secret.txt", "/none/../../secret.txt", "/%2e%2e/secret.txt", "/none/%2e%2e/%2e%2e/secret.txt",
                "/_audio/../secret.txt", "/_audio/%2e%2e/secret.txt", "/%00", "/none/x.wav%00.txt"):
        assert get(server, bad)[0] == 404, bad
    assert get(server, "/nothing.wav")[0] == 404


def test_audio_dir_is_mounted_and_symlinks_inside_the_tree_work(server):
    assert get(server, "/_audio/orig.wav")[::2] == (200, b"ORIGINAL-AUDIO")
    assert get(server, "/_audio/nope.wav")[0] == 404
    assert get(server, "/link.wav")[::2] == (200, b"ORIGINAL-AUDIO")           # link followed


def test_no_audio_dir_means_no_mount(tmp_path):
    srv = sr.make_server(tmp_path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert get(srv.server_address[1], "/_audio/x.wav")[0] == 404
    finally:
        srv.shutdown()
        srv.server_close()


def test_keep_alive_serves_several_range_requests_on_one_connection(server):
    c = http.client.HTTPConnection("127.0.0.1", server, timeout=10)
    for a, b in ((0, 99), (500, 599), (1000, 1099)):
        c.request("GET", "/none/x.wav", headers={"Range": f"bytes={a}-{b}"})
        r = c.getresponse()
        assert r.status == 206 and r.read() == DATA[a:b + 1]
    c.close()


def test_big_range_crosses_chunk_boundaries(tmp_path):
    root = tmp_path / "o"
    root.mkdir()
    big = (bytes(range(256)) * 20000)[:3 * sr.CHUNK + 12345]
    (root / "big.wav").write_bytes(big)
    srv = sr.make_server(root)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        status, _, body = get(srv.server_address[1], "/big.wav", headers={"Range": f"bytes={sr.CHUNK - 5}-{2 * sr.CHUNK + 7}"})
        assert status == 206 and body == big[sr.CHUNK - 5:2 * sr.CHUNK + 8]
    finally:
        srv.shutdown()
        srv.server_close()
