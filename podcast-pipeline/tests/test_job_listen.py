"""Backend of the conversation player: list, detail, audio with Range, path safety, cache.

A real ThreadingHTTPServer on port 0 calls job_listen.handle. Synthetic export folders
built in tmp_path; WAV files written with the stdlib wave module.

Run:  python -m pytest tests/test_job_listen.py -q -p no:cacheprovider     (from podcast-pipeline/)
"""
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tools import job_listen


def write_wav(path, frames=1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(bytes((i * 37) % 256 for i in range(frames * 2)))
    return path


def write_conv(final, rel, *, topic="", summary="", duration=1.0, turns=2, mtime=None,
               wavs=(), source_stem="src", overlap="good", raw=None):
    d = final / rel
    d.mkdir(parents=True, exist_ok=True)
    jp = d / "conversation.json"
    if raw is not None:
        jp.write_text(raw, encoding="utf-8")
    else:
        meta = {
            "id": rel, "topic": topic, "summary": summary, "duration": duration,
            "overlap_quality": overlap, "source_stem": source_stem,
            "conversation": [{"index": i, "role": "user" if i % 2 else "assistant",
                              "start": float(i), "end": float(i) + 0.5, "text": f"t{i}"}
                             for i in range(turns)],
        }
        jp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    for name in wavs:
        write_wav(d / name)
    if mtime is not None:
        os.utime(jp, (mtime, mtime))
    return d


def build_library(final):
    write_conv(final, "interrupt/a_ca", topic="Xin chào các bạn", summary="Chào hỏi",
               duration=30.0, turns=3, source_stem="src_one", mtime=1000,
               wavs=("stereo_2ch.wav", "stereo_2ch_clean.wav", "assistant.wav", "user.wav"))
    write_conv(final, "interrupt/b_nau", topic="Nấu canh chua", summary="Công thức",
               duration=10.5, turns=2, mtime=3000, wavs=("stereo_2ch.wav",))
    write_conv(final, "backchannel/c_sang", topic="Chào buổi sáng", summary="",
               duration=60.0, turns=5, mtime=2000, wavs=("stereo_2ch.wav",))
    write_conv(final, "turn_by_turn/d_none", topic="Khác", summary="x",
               duration=None, turns=1, mtime=4000)
    write_conv(final, "_quarantine/e_q", topic="Bị cách ly", duration=5.0, turns=2, mtime=500)
    write_conv(final, "interrupt/broken", raw="{not json", mtime=5000)
    write_conv(final, ".hidden/x", topic="ẩn", duration=1.0, mtime=9000)
    write_conv(final, "lhotse/y", topic="lhotse", duration=1.0, mtime=9000)


@pytest.fixture(autouse=True)
def fresh_caches(monkeypatch):
    monkeypatch.setattr(job_listen, "_scan_cache", {})
    monkeypatch.setattr(job_listen, "_meta_cache", {})


@pytest.fixture
def server(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    root = str(final)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parts = urllib.parse.urlsplit(self.path)
            path = parts.path.rstrip("/") or "/"
            if not job_listen.handle(self, path, urllib.parse.parse_qs(parts.query), root):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(base=f"http://127.0.0.1:{srv.server_port}", final=final, root=tmp_path)
    srv.shutdown()
    srv.server_close()


def get(base, url, headers=None):
    req = urllib.request.Request(base + url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read()


def get_json(base, url):
    status, headers, body = get(base, url)
    return status, json.loads(body.decode("utf-8"))


def list_url(**params):
    return "/api/conversations?" + urllib.parse.urlencode(params)


def ids(payload):
    return [item["id"] for item in payload["items"]]


# -- /api/conversations ------------------------------------------------------------------
def test_list_counts_total_and_skips_hidden_lhotse_and_broken(server):
    build_library(server.final)
    status, data = get_json(server.base, list_url())
    assert status == 200
    assert data["total"] == 5
    assert data["counts"] == {"interrupt": 2, "backchannel": 1, "turn_by_turn": 1, "_quarantine": 1}
    assert "interrupt/broken" not in ids(data)
    assert not any(i.startswith((".hidden", "lhotse")) for i in ids(data))


def test_item_fields(server):
    build_library(server.final)
    _, data = get_json(server.base, list_url(sort="name"))
    a = next(i for i in data["items"] if i["id"] == "interrupt/a_ca")
    assert a == {"id": "interrupt/a_ca", "category": "interrupt", "folder": "a_ca",
                 "duration": 30.0, "topic": "Xin chào các bạn", "summary": "Chào hỏi",
                 "turns": 3, "has_clean": True, "overlap_quality": "good",
                 "source_stem": "src_one", "mtime": 1000.0}
    b = next(i for i in data["items"] if i["id"] == "interrupt/b_nau")
    assert b["has_clean"] is False
    d = next(i for i in data["items"] if i["id"] == "turn_by_turn/d_none")
    assert d["duration"] is None


def test_category_filter(server):
    build_library(server.final)
    _, data = get_json(server.base, list_url(category="interrupt"))
    assert data["total"] == 2
    assert set(ids(data)) == {"interrupt/a_ca", "interrupt/b_nau"}
    assert data["counts"] == {"interrupt": 2, "backchannel": 1, "turn_by_turn": 1, "_quarantine": 1}
    _, none = get_json(server.base, list_url(category="nope"))
    assert none["total"] == 0 and none["items"] == []


def test_q_is_accent_and_case_insensitive(server):
    build_library(server.final)
    _, data = get_json(server.base, list_url(q="xin chao"))
    assert ids(data) == ["interrupt/a_ca"]
    _, data = get_json(server.base, list_url(q="NAU CANH"))
    assert ids(data) == ["interrupt/b_nau"]
    _, data = get_json(server.base, list_url(q="chao"))
    assert data["total"] == 2
    assert data["counts"] == {"interrupt": 1, "backchannel": 1}
    _, data = get_json(server.base, list_url(q="src_one"))
    assert ids(data) == ["interrupt/a_ca"]
    _, data = get_json(server.base, list_url(q="chao", category="backchannel"))
    assert ids(data) == ["backchannel/c_sang"]


def test_sorts(server):
    build_library(server.final)
    _, data = get_json(server.base, list_url(sort="new"))
    assert ids(data) == ["turn_by_turn/d_none", "interrupt/b_nau", "backchannel/c_sang",
                         "interrupt/a_ca", "_quarantine/e_q"]
    _, data = get_json(server.base, list_url(sort="long"))
    assert ids(data) == ["backchannel/c_sang", "interrupt/a_ca", "interrupt/b_nau",
                         "_quarantine/e_q", "turn_by_turn/d_none"]
    _, data = get_json(server.base, list_url(sort="short"))
    assert ids(data) == ["_quarantine/e_q", "interrupt/b_nau", "interrupt/a_ca",
                         "backchannel/c_sang", "turn_by_turn/d_none"]
    _, data = get_json(server.base, list_url(sort="name"))
    assert ids(data) == ["interrupt/a_ca", "interrupt/b_nau", "backchannel/c_sang",
                         "turn_by_turn/d_none", "_quarantine/e_q"]
    _, data = get_json(server.base, list_url(sort="bogus"))
    assert ids(data) == ["turn_by_turn/d_none", "interrupt/b_nau", "backchannel/c_sang",
                         "interrupt/a_ca", "_quarantine/e_q"]


def test_pagination_and_limit_bounds(server):
    build_library(server.final)
    _, page1 = get_json(server.base, list_url(sort="new", limit=2, offset=0))
    assert page1["total"] == 5
    assert ids(page1) == ["turn_by_turn/d_none", "interrupt/b_nau"]
    _, page3 = get_json(server.base, list_url(sort="new", limit=2, offset=4))
    assert ids(page3) == ["_quarantine/e_q"]
    _, past = get_json(server.base, list_url(offset=10))
    assert past["items"] == [] and past["total"] == 5
    _, one = get_json(server.base, list_url(limit=0))
    assert len(one["items"]) == 1
    status, big = get_json(server.base, list_url(limit=5000))
    assert status == 200 and len(big["items"]) == 5
    _, junk = get_json(server.base, list_url(limit="abc", offset="-3"))
    assert len(junk["items"]) == 5


def test_empty_or_missing_root_lists_nothing(server):
    status, data = get_json(server.base, list_url())
    assert status == 200
    assert data == {"total": 0, "counts": {}, "items": []}


def test_broken_json_and_unreadable_files_do_not_crash_listing(server, monkeypatch):
    build_library(server.final)

    def boom(path):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(job_listen, "_read_json", boom)
    status, data = get_json(server.base, list_url())
    assert status == 200 and data["total"] == 0 and data["items"] == []


# -- /api/conversation -------------------------------------------------------------------
def test_detail_lists_only_existing_audio_urls(server):
    build_library(server.final)
    status, data = get_json(server.base, "/api/conversation?" + urllib.parse.urlencode({"id": "interrupt/a_ca"}))
    assert status == 200
    assert data["id"] == "interrupt/a_ca"
    assert data["meta"]["topic"] == "Xin chào các bạn"
    assert data["meta"]["conversation"][1]["text"] == "t1"
    assert set(data["files"]) == {"stereo", "stereo_clean", "assistant", "user"}
    assert data["files"]["stereo"] == "/audio/interrupt/a_ca/stereo_2ch.wav"
    _, headers, body = get(server.base, data["files"]["stereo_clean"])
    assert body == (server.final / "interrupt/a_ca/stereo_2ch_clean.wav").read_bytes()


def test_detail_unknown_or_invalid_id_is_404_json(server):
    build_library(server.final)
    for bad in ("", "interrupt/missing", "../final", "interrupt/../interrupt/a_ca",
                ".hidden/x", "lhotse/y", "/etc", "interrupt", "interrupt/a_ca/x/y/z"):
        status, data = get_json(server.base, "/api/conversation?" + urllib.parse.urlencode({"id": bad}))
        assert status == 404, bad
        assert "error" in data


def test_detail_of_broken_json_gives_null_meta(server):
    build_library(server.final)
    status, data = get_json(server.base, "/api/conversation?" + urllib.parse.urlencode({"id": "interrupt/broken"}))
    assert status == 200 and data["meta"] is None and data["files"] == {}


def test_special_characters_in_folder_name(server):
    rel = "interrupt/ghi #1 100% (ảnh) ‪ x‬‏"
    write_conv(server.final, rel, topic="Ký tự đặc biệt", duration=2.0, mtime=1500,
               wavs=("stereo_2ch.wav",))
    _, data = get_json(server.base, list_url(q="ky tu dac biet"))
    assert ids(data) == [rel]
    _, detail = get_json(server.base, "/api/conversation?" + urllib.parse.urlencode({"id": rel}))
    assert detail["meta"]["topic"] == "Ký tự đặc biệt"
    url = detail["files"]["stereo"]
    assert url.startswith("/audio/") and "#" not in url and " " not in url
    status, headers, body = get(server.base, url, {"Range": "bytes=10-"})
    assert status == 206
    assert body == (server.final / rel / "stereo_2ch.wav").read_bytes()[10:]


# -- /audio ------------------------------------------------------------------------------
@pytest.fixture
def audio(server):
    write_conv(server.final, "interrupt/a_ca", topic="t", mtime=1000, wavs=("stereo_2ch.wav",))
    data = (server.final / "interrupt/a_ca/stereo_2ch.wav").read_bytes()
    return SimpleNamespace(url="/audio/interrupt/a_ca/stereo_2ch.wav", data=data, size=len(data))


def test_audio_full_without_range(server, audio):
    status, headers, body = get(server.base, audio.url)
    assert status == 200
    assert body == audio.data
    assert headers["Content-Length"] == str(audio.size)
    assert headers["Accept-Ranges"] == "bytes"
    assert headers["Content-Type"] == "audio/wav"
    assert headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("spec, start, end", [
    ("bytes=0-99", 0, 99),
    ("bytes=100-199", 100, 199),
    ("bytes=500-", 500, None),
    ("bytes=-50", None, 50),
    ("bytes=-100000", None, 100000),
    ("bytes=0-99999999", 0, 99999999),
    ("bytes=5-5", 5, 5),
])
def test_audio_range_forms(server, audio, spec, start, end):
    status, headers, body = get(server.base, audio.url, {"Range": spec})
    size = audio.size
    if start is None:
        first = max(0, size - end)
        last = size - 1
    else:
        first = start
        last = size - 1 if end is None else min(end, size - 1)
    assert status == 206
    assert body == audio.data[first:last + 1]
    assert headers["Content-Length"] == str(last - first + 1)
    assert headers["Content-Range"] == f"bytes {first}-{last}/{size}"
    assert headers["Accept-Ranges"] == "bytes"


@pytest.mark.parametrize("spec", [
    "bytes=2044-", "bytes=99999-", "bytes=abc", "bytes=5-3", "bytes=-0", "bytes=-",
    "bytes=0-1,3-4", "bytes=", "items=0-9",
])
def test_audio_bad_range_is_416(server, audio, spec):
    status, headers, body = get(server.base, audio.url, {"Range": spec})
    assert status == 416
    assert headers["Content-Range"] == f"bytes */{audio.size}"
    assert body == b""


def test_audio_rejects_traversal_absolute_nul_and_non_wav(server, audio):
    secret = server.root / "secret.wav"
    write_wav(secret)
    bad = [
        "/audio/interrupt/a_ca/../../../secret.wav",
        "/audio/interrupt/a_ca/%2E%2E/%2E%2E/%2E%2E/secret.wav",
        "/audio/interrupt/a_ca/..%2F..%2F..%2Fsecret.wav",
        "/audio/" + urllib.parse.quote(str(secret)),
        "/audio/interrupt/a_ca/stereo_2ch.wav%00",
        "/audio/interrupt/a_ca%00/stereo_2ch.wav",
        "/audio/interrupt/a_ca/conversation.json",
        "/audio/interrupt/a_ca/stereo_2ch.txt",
        "/audio/interrupt/a_ca/missing.wav",
        "/audio/stereo_2ch.wav",
        "/audio/",
    ]
    for url in bad:
        status, _, _ = get(server.base, url)
        assert status == 404, url


def test_audio_symlinks_leaving_the_root_are_404(server, audio):
    secret = write_wav(server.root / "secret.wav")
    outside = server.root / "outside"
    write_wav(outside / "x.wav")
    os.symlink(secret, server.final / "interrupt/a_ca/link.wav")
    os.symlink(outside, server.final / "linkdir")
    assert get(server.base, "/audio/interrupt/a_ca/link.wav")[0] == 404
    assert get(server.base, "/audio/linkdir/x.wav")[0] == 404


def test_audio_special_characters_in_path(server):
    rel = "turn_by_turn/50% #2 ệ ‪‬‏"
    d = write_conv(server.final, rel, topic="x", mtime=1000, wavs=("user.wav",))
    url = "/audio/" + urllib.parse.quote(f"{rel}/user.wav", safe="/")
    status, headers, body = get(server.base, url, {"Range": "bytes=0-43"})
    assert status == 206
    assert body == (d / "user.wav").read_bytes()[:44]


def test_handle_returns_false_for_foreign_paths(tmp_path):
    assert job_listen.handle(object(), "/nope", {}, str(tmp_path)) is False
    assert job_listen.handle(object(), "/", {}, str(tmp_path)) is False


# -- /listen -----------------------------------------------------------------------------
def test_listen_page_missing_is_404_with_message(server, monkeypatch, tmp_path):
    monkeypatch.setattr(job_listen, "PAGE_PATH", str(tmp_path / "nope.html"))
    status, headers, body = get(server.base, "/listen")
    assert status == 404
    assert b"nope.html" in body


def test_listen_page_served_as_html(server, monkeypatch, tmp_path):
    page = tmp_path / "listen_page.html"
    page.write_text("<!doctype html><title>x</title>héllo", encoding="utf-8")
    monkeypatch.setattr(job_listen, "PAGE_PATH", str(page))
    status, headers, body = get(server.base, "/listen")
    assert status == 200
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert headers["Cache-Control"] == "no-store"
    assert body == page.read_bytes()


# -- cache -------------------------------------------------------------------------------
def test_meta_is_reread_only_when_the_file_changes(server, monkeypatch):
    build_library(server.final)
    monkeypatch.setattr(job_listen, "SCAN_TTL", 0)
    reads = []
    real = job_listen._read_json

    def counting(path):
        reads.append(path)
        return real(path)

    monkeypatch.setattr(job_listen, "_read_json", counting)
    get_json(server.base, list_url())
    assert len(reads) == 6          # five readable plus interrupt/broken
    get_json(server.base, list_url())
    get_json(server.base, list_url(q="chao"))
    assert len(reads) == 6          # unchanged files are not opened again

    jp = server.final / "interrupt/a_ca/conversation.json"
    meta = json.loads(jp.read_text(encoding="utf-8"))
    meta["topic"] = "Đổi rồi"
    jp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    os.utime(jp, (7000, 7000))
    _, data = get_json(server.base, list_url())
    assert len(reads) == 7
    assert reads[-1] == str(jp)
    assert next(i for i in data["items"] if i["id"] == "interrupt/a_ca")["topic"] == "Đổi rồi"


def test_scan_is_reused_within_ttl(server, monkeypatch):
    build_library(server.final)
    walks = []
    real = job_listen._walk

    def counting(final):
        walks.append(final)
        return real(final)

    monkeypatch.setattr(job_listen, "_walk", counting)
    monkeypatch.setattr(job_listen, "SCAN_TTL", 60)
    get_json(server.base, list_url())
    get_json(server.base, list_url(sort="long"))
    assert len(walks) == 1
    monkeypatch.setattr(job_listen, "SCAN_TTL", 0)
    get_json(server.base, list_url())
    assert len(walks) == 2


def test_fold_drops_tone_marks_and_d_stroke():
    assert job_listen._fold("Xin Chào") == "xin chao"
    assert job_listen._fold("Đường Đi") == "duong di"
    assert job_listen._fold("  nấu   canh ") == "nau canh"
