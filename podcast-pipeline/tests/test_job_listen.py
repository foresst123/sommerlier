"""Backend of the conversation player: list, detail, audio with Range, path safety, cache.

A real ThreadingHTTPServer on port 0 calls job_listen.handle. Synthetic export folders
built in tmp_path; WAV files written with the stdlib wave module.

Run:  python -m pytest tests/test_job_listen.py -q -p no:cacheprovider     (from podcast-pipeline/)
"""
import csv
import io
import json
import os
import re
import shutil
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
    monkeypatch.setattr(job_listen, "_audio_scan", {})
    monkeypatch.setattr(job_listen, "_dur_cache", {})


@pytest.fixture
def server(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    root = str(final)

    ctx = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parts = urllib.parse.urlsplit(self.path)
            path = parts.path.rstrip("/") or "/"
            if not job_listen.handle(self, path, urllib.parse.parse_qs(parts.query), root, ctx):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def do_POST(self):
            parts = urllib.parse.urlsplit(self.path)
            path = parts.path.rstrip("/") or "/"
            if not job_listen.handle_post(self, path, root, ctx):
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(base=f"http://127.0.0.1:{srv.server_port}", final=final, root=tmp_path,
                          ctx=ctx)
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


def post(base, url, body):
    data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base + url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "null")


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


def test_handle_ctx_is_optional(server):
    build_library(server.final)
    assert job_listen.handle(object(), "/nope", {}, str(server.final), {"audio_dir": None}) is False
    status, data = get_json(server.base, list_url())
    assert status == 200 and data["total"] == 5


# -- three films for sample, review and stats --------------------------------------------
FILMS = {
    "dQw4w9WgXcQ_Camera Công Sở Tập 12": [("interrupt", "a1", 100), ("backchannel", "a2", 50),
                                          ("turn_by_turn", "a3", 20)],
    "dQw4w9WgXcQ_Camera Công Sở - Tập 13": [("interrupt", "b1", 100), ("backchannel", "b2", 50)],
    "‎Cooking Show | Tập 1": [("interrupt", "c1", 30), ("backchannel", "c2", 40)],
    "cooking show #2": [("turn_by_turn", "c3", 10)],
    "Phim Việt Nam: Tập 5": [("interrupt", "d1", 60), ("turn_by_turn", "d2", 70)],
}


def build_films(final):
    mtime = 1000
    for source, convs in FILMS.items():
        for cat, name, dur in convs:
            mtime += 1
            write_conv(final, f"{cat}/{name}", topic=f"{source[-6:]} {name}", duration=float(dur),
                       source_stem=source, mtime=mtime)
    write_conv(final, "_quarantine/q1", topic="q", duration=5.0, source_stem="Cooking Show | Tập 1")


def sample(base, **params):
    status, data = get_json(base, "/api/sample?" + urllib.parse.urlencode(params))
    assert status == 200
    return data


@pytest.mark.parametrize("stem, film", [
    ("dQw4w9WgXcQ_Camera Công Sở Tập 12", "Camera Công Sở"),
    ("dQw4w9WgXcQ_Camera Công Sở - Tập 13", "Camera Công Sở"),
    ("‎‪Cooking Show | Tập 1‬", "Cooking Show"),
    ("cooking show #2", "cooking show"),
    ("Phim Việt Nam: Tập 5", "Phim Việt Nam"),
    ("Abc｜x", "Abc"), ("Abc ： x", "Abc"), ("Phim — tập 2", "Phim"),
    ("Show Ep 3", "Show"), ("Show EP 3", "Show"), ("Show Episode 4", "Show"), ("Show Tap 5", "Show"),
    ("plain title", "plain title"),
    ("", job_listen.FILM_UNKNOWN), (None, job_listen.FILM_UNKNOWN), (": x", job_listen.FILM_UNKNOWN),
])
def test_film_name_from_source_stem(stem, film):
    assert job_listen._film_name(stem) == film


def test_sample_covers_every_film_when_n_allows(server):
    build_films(server.final)
    data = sample(server.base, n=3, seed=5)
    assert data["n"] == 3 and data["films"] == 3
    assert {i["film"] for i in data["items"]} == {"Camera Công Sở", "Cooking Show", "Phim Việt Nam"}


def test_sample_takes_every_conversation_when_n_is_large_and_skips_quarantine(server):
    build_films(server.final)
    data = sample(server.base, n=500)
    assert data["n"] == 10 and data["films"] == 3
    assert not any(i["category"] == "_quarantine" for i in data["items"])
    assert sum(i["film"] == "Camera Công Sở" for i in data["items"]) == 5
    assert sum(i["film"] == "Cooking Show" for i in data["items"]) == 3


def test_sample_is_deterministic_per_seed_and_varies_between_seeds(server):
    build_films(server.final)
    first = sample(server.base, n=10, seed=7)
    again = sample(server.base, n=10, seed=7)
    assert ids(first) == ids(again)
    assert ids(sample(server.base, n=10, seed=1)) != ids(sample(server.base, n=10, seed=2))


def test_sample_round_robin_gives_each_film_its_share(server):
    build_films(server.final)
    for seed in range(4):
        data = sample(server.base, n=6, seed=seed)
        counts = {}
        for item in data["items"]:
            counts[item["film"]] = counts.get(item["film"], 0) + 1
        assert counts == {"Camera Công Sở": 2, "Cooking Show": 2, "Phim Việt Nam": 2}


def test_sample_prefers_unused_recordings_within_a_film(server):
    write_conv(server.final, "interrupt/s_a1", topic="x", duration=1.0,
               source_stem="Solo Show Tập 1", mtime=1)
    write_conv(server.final, "interrupt/s_a2", topic="x", duration=1.0,
               source_stem="Solo Show Tập 1", mtime=2)
    write_conv(server.final, "interrupt/s_b1", topic="x", duration=1.0,
               source_stem="Solo Show Tập 2", mtime=3)
    for seed in range(3):
        data = sample(server.base, n=2, seed=seed)
        assert len({i["source_stem"] for i in data["items"]}) == 2


def test_sample_rotates_categories(server):
    for rec in ("Rot Show Tập 1", "Rot Show Tập 2", "Rot Show Tập 3"):
        for cat in ("interrupt", "backchannel", "turn_by_turn"):
            write_conv(server.final, f"{cat}/{rec.replace(' ', '_')}_{cat}", topic="x", duration=1.0,
                       source_stem=rec, mtime=1)
    data = sample(server.base, n=3, seed=3)
    assert {i["category"] for i in data["items"]} == {"interrupt", "backchannel", "turn_by_turn"}
    assert len({i["source_stem"] for i in data["items"]}) == 3


def test_sample_category_filter_and_labeled_flag(server):
    build_films(server.final)
    data = sample(server.base, n=500, category="backchannel")
    assert data["n"] == 3 and all(i["category"] == "backchannel" for i in data["items"])
    assert all(i["labeled"] is None for i in data["items"])
    post(server.base, "/api/review", {"id": "backchannel/a2", "verdict": "bad", "tags": ["x"]})
    data = sample(server.base, n=500, category="backchannel")
    labeled = {i["id"]: i["labeled"] for i in data["items"]}
    assert labeled == {"backchannel/a2": "bad", "backchannel/b2": None, "backchannel/c2": None}


def test_sample_clamps_n_and_handles_empty_library(server):
    build_films(server.final)
    assert sample(server.base, n=0)["n"] == 1
    assert sample(server.base, n=99999)["n"] == 10
    empty = sample(server.base, n=5, category="nope")
    assert empty == {"n": 0, "films": 0, "seed": 0, "items": []}


# -- /api/review -------------------------------------------------------------------------
def test_review_empty_summary(server):
    status, data = get_json(server.base, "/api/review")
    assert status == 200
    assert data == {"set": "default", "labels": {}, "summary": {"total": 0, "ok": 0, "bad": 0,
                                                                 "bad_rate": None, "by_tag": {}}}


def test_post_sets_and_clears_a_label(server):
    build_library(server.final)
    status, data = post(server.base, "/api/review", {
        "id": "interrupt/a_ca", "verdict": "bad", "tags": ["ngắt lời", "ngắt lời", "  nhiễu "],
        "note": "lệch pha"})
    assert status == 200
    label = data["labels"]["interrupt/a_ca"]
    assert label["verdict"] == "bad" and label["tags"] == ["ngắt lời", "nhiễu"]
    assert label["note"] == "lệch pha" and isinstance(label["time"], (int, float))
    _, stored = get_json(server.base, "/api/review")
    assert stored["labels"] == data["labels"]
    assert (server.root / "review_sets" / "default.json").is_file()
    assert not (server.final / "_review").exists()

    status, cleared = post(server.base, "/api/review", {"id": "interrupt/a_ca", "verdict": None})
    assert status == 200 and cleared["labels"] == {}
    assert cleared["summary"]["total"] == 0


def test_tags_and_note_are_cut(server):
    build_library(server.final)
    tags = [f"t{i}-" + "x" * 50 for i in range(10)]
    status, data = post(server.base, "/api/review", {
        "id": "backchannel/c_sang", "verdict": "ok", "tags": tags, "note": "n" * 600})
    assert status == 200
    label = data["labels"]["backchannel/c_sang"]
    assert len(label["tags"]) == 8
    assert all(len(t) <= 40 for t in label["tags"])
    assert label["tags"][0] == tags[0][:40]
    assert len(label["note"]) == 500


def test_unknown_or_invalid_ids_are_404_and_write_nothing(server):
    build_library(server.final)
    for bad in ("interrupt/missing", "../final", "_review/labels", ".hidden/x", "lhotse/y",
                "", "interrupt", "/etc", "interrupt/../interrupt/a_ca"):
        status, data = post(server.base, "/api/review", {"id": bad, "verdict": "ok"})
        assert status == 404, bad
        assert "error" in data
    assert not (server.root / "review_sets" / "default.json").exists()


def test_bad_bodies_are_400_and_too_large_is_413(server):
    build_library(server.final)
    cases = [
        b"{not json",
        b"[]",
        b"",
        {"id": 5, "verdict": "ok"},
        {"id": "interrupt/a_ca", "verdict": "maybe"},
        {"id": "interrupt/a_ca", "verdict": "ok", "tags": "x"},
        {"id": "interrupt/a_ca", "verdict": "ok", "tags": [1]},
        {"id": "interrupt/a_ca", "verdict": "ok", "note": ["x"]},
    ]
    for body in cases:
        status, data = post(server.base, "/api/review", body)
        assert status == 400, body
        assert "error" in data
    status, data = post(server.base, "/api/review",
                        {"id": "interrupt/a_ca", "verdict": "ok", "note": "x" * 70000})
    assert status == 413
    assert "error" in data
    assert not (server.root / "review_sets" / "default.json").exists()


def test_summary_bad_rate_and_by_tag(server):
    build_library(server.final)
    post(server.base, "/api/review", {"id": "interrupt/a_ca", "verdict": "bad", "tags": ["x", "y"]})
    post(server.base, "/api/review", {"id": "interrupt/b_nau", "verdict": "bad", "tags": ["x"]})
    status, data = post(server.base, "/api/review", {"id": "backchannel/c_sang", "verdict": "ok",
                                                     "tags": ["z"]})
    summary = data["summary"]
    assert summary["total"] == 3 and summary["ok"] == 1 and summary["bad"] == 2
    assert summary["bad_rate"] == pytest.approx(2 / 3)
    assert summary["by_tag"] == {"x": 2, "y": 1}


def test_damaged_set_file_is_kept_not_deleted(server):
    build_library(server.final)
    rdir = server.root / "review_sets"
    rdir.mkdir()
    (rdir / "default.json").write_text("{broken", encoding="utf-8")
    status, data = get_json(server.base, "/api/review")
    assert status == 200 and data["summary"]["total"] == 0
    assert (rdir / "default.json").read_text(encoding="utf-8") == "{broken"

    status, data = post(server.base, "/api/review", {"id": "interrupt/a_ca", "verdict": "ok"})
    assert status == 200 and set(data["labels"]) == {"interrupt/a_ca"}
    kept = list(rdir.glob("default.json.bad-*"))
    assert len(kept) == 1 and kept[0].read_text(encoding="utf-8") == "{broken"
    stored = json.loads((rdir / "default.json").read_text(encoding="utf-8"))
    assert set(stored["labels"]) == {"interrupt/a_ca"}


def test_review_folder_is_never_a_category(server):
    build_library(server.final)
    write_conv(server.final, "_review/fake", topic="fake", duration=1.0, mtime=1)
    post(server.base, "/api/review", {"id": "interrupt/a_ca", "verdict": "ok"})
    _, data = get_json(server.base, list_url())
    assert data["total"] == 5
    assert "_review" not in data["counts"]
    assert not any(i.startswith("_review") for i in ids(data))
    assert get_json(server.base, "/api/conversation?" + urllib.parse.urlencode({"id": "_review/fake"}))[0] == 404
    sampled = sample(server.base, n=500)
    assert not any(i["id"].startswith("_review") for i in sampled["items"])


# -- /api/stats --------------------------------------------------------------------------
def build_stats(final, audio):
    write_conv(final, "interrupt/s1", topic="x", duration=1800.0,
               source_stem="dQw4w9WgXcQ_Camera Công Sở Tập 12", mtime=1000)
    write_conv(final, "interrupt/s2", topic="x", duration=1800.0,
               source_stem="dQw4w9WgXcQ_Camera Công Sở Tập 12", mtime=1001)
    write_conv(final, "backchannel/s3", topic="x", duration=900.0,
               source_stem="Cooking Show | Tập 1", mtime=1002)
    write_conv(final, "turn_by_turn/s4", topic="x", duration=50.0,
               source_stem="Phim Việt Nam: Tập 5", mtime=1003)
    write_conv(final, "_quarantine/s5", topic="x", duration=999.0,
               source_stem="Phim Việt Nam: Tập 5", mtime=1004)
    audio.mkdir(parents=True, exist_ok=True)
    write_wav(audio / "dQw4w9WgXcQ_Camera Công Sở Tập 12.wav", frames=16000)   # 2 s
    write_wav(audio / "Cooking_Show___T_p_1.wav", frames=24000)                # 3 s, sanitized name
    write_wav(audio / "unmatched.wav", frames=40000)                           # 5 s
    (audio / "notes.txt").write_text("not audio", encoding="utf-8")


def test_stats_without_audio_dir_gives_nulls_for_input_hours(server):
    build_stats(server.final, server.root / "audio")
    _, data = get_json(server.base, "/api/stats")
    assert data["input"] == {"recordings_total": None, "recordings_with_conversations": 3,
                             "hours_with_conversations": None, "hours_total": None}
    assert data["yield"] == {"hours_ratio": None,
                             "conversations_per_recording": pytest.approx(4 / 3)}
    assert data["output"]["conversations"] == 4 and data["output"]["films"] == 3


def test_stats_numbers_with_audio_dir(server):
    audio = server.root / "audio"
    build_stats(server.final, audio)
    server.ctx.update({"audio_dir": str(audio), "total_recordings": 4})
    _, data = get_json(server.base, "/api/stats")
    assert data["input"]["recordings_total"] == 4
    assert data["input"]["recordings_with_conversations"] == 3
    assert data["input"]["hours_with_conversations"] == pytest.approx(5 / 3600)
    assert data["input"]["hours_total"] == pytest.approx(10 / 3600)
    out = data["output"]
    assert out["conversations"] == 4 and out["films"] == 3
    assert out["hours"] == pytest.approx(4550 / 3600)
    assert out["by_category"]["interrupt"]["n"] == 2
    assert out["by_category"]["interrupt"]["hours"] == pytest.approx(1.0)
    assert out["by_category"]["backchannel"]["hours"] == pytest.approx(900 / 3600)
    assert out["by_category"]["turn_by_turn"]["n"] == 1
    assert data["yield"]["hours_ratio"] == pytest.approx(455.0)
    assert data["yield"]["conversations_per_recording"] == pytest.approx(4 / 3)
    films = data["films"]
    assert [f["film"] for f in films] == ["Camera Công Sở", "Cooking Show", "Phim Việt Nam"]
    assert films[0]["recordings"] == 1 and films[0]["conversations"] == 2
    assert films[0]["hours"] == pytest.approx(1.0)


def test_stats_missing_audio_dir_is_null_not_an_error(server):
    build_stats(server.final, server.root / "audio")
    server.ctx.update({"audio_dir": str(server.root / "missing")})
    _, data = get_json(server.base, "/api/stats")
    assert data["input"]["hours_total"] is None
    assert data["input"]["hours_with_conversations"] is None
    assert data["yield"]["hours_ratio"] is None


def test_stats_reads_each_audio_file_once_within_the_cache_window(server, monkeypatch):
    audio = server.root / "audio"
    build_stats(server.final, audio)
    reads = []
    real = job_listen._read_seconds

    def counting(path):
        reads.append(path)
        return real(path)

    monkeypatch.setattr(job_listen, "_read_seconds", counting)
    index = job_listen._audio_index(str(audio))
    assert index is not None and index["hours"] == pytest.approx(10 / 3600)
    assert len(reads) == 3
    job_listen._audio_index(str(audio))
    assert len(reads) == 3


def test_the_page_frame_loop_never_schedules_two_frames_per_frame():
    """Regression: frame() panned with panTo(), which asked for another frame, and then scheduled its own. Past 92% of a
    fitted timeline that happened on every frame, so the frames doubled each time and the page froze near the end."""
    import re
    page = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "listen_page.html"), encoding="utf-8").read()
    assert re.search(r"function panTo\(t, quiet\)", page)
    frame = page[page.index("function frame()"):page.index("function kick()")]
    assert "panTo(now - 0.15 * W / pps, true)" in frame          # the pan inside the loop is quiet
    assert "!fitMode" in frame                                   # and does nothing while the whole file fits
    assert "!raf) raf = requestAnimationFrame(frame)" in frame   # at most one pending frame


# -- review sets -------------------------------------------------------------------------
def sample_url(**params):
    return "/api/sample?" + urllib.parse.urlencode(params)


def set_names(base):
    return [s["name"] for s in get_json(base, "/api/review_sets")[1]["sets"]]


def rescan():
    job_listen._scan_cache.clear()
    job_listen._meta_cache.clear()


def test_labels_survive_clearing_the_export_folder(server):
    build_library(server.final)
    post(server.base, "/api/review", {"id": "interrupt/a_ca", "verdict": "bad", "tags": ["x"]})
    shutil.rmtree(server.final)
    server.final.mkdir()
    build_library(server.final)
    rescan()
    status, data = get_json(server.base, "/api/review?set=default")
    assert status == 200 and data["labels"]["interrupt/a_ca"]["verdict"] == "bad"
    assert (server.root / "review_sets" / "default.json").is_file()


def test_create_list_and_delete_sets(server):
    build_library(server.final)
    status, data = get_json(server.base, "/api/review_sets")
    assert status == 200
    assert data["dir"] == str(server.root / "review_sets")
    assert [s["name"] for s in data["sets"]] == ["default"]
    assert data["sets"][0]["labeled"] == 0 and data["sets"][0]["n_sample"] == 0

    status, row = post(server.base, "/api/review_sets", {"name": "bo_1"})
    assert status == 201 and row["name"] == "bo_1" and row["labeled"] == 0
    assert row["created"] and row["updated"]
    assert post(server.base, "/api/review_sets", {"name": "bo_1"})[0] == 409
    assert post(server.base, "/api/review_sets", {"name": "default"})[0] == 409
    assert (server.root / "review_sets" / "bo_1.json").is_file()

    post(server.base, "/api/review", {"id": "interrupt/a_ca", "verdict": "ok", "set": "bo_1"})
    status, data = post(server.base, "/api/review_sets", {"name": "bo_1", "delete": True})
    assert status == 409 and data["labeled"] == 1
    status, data = post(server.base, "/api/review_sets",
                        {"name": "bo_1", "delete": True, "confirm": True})
    assert status == 200 and data == {"deleted": "bo_1"}
    assert not (server.root / "review_sets" / "bo_1.json").exists()
    assert post(server.base, "/api/review_sets", {"name": "bo_1", "delete": True})[0] == 404
    assert post(server.base, "/api/review_sets", {"name": "default", "delete": True,
                                                  "confirm": True})[0] == 400
    assert set_names(server.base) == ["default"]


def test_sets_are_listed_newest_first_with_counts(server):
    build_library(server.final)
    post(server.base, "/api/review_sets", {"name": "older"})
    post(server.base, "/api/review_sets", {"name": "newer"})
    path = server.root / "review_sets" / "older.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["updated"] = "2001-01-01T00:00:00+00:00"
    path.write_text(json.dumps(doc), encoding="utf-8")
    post(server.base, "/api/review", {"id": "interrupt/a_ca", "verdict": "bad", "set": "newer"})
    post(server.base, "/api/review", {"id": "interrupt/b_nau", "verdict": "ok", "set": "newer"})
    sets = get_json(server.base, "/api/review_sets")[1]["sets"]
    assert [s["name"] for s in sets] == ["newer", "older", "default"]
    newer = sets[0]
    assert (newer["labeled"], newer["ok"], newer["bad"]) == (2, 1, 1)
    assert newer["bad_rate"] == pytest.approx(0.5)


def test_invalid_set_names_are_400(server):
    build_library(server.final)
    for bad in ("", "   ", "x" * 41, None, 5):
        status, data = post(server.base, "/api/review_sets", {"name": bad})
        assert status == 400 and "error" in data, bad
    assert post(server.base, "/api/review", {"id": "interrupt/a_ca", "verdict": "ok",
                                             "set": ""})[0] == 400
    assert get_json(server.base, "/api/review?set=" + urllib.parse.quote("  "))[0] == 400


def test_vietnamese_names_lose_their_marks(server):
    status, row = post(server.base, "/api/review_sets", {"name": "Bộ chấm đầu!"})
    assert status == 201 and row["name"] == "Bo_cham_dau_"
    assert (server.root / "review_sets" / "Bo_cham_dau_.json").is_file()
    assert post(server.base, "/api/review_sets", {"name": "x" * 40})[0] == 201


def test_unknown_set_is_404_for_labels_and_reads(server):
    build_library(server.final)
    assert get_json(server.base, "/api/review?set=nope")[0] == 404
    status, data = post(server.base, "/api/review", {"id": "interrupt/a_ca", "verdict": "ok",
                                                     "set": "nope"})
    assert status == 404 and "error" in data
    assert not (server.root / "review_sets" / "nope.json").exists()


def test_labels_are_separate_between_sets(server):
    build_library(server.final)
    post(server.base, "/api/review_sets", {"name": "a"})
    post(server.base, "/api/review_sets", {"name": "b"})
    post(server.base, "/api/review", {"id": "interrupt/a_ca", "verdict": "ok", "set": "a"})
    post(server.base, "/api/review", {"id": "interrupt/b_nau", "verdict": "bad", "set": "b"})
    _, a = get_json(server.base, "/api/review?set=a")
    _, b = get_json(server.base, "/api/review?set=b")
    _, d = get_json(server.base, "/api/review")
    assert a["set"] == "a" and set(a["labels"]) == {"interrupt/a_ca"}
    assert set(b["labels"]) == {"interrupt/b_nau"} and d["labels"] == {}


def test_create_from_copies_the_sample_not_the_labels(server):
    build_films(server.final)
    get_json(server.base, sample_url(set="src", new=1, n=4, seed=3))
    src_ids = get_json(server.base, sample_url(set="src"))[1]["items"]
    post(server.base, "/api/review", {"id": src_ids[0]["id"], "verdict": "bad", "set": "src"})
    status, row = post(server.base, "/api/review_sets", {"name": "again", "from": "src"})
    assert status == 201 and row["labeled"] == 0 and row["n_sample"] == 4
    copied = get_json(server.base, sample_url(set="again"))[1]
    assert ids(copied) == ids({"items": src_ids})
    assert all(i["labeled"] is None for i in copied["items"])
    assert post(server.base, "/api/review_sets", {"name": "x", "from": "missing"})[0] == 404


def test_saved_sample_is_restored_in_order_and_counts_missing(server):
    build_films(server.final)
    first = get_json(server.base, sample_url(set="s", new=1, n=6, seed=2))[1]
    assert first["set"] == "s" and len(first["items"]) == 6
    shutil.rmtree(server.final / first["items"][0]["id"])
    rescan()
    again = get_json(server.base, sample_url(set="s"))[1]
    assert again["missing"] == 1 and ids(again) == ids(first)[1:]
    assert "restored_from_labels" not in again


def test_sample_repeats_exactly_when_final_grows_and_only_new_redraws(server):
    build_films(server.final)
    saved = get_json(server.base, sample_url(set="fixed", new=1, n=10, seed=7))[1]
    for i in range(20):
        write_conv(server.final, f"interrupt/extra{i}", topic="x", duration=1.0,
                   source_stem=f"Extra Show {i % 4} Tập 1", mtime=50 + i)
    rescan()
    repeated = get_json(server.base, sample_url(set="fixed", n=10, seed=7))[1]
    assert repeated["items"] == saved["items"] and repeated["missing"] == 0
    assert repeated["set"] == "fixed"
    redrawn = get_json(server.base, sample_url(set="fixed", new=1, n=10, seed=7))[1]
    assert ids(redrawn) != ids(saved)
    assert ids(get_json(server.base, sample_url(set="fixed"))[1]) == ids(redrawn)


def test_new_sample_keeps_labels_and_a_label_never_edits_the_sample(server):
    build_films(server.final)
    get_json(server.base, sample_url(set="k", new=1, n=3, seed=1))
    path = server.root / "review_sets" / "k.json"
    before = json.loads(path.read_text("utf-8"))["sample"]
    outside = next(i for i in ("interrupt/d1", "turn_by_turn/c3") if i not in before["ids"])
    status, _ = post(server.base, "/api/review", {"id": outside, "verdict": "ok", "set": "k"})
    assert status == 200
    doc = json.loads(path.read_text("utf-8"))
    assert doc["sample"] == before and outside in doc["labels"]
    post(server.base, "/api/review", {"id": before["ids"][0], "verdict": "bad", "set": "k"})
    get_json(server.base, sample_url(set="k", new=1, n=3, seed=1))
    doc = json.loads(path.read_text("utf-8"))
    assert doc["labels"][before["ids"][0]]["verdict"] == "bad"


def test_set_without_sample_and_without_labels_is_409(server):
    build_films(server.final)
    post(server.base, "/api/review_sets", {"name": "empty"})
    assert get_json(server.base, sample_url(set="empty"))[0] == 409
    assert get_json(server.base, sample_url(set="nope"))[0] == 404


def test_labelled_set_without_sample_restores_its_labelled_ids_in_order(server):
    build_films(server.final)
    raw = {"backchannel/a2": {"verdict": "bad", "tags": [], "note": "", "time": 300.0},
           "interrupt/a1": {"verdict": "ok", "tags": [], "note": "", "time": 100.0}}
    rdir = server.root / "review_sets"
    rdir.mkdir()
    (rdir / "old.json").write_text(json.dumps({"name": "old", "created": None, "updated": None,
                                               "sample": None, "labels": raw}), encoding="utf-8")
    status, data = get_json(server.base, sample_url(set="old"))
    assert status == 200 and data["restored_from_labels"] is True
    assert ids(data) == ["interrupt/a1", "backchannel/a2"]
    stored = json.loads((rdir / "old.json").read_text("utf-8"))
    assert stored["sample"]["ids"] == ["interrupt/a1", "backchannel/a2"]
    again = get_json(server.base, sample_url(set="old"))[1]
    assert "restored_from_labels" not in again and ids(again) == ids(data)


def test_default_set_takes_over_the_old_labels_file(server):
    build_library(server.final)
    folder = server.final / "_review"
    folder.mkdir()
    legacy = {"interrupt/a_ca": {"verdict": "bad", "tags": ["old"], "note": "", "time": 5.0}}
    (folder / "labels.json").write_text(json.dumps(legacy), encoding="utf-8")
    _, data = get_json(server.base, "/api/review")
    assert set(data["labels"]) == {"interrupt/a_ca"}
    status, data = post(server.base, "/api/review", {"id": "interrupt/b_nau", "verdict": "ok"})
    assert status == 200 and set(data["labels"]) == {"interrupt/a_ca", "interrupt/b_nau"}
    assert json.loads((folder / "labels.json").read_text("utf-8")) == legacy
    stored = json.loads((server.root / "review_sets" / "default.json").read_text("utf-8"))
    assert set(stored["labels"]) == {"interrupt/a_ca", "interrupt/b_nau"}
    assert stored["labels"]["interrupt/a_ca"]["tags"] == ["old"]


def read_csv(base, url):
    status, headers, body = get(base, url)
    assert status == 200
    assert headers["Content-Type"] == "text/csv; charset=utf-8"
    assert body.startswith(b"\xef\xbb\xbf")
    return headers, list(csv.reader(io.StringIO(body[3:].decode("utf-8"))))


def test_csv_has_bom_columns_escapes_and_unlabelled_rows(server):
    build_films(server.final)
    get_json(server.base, sample_url(set="out", new=1, n=3, seed=0))
    rel = get_json(server.base, sample_url(set="out"))[1]["items"][0]["id"]
    post(server.base, "/api/review", {
        "id": rel, "verdict": "bad", "set": "out",
        "tags": ["a,b", 'q"z', "=HYPERLINK(1)"], "note": '=1+2, "q"\nline2'})
    headers, rows = read_csv(server.base, "/api/review.csv?set=out")
    assert headers["Content-Disposition"] == 'attachment; filename="review_out.csv"'
    assert rows[0] == ["id", "film", "category", "verdict", "tags", "note", "time",
                       "source_stem", "duration"]
    assert len(rows) == 4
    by_id = {r[0]: r for r in rows[1:]}
    row = by_id[rel]
    assert row[3] == "bad"
    assert row[4] == 'a,b;q"z;=HYPERLINK(1)'
    assert row[5] == '\'=1+2, "q"\nline2'
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", row[6])
    assert row[7] and row[8]
    unlabelled = [r for r in rows[1:] if r[0] != rel]
    assert len(unlabelled) == 2 and all(r[3] == "" and r[5] == "" for r in unlabelled)
    assert all(r[7] for r in rows[1:])


def test_csv_without_sample_lists_labels_and_flags_formula_text(server):
    build_library(server.final)
    post(server.base, "/api/review", {"id": "interrupt/a_ca", "verdict": "ok",
                                      "note": "@SUM(1)"})
    post(server.base, "/api/review", {"id": "backchannel/c_sang", "verdict": "bad",
                                      "note": "-1"})
    _, rows = read_csv(server.base, "/api/review.csv")
    assert [r[0] for r in rows[1:]] == ["backchannel/c_sang", "interrupt/a_ca"]
    notes = {r[0]: r[5] for r in rows[1:]}
    assert notes == {"backchannel/c_sang": "'-1", "interrupt/a_ca": "'@SUM(1)"}
    assert get_json(server.base, "/api/review.csv?set=nope")[0] == 404
