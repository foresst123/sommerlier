import json
import sys
import time
import urllib.request
from pathlib import Path
from threading import Thread
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import job_dashboard as jd  # noqa: E402


def make_out(tmp_path):
    out = tmp_path / "out"
    (out / "_resident" / "logs").mkdir(parents=True)
    inp = out / "_resident" / "input_1"
    inp.mkdir()
    for s in ("a", "b", "c"):
        (inp / f"{s}.wav").write_text("x")
    prod = out / "prod"
    prod.mkdir()
    (prod / ".ready_denoise.a").write_text("{}")
    (prod / ".ready_denoise.b").write_text("{}")
    (prod / ".claim_denoise.c").write_text(json.dumps({"host": "h", "pid": 1}))
    (prod / ".ready_diarization.a").write_text("{}")
    (prod / ".failed_diarization.b").write_text(json.dumps({"reason": "boom"}))
    (out / "_resident" / "logs" / "front_1_denoise_gpu0.log").write_text("hello\nlast line\n")
    (out / "_resident" / "logs" / "front.pids").write_text("999999 front_1_denoise_gpu0\n")
    return out


def test_counts_and_failures(tmp_path):
    out = make_out(tmp_path)
    data = jd.collect(out, out / "final")
    assert data["files"] == 3
    den = data["stages"]["denoise"]
    assert (den["done"], den["running"], den["failed"], den["total"]) == (2, 1, 0, 3)
    dia = data["stages"]["diarization"]
    assert (dia["done"], dia["failed"]) == (1, 1)
    assert data["failures"][0]["file"] == "b" and data["failures"][0]["reason"] == "boom"
    assert data["processes"][0]["line"] == "last line"
    assert data["final"]["manifest"] is False


def test_rate_and_eta(tmp_path):
    out = make_out(tmp_path)
    den = jd.collect(out, out / "final")["stages"]["denoise"]
    assert den["rate"] > 0 and den["eta_min"] is not None


def test_final_manifest(tmp_path):
    out = make_out(tmp_path)
    parts = out / "final" / "lhotse" / ".parts"
    parts.mkdir(parents=True)
    (parts / "a.jsonl.gz").write_bytes(b"x")
    (out / "final" / "lhotse" / "manifest.jsonl.gz").write_bytes(b"xx")
    f = jd.collect(out, out / "final")["final"]
    assert f["manifest"] and f["recordings"] == 1


def test_server_serves_page_and_api(tmp_path):
    out = make_out(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), jd.make_handler(out, out / "final"))
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        assert b"<title>Job dashboard</title>" in urllib.request.urlopen(base + "/").read()
        api = json.loads(urllib.request.urlopen(base + "/api").read())
        assert api["files"] == 3
    finally:
        server.shutdown()


def test_gpus_of_other_hosts_are_read(tmp_path):
    out = make_out(tmp_path)
    (out / "_resident" / "gpu_other.json").write_text(json.dumps(
        {"host": "other", "time": time.time(), "gpus": [{"index": "2", "name": "x", "used": 1, "total": 2, "util": 3}]}))
    (out / "_resident" / "gpu_old.json").write_text(json.dumps(
        {"host": "old", "time": time.time() - 999, "gpus": [{"index": "0", "name": "x", "used": 1, "total": 2, "util": 3}]}))
    hosts = jd.all_gpus(out)
    assert "other" in hosts and "old" not in hosts


def test_publish_gpus_writes_file(tmp_path, monkeypatch):
    out = make_out(tmp_path)
    monkeypatch.setattr(jd, "gpus", lambda: [{"index": "0", "name": "x", "used": 1, "total": 2, "util": 3}])
    jd.publish_gpus(out)
    assert list((out / "_resident").glob("gpu_*.json"))


def test_ansi_codes_are_stripped_from_the_log_line(tmp_path):
    out = make_out(tmp_path)
    (out / "_resident" / "logs" / "front_1_denoise_gpu0.log").write_text("\x1b[1;34mhello\x1b[0m\n")
    assert jd.collect(out, out / "final")["processes"][0]["line"] == "hello"


def test_manifest_hours_total_and_by_category(tmp_path):
    import gzip
    out = make_out(tmp_path)
    lh = out / "final" / "lhotse"
    lh.mkdir(parents=True)
    cuts = [
        {"duration": 3600.0, "recording": {"sources": [{"source": "/x/final/interrupt/a_10s/stereo_2ch.wav"}]}},
        {"duration": 1800.0, "recording": {"sources": [{"source": "/x/final/turn_by_turn/b_5s/stereo_2ch.wav"}]}},
        {"duration": 900.0, "recording": {"sources": [{"source": "/x/final/interrupt/c_7s/stereo_2ch.wav"}]}},
    ]
    with gzip.open(lh / "manifest.jsonl.gz", "wt", encoding="utf-8") as f:
        for c in cuts:
            f.write(json.dumps(c) + "\n")
        f.write("not json\n")
    data = jd.final_state(out / "final")["data"]
    assert data["cuts"] == 3 and data["hours"] == 1.75
    assert data["by_category"]["interrupt"] == {"cuts": 2, "hours": 1.25}
    assert data["by_category"]["turn_by_turn"]["cuts"] == 1


def test_manifest_hours_is_none_without_a_manifest(tmp_path):
    out = make_out(tmp_path)
    assert jd.final_state(out / "final")["data"] is None


def test_category_is_read_from_the_recording_id_when_the_audio_is_a_swapped_copy(tmp_path):
    import gzip
    out = make_out(tmp_path)
    lh = out / "final" / "lhotse"
    lh.mkdir(parents=True)
    cut = {"duration": 3600.0, "recording": {"id": "final-backchannel-stereo_2ch-abc",
           "sources": [{"source": "/x/final/lhotse/audio/item_5s_1234abcd.wav"}]}}
    with gzip.open(lh / "manifest.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(cut) + "\n")
    data = jd.final_state(out / "final")["data"]
    assert data["by_category"] == {"backchannel": {"cuts": 1, "hours": 1.0}}


def test_the_listening_routes_and_the_review_post_work_through_the_dashboard(tmp_path):
    import urllib.request
    out = make_out(tmp_path)
    folder = out / "final" / "interrupt" / "xxxxxxxxxxx_Phim A - Tập 1_10s"
    folder.mkdir(parents=True)
    (folder / "conversation.json").write_text(json.dumps(
        {"topic": "t", "duration": 12, "source_stem": "xxxxxxxxxxx_Phim A - Tập 1", "conversation": []}))
    server = ThreadingHTTPServer(("127.0.0.1", 0), jd.make_handler(out, out / "final", {"audio_dir": None, "total_recordings": 3}))
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        listing = json.loads(urllib.request.urlopen(base + "/api/conversations").read())
        assert listing["total"] == 1
        cid = listing["items"][0]["id"]
        sample = json.loads(urllib.request.urlopen(base + "/api/sample?n=5").read())
        assert sample["n"] == 1 and sample["items"][0]["film"]
        req = urllib.request.Request(base + "/api/review", data=json.dumps({"id": cid, "verdict": "bad", "tags": ["nhiễu"]}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        assert json.loads(urllib.request.urlopen(req).read())["summary"]["bad"] == 1
        stats = json.loads(urllib.request.urlopen(base + "/api/stats").read())
        assert stats["input"]["recordings_total"] == 3 and stats["output"]["conversations"] == 1
        try:
            urllib.request.urlopen(urllib.request.Request(base + "/nope", data=b"{}", method="POST"))
            assert False, "an unknown POST must be a 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
