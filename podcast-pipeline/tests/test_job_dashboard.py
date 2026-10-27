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
