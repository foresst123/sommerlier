"""Backend of the diarization lab: audio list, parameter schema, runs (queue, stop, delete),
results, compare, Range. A real ThreadingHTTPServer on port 0 calls job_diar.handle/handle_post.
ctx["python"] is a fake main.py script that writes segments.json from the --set values it gets.

Run:  python -m pytest tests/test_job_diar.py -q -p no:cacheprovider     (from podcast-pipeline/)
"""
import http.client
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import pytest

from tools import job_diar

PIPE = Path(job_diar.__file__).resolve().parent.parent

FAKE = """#!{python}
import json, os, sys, time
args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1]
sets = {{}}
for i, a in enumerate(args):
    if a == "--set":
        k, _, v = args[i + 1].partition("=")
        sets[k] = v
print("fake main CUDA_VISIBLE_DEVICES=" + str(os.environ.get("CUDA_VISIBLE_DEVICES")), flush=True)
if sets.get("models.nemotron.batch_size") == "7":
    time.sleep(60)
swap = sets.get("models.nemotron.batch_size") == "9"
drop_b = sets.get("models.nemotron.vad_threshold") == "0.9"
a_name, b_name = ("SPEAKER_01", "SPEAKER_00") if swap else ("SPEAKER_00", "SPEAKER_01")
segs = [{{"start": 0.0, "end": 2.0, "speaker": a_name}}]
if not drop_b:
    segs.append({{"start": 2.0, "end": 4.0, "speaker": b_name}})
save = opt("--save_path")
stem = os.path.splitext(os.path.basename(opt("--audio")))[0]
d = os.path.join(save, stem, "02_diarization")
os.makedirs(d, exist_ok=True)
for name in ("segments.json", "segments_raw.json"):
    with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
        json.dump(segs, fh)
with open(os.path.join(d, "stats.json"), "w", encoding="utf-8") as fh:
    json.dump({{"n_segments": len(segs)}}, fh)
print("fake main done", flush=True)
"""


def write_wav(path, seconds, rate=16000, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes((i * 37) % 256 for i in range(frames * 2 * channels)))
    return path


@pytest.fixture
def lab(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    write_wav(audio / "Chào_buổi_sáng.wav", 6.0)
    write_wav(audio / "a" / "b" / "deep.wav", 1.0)
    write_wav(audio / "a" / "b" / "c" / "too_deep.wav", 1.0)
    (audio / "x.mp3").write_bytes(b"not really mp3")
    (audio / "notes.txt").write_text("hi", encoding="utf-8")
    outside = tmp_path / "outside"
    write_wav(outside / "secret.wav", 1.0)
    os.symlink(outside / "secret.wav", audio / "link.wav")

    fake = tmp_path / "fake_main.py"
    fake.write_text(FAKE.format(python=sys.executable), encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    ctx = {"audio_dir": audio, "lab_dir": tmp_path / "lab", "pipeline_dir": PIPE,
           "python": str(fake), "env_name": "a100", "config_path": PIPE / "config.json"}
    monkeypatch.setattr(job_diar, "_LIST_CACHE", {})
    monkeypatch.setattr(job_diar, "_DUR_CACHE", {})
    monkeypatch.setattr(job_diar, "_RESULTS", {})
    yield {"ctx": ctx, "audio": audio, "lab": tmp_path / "lab", "tmp": tmp_path}
    job_diar._stop_all_for_tests()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with job_diar._CV:
            if job_diar._CURRENT is None and not job_diar._QUEUE:
                break
        time.sleep(0.05)


class Server:
    def __init__(self, ctx):
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _route(self, method):
                u = urlsplit(self.path)
                path = u.path.rstrip("/") or "/"
                if method == "GET":
                    handled = job_diar.handle(self, path, parse_qs(u.query), outer.ctx)
                else:
                    handled = job_diar.handle_post(self, path, outer.ctx)
                if not handled:
                    self.send_error(404)

            def do_GET(self):
                self._route("GET")

            def do_POST(self):
                self._route("POST")

        self.ctx = ctx
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()

    def call(self, method, path, body=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        data = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
        hdrs = dict(headers or {})
        if data is not None and "Content-Type" not in hdrs:
            hdrs["Content-Type"] = "application/json"
        conn.request(method, path, body=data, headers=hdrs)
        resp = conn.getresponse()
        payload = resp.read()
        conn.close()
        return resp.status, dict(resp.getheaders()), payload

    def js(self, method, path, body=None):
        status, _, payload = self.call(method, path, body)
        return status, json.loads(payload.decode("utf-8")) if payload else None

    def run(self, body):
        status, doc = self.js("POST", "/api/diar/run", body)
        assert status == 201, doc
        return doc["id"]

    def wait(self, rid, states=("done", "failed", "stopped"), timeout=60):
        deadline = time.monotonic() + timeout
        seen = []
        while time.monotonic() < deadline:
            _, doc = self.js("GET", f"/api/diar/run?id={rid}")
            if not seen or seen[-1] != doc["state"]:
                seen.append(doc["state"])
            if doc["state"] in states:
                return doc, seen
            time.sleep(0.05)
        raise AssertionError(f"run {rid} did not reach {states}; seen {seen}")


@pytest.fixture
def server(lab):
    s = Server(lab["ctx"])
    yield s
    s.close()


# -- page, audio list, params, gpus -------------------------------------------------------
def test_page_present_and_missing(server, monkeypatch, lab):
    page = lab["tmp"] / "diar_lab_page.html"
    page.write_text("<html>ok</html>", encoding="utf-8")
    monkeypatch.setattr(job_diar, "PAGE_PATH", page)
    status, headers, body = server.call("GET", "/diar")
    assert status == 200 and body == b"<html>ok</html>"
    assert headers["Content-Type"].startswith("text/html")
    assert headers["Cache-Control"] == "no-store"
    monkeypatch.setattr(job_diar, "PAGE_PATH", lab["tmp"] / "missing.html")
    status, _, body = server.call("GET", "/diar")
    assert status == 404 and b"missing" in body


def test_audio_list_depth_filter_and_duration(server):
    _, doc = server.js("GET", "/api/diar/audio?limit=50")
    paths = [f["path"] for f in doc["files"]]
    assert "Chào_buổi_sáng.wav" in paths
    assert "a/b/deep.wav" in paths
    assert "x.mp3" in paths
    assert "a/b/c/too_deep.wav" not in paths
    assert "notes.txt" not in paths
    assert "link.wav" not in paths      # symlink to a file outside audio_dir
    by = {f["path"]: f for f in doc["files"]}
    assert by["Chào_buổi_sáng.wav"]["duration"] == pytest.approx(6.0, abs=0.01)
    assert by["x.mp3"]["duration"] is None
    assert by["x.mp3"]["size"] == len(b"not really mp3")


def test_audio_list_query_ignores_case_and_tone_marks(server):
    for q in ("chao", "CHAO", "Chào", "sang", "BUOI"):
        _, doc = server.js("GET", "/api/diar/audio?q=" + quote(q))
        assert [f["path"] for f in doc["files"]] == ["Chào_buổi_sáng.wav"], q
    _, doc = server.js("GET", "/api/diar/audio?q=" + quote("b/c"))
    assert doc["files"] == []


def test_audio_list_limit_and_no_audio_dir(server, lab):
    _, doc = server.js("GET", "/api/diar/audio?limit=1")
    assert len(doc["files"]) == 1 and doc["total"] >= 3
    none_srv = Server(dict(lab["ctx"], audio_dir=None))
    try:
        _, doc = none_srv.js("GET", "/api/diar/audio")
        assert doc["files"] == [] and "note" in doc
    finally:
        none_srv.close()


def test_params_schema_defaults_from_config(server):
    status, doc = server.js("GET", "/api/diar/params")
    assert status == 200 and doc["env"] == "a100"
    items = {p["key"]: p for g in doc["groups"] for p in g["params"]}
    assert len(items) == sum(len(g["params"]) for g in doc["groups"])   # no duplicates
    cfg = json.loads((PIPE / "config.json").read_text(encoding="utf-8"))["environments"]["a100"]
    nem = cfg["models"]["nemotron"]
    vad = items["models.nemotron.vad_threshold"]
    assert vad["type"] == "float" and vad["default"] == nem["vad_threshold"]
    assert vad["min"] < vad["default"] < vad["max"]
    assert items["models.nemotron.two_pass"]["type"] == "bool"
    assert items["models.nemotron.two_pass"]["default"] is True
    assert items["models.nemotron.batch_size"]["type"] == "int"
    backend = items["models.diarization_backend"]
    assert backend["type"] == "choice" and backend["default"] == "nemotron"
    assert backend["choices"] == ["nemotron", "diarizen"]
    denoise = items["pipeline.denoise"]
    assert denoise["default"] == "rf_crowd>blend50" and "none" in denoise["choices"]
    assert "models.diarizen.num_speakers" not in items        # not in config, so not offered
    assert items["models.diarizen.max_speakers"]["default"] == cfg["models"]["diarizen"]["max_speakers"]
    for p in items.values():
        assert p["help"].strip() and p["label"].strip()


def test_gpus_parses_nvidia_smi(server, monkeypatch):
    out = "0, NVIDIA A100-SXM4-80GB, 1024, 81920, 37\n1, NVIDIA A100, [N/A], 81920, 0\n"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=out, stderr="")

    monkeypatch.setattr(job_diar.subprocess, "run", fake_run)
    _, doc = server.js("GET", "/api/diar/gpus")
    assert doc["gpus"][0] == {"index": 0, "name": "NVIDIA A100-SXM4-80GB", "used_mb": 1024,
                              "total_mb": 81920, "util": 37}
    assert doc["gpus"][1]["used_mb"] is None

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(job_diar.subprocess, "run", missing)
    _, doc = server.js("GET", "/api/diar/gpus")
    assert doc == {"gpus": []}


# -- build_cmd -----------------------------------------------------------------------------
def test_build_cmd_flags_and_set(lab):
    run_dir = lab["lab"] / "runs" / "rid_1"
    params = {"params": {"models.nemotron.vad_threshold": 0.5, "pipeline.denoise": "none",
                         "models.nemotron.two_pass": False, "models.nemotron.batch_size": 9}}
    cmd = job_diar.build_cmd(run_dir, params, lab["ctx"])
    assert cmd[0] == lab["ctx"]["python"] and cmd[1] == "main.py"

    def val(flag):
        return cmd[cmd.index(flag) + 1]

    assert val("--audio") == str((run_dir / "input.wav").resolve())
    assert val("--env") == "a100"
    assert val("--stop_after") == "diarization"
    assert val("--save_path") == str((run_dir / "out").resolve())
    assert val("--cache_dir") == str((run_dir / "cache").resolve())
    assert val("--job_id") == "rid_1"
    assert "--no_stage_audio" in cmd
    assert val("--gpu_1") == "0" and val("--gpu_2") == "0"
    assert val("--config") == str(PIPE / "config.json")
    sets = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--set"]
    assert sets == ["models.nemotron.vad_threshold=0.5", "pipeline.denoise=none",
                    "models.nemotron.two_pass=false", "models.nemotron.batch_size=9"]


# -- runs ----------------------------------------------------------------------------------
def test_run_full_flow_and_result(server, lab):
    rid = server.run({"audio": "Chào_buổi_sáng.wav", "start": 1.0, "end": 3.0,
                      "params": {"models.nemotron.vad_threshold": 0.5}, "gpu": 0, "label": "thử"})
    doc, seen = server.wait(rid)
    assert doc["state"] == "done", doc
    assert set(seen) <= {"queued", "running", "done"} and seen[-1] == "done"
    assert doc["summary"]["n_speakers"] == 2
    assert doc["summary"]["speech_s"] == pytest.approx(4.0)
    assert doc["label"] == "thử" and doc["params"] == {"models.nemotron.vad_threshold": 0.5}
    assert any("CUDA_VISIBLE_DEVICES=0" in line for line in doc["log_tail"])
    assert "fake main done" in doc["log_tail"]
    assert "--set" in doc["cmd"] and "models.nemotron.vad_threshold=0.5" in doc["cmd"]

    run_dir = lab["lab"] / "runs" / rid
    assert (run_dir / "params.json").is_file() and (run_dir / "run.log").is_file()
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "done" and status["returncode"] == 0

    _, res = server.js("GET", f"/api/diar/result?id={rid}")
    assert res["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert res["segments"] == [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
                               {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"}]
    assert res["raw_segments"] == res["segments"]
    assert res["audio_url"] == f"/api/diar/audio_file?id={rid}"
    assert res["stats"]["n_speakers"] == 2 and res["stats"]["overlap_s"] == 0.0
    assert res["stats"]["per_speaker"]["SPEAKER_01"] == {"s": 2.0, "n": 1}
    assert res["stats"]["n_segments"] == 2 and res["stats"]["short_lt_0_5s"] == 0
    assert res["duration"] == pytest.approx(2.0, abs=0.01)


def test_cut_length_and_resample_from_48k_stereo(server, lab):
    import soundfile as sf
    import numpy as np
    src = lab["audio"] / "stereo48.wav"
    sf.write(str(src), np.zeros((48000 * 3, 2), dtype="float32"), 48000, subtype="PCM_16")
    rid = server.run({"audio": "stereo48.wav", "start": 0.5, "end": 2.5})
    server.wait(rid)
    info = sf.info(str(lab["lab"] / "runs" / rid / "input.wav"))
    assert info.samplerate == 16000 and info.channels == 1
    assert abs(info.frames - 32000) <= 2


def test_cut_full_file_without_start_end(server, lab):
    rid = server.run({"audio": "Chào_buổi_sáng.wav"})
    server.wait(rid)
    params = json.loads((lab["lab"] / "runs" / rid / "params.json").read_text(encoding="utf-8"))
    assert params["start"] == 0.0 and params["end"] == pytest.approx(6.0, abs=0.01)


def test_cut_with_ffmpeg_fallback(server, lab, monkeypatch):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    monkeypatch.setitem(job_diar._MODS, "soundfile", None)
    monkeypatch.setitem(job_diar._MODS, "numpy", None)
    rid = server.run({"audio": "Chào_buổi_sáng.wav", "start": 1.0, "end": 3.0})
    server.wait(rid)
    wav = lab["lab"] / "runs" / rid / "input.wav"
    with wave.open(str(wav), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        assert abs(w.getnframes() / 16000 - 2.0) < 0.05


def test_cut_without_any_tool_is_500(server, lab, monkeypatch):
    monkeypatch.setitem(job_diar._MODS, "soundfile", None)
    monkeypatch.setitem(job_diar._MODS, "numpy", None)
    monkeypatch.setattr(job_diar.shutil, "which", lambda name: None)
    status, doc = server.js("POST", "/api/diar/run", {"audio": "Chào_buổi_sáng.wav", "start": 1, "end": 2})
    assert status == 500 and "thiếu công cụ" in doc["error"]
    runs = lab["lab"] / "runs"
    assert not runs.exists() or not any(runs.iterdir())


def test_validation_rejects_bad_input(server, lab, monkeypatch):
    base = {"audio": "Chào_buổi_sáng.wav", "start": 1.0, "end": 3.0}
    cases = [
        dict(base, params={"models.nemotron.nope": 1}),                       # unknown key
        dict(base, params={"models.nemotron.vad_threshold": "abc"}),           # wrong type
        dict(base, params={"models.nemotron.vad_threshold": 1.5}),             # out of range
        dict(base, params={"models.nemotron.batch_size": True}),               # bool for int
        dict(base, params={"models.nemotron.batch_size": 2.5}),                # not integral
        dict(base, params={"models.diarization_backend": "xyz"}),              # bad choice
        dict(base, params={"models.nemotron.two_pass": "yes"}),                # bool expected
        dict(base, params={"models.nemotron.chunk_min_seconds": 40.0}),       # min > target
        dict(base, params={"models.diarizen.min_speakers": 5, "models.diarizen.max_speakers": 3}),
        dict(base, end=1.0),                                                   # end <= start
        dict(base, start=-1.0),
        dict(base, start="x"),
        dict(base, end=9.0),                                                   # beyond the file
        dict(base, gpu=99),
        dict(base, gpu=True),
        dict(base, label="x" * 81),
        dict(base, params=[1, 2]),
        {"audio": "../outside/secret.wav", "start": 0, "end": 1},
        {"audio": "/etc/passwd", "start": 0, "end": 1},
        {"audio": "a\x00b.wav", "start": 0, "end": 1},
        {"audio": "a//b.wav", "start": 0, "end": 1},
        {"audio": "link.wav", "start": 0, "end": 1},                           # symlink out of audio_dir
        {"audio": "notes.txt", "start": 0, "end": 1},                          # not audio
        {"audio": None, "start": 0, "end": 1},
    ]
    for body in cases:
        status, doc = server.js("POST", "/api/diar/run", body)
        assert status == 400, (body, status, doc)
    status, doc = server.js("POST", "/api/diar/run", {"audio": "missing.wav", "start": 0, "end": 1})
    assert status == 404
    monkeypatch.setattr(job_diar, "MAX_RUN_SECONDS", 1.5)
    status, doc = server.js("POST", "/api/diar/run", base)
    assert status == 400 and "tối đa" in doc["error"]
    status, _, _ = server.call("POST", "/api/diar/run", raw=b"{not json", headers={"Content-Type": "application/json"})
    assert status == 400
    status, _, _ = server.call("POST", "/api/diar/run", raw=b"[1,2]")
    assert status == 400
    status, _, _ = server.call("POST", "/api/diar/run", raw=b"x" * (70 * 1024))
    assert status == 413
    runs = lab["lab"] / "runs"
    assert not runs.exists() or not any(runs.iterdir())


def test_run_and_result_ids_and_404(server):
    assert server.js("GET", "/api/diar/run?id=../x")[0] == 400
    assert server.js("GET", "/api/diar/run?id=nope")[0] == 404
    assert server.js("GET", "/api/diar/result?id=nope")[0] == 404
    assert server.js("GET", "/api/diar/compare?a=nope&b=nope")[0] == 404
    assert server.js("POST", "/api/diar/stop", {"id": "nope"})[0] == 404


def test_queued_then_stopped_and_running_protected(server, lab):
    slow = server.run({"audio": "Chào_buổi_sáng.wav", "start": 0, "end": 2,
                       "params": {"models.nemotron.batch_size": 7}})
    server.wait(slow, states=("running",))
    fast = server.run({"audio": "Chào_buổi_sáng.wav", "start": 0, "end": 2})
    doc, _ = server.wait(fast, states=("queued",), timeout=5)
    assert doc["state"] == "queued"
    assert server.js("GET", f"/api/diar/result?id={fast}")[0] == 409
    assert server.js("POST", "/api/diar/delete", {"id": slow, "confirm": True})[0] == 409
    assert server.js("POST", "/api/diar/delete", {"id": fast})[0] == 400  # confirm missing

    status, doc = server.js("POST", "/api/diar/stop", {"id": fast})
    assert status == 200 and doc["state"] == "stopped"
    t0 = time.monotonic()
    status, doc = server.js("POST", "/api/diar/stop", {"id": slow})
    assert status == 200 and doc["state"] == "stopped", doc
    assert time.monotonic() - t0 < job_diar.STOP_WAIT
    assert server.js("POST", "/api/diar/stop", {"id": slow})[0] == 409
    assert server.js("GET", f"/api/diar/result?id={slow}")[0] == 409

    assert server.js("POST", "/api/diar/delete", {"id": slow, "confirm": True})[0] == 200
    assert not (lab["lab"] / "runs" / slow).exists()
    assert server.js("GET", f"/api/diar/run?id={slow}")[0] == 404


def test_stop_sends_group_kill_and_run_is_stopped(server):
    rid = server.run({"audio": "Chào_buổi_sáng.wav", "start": 0, "end": 2,
                      "params": {"models.nemotron.batch_size": 7}})
    server.wait(rid, states=("running",))
    status, doc = server.js("POST", "/api/diar/stop", {"id": rid})
    assert status == 200 and doc["state"] == "stopped"
    final, _ = server.wait(rid, states=("stopped",), timeout=5)
    assert final["state"] == "stopped" and final["returncode"] not in (None, 0)


def test_failed_run_reports_returncode(server, lab):
    bad = lab["tmp"] / "bad_main.py"
    bad.write_text(f"#!{sys.executable}\nimport sys\nprint('boom')\nsys.exit(3)\n", encoding="utf-8")
    bad.chmod(0o755)
    server.ctx["python"] = str(bad)
    rid = server.run({"audio": "Chào_buổi_sáng.wav", "start": 0, "end": 2})
    doc, _ = server.wait(rid)
    assert doc["state"] == "failed" and "3" in doc["error"]
    assert doc["log_tail"][-1] == "boom"
    server.ctx["python"] = str(lab["tmp"] / "fake_main.py")


def test_runs_list_newest_first_and_changed(server):
    first = server.run({"audio": "Chào_buổi_sáng.wav", "start": 0, "end": 2,
                        "params": {"models.nemotron.vad_threshold": 0.5}})
    server.wait(first)
    second = server.run({"audio": "Chào_buổi_sáng.wav", "start": 0, "end": 2,
                         "params": {"models.nemotron.batch_size": 9,
                                    "models.nemotron.vad_threshold": 0.35}})
    server.wait(second)
    _, doc = server.js("GET", "/api/diar/runs")
    ids = [r["id"] for r in doc["runs"]]
    assert ids.index(second) < ids.index(first)
    rows = {r["id"]: r for r in doc["runs"]}
    assert rows[first]["changed"] == 1
    assert rows[second]["changed"] == 1          # vad equals its default, so only batch_size counts
    assert rows[second]["n_speakers"] == 2 and rows[second]["state"] == "done"


def test_delete_requires_confirm_and_done_run_removed(server, lab):
    rid = server.run({"audio": "Chào_buổi_sáng.wav", "start": 0, "end": 2})
    server.wait(rid)
    assert server.js("POST", "/api/diar/delete", {"id": rid, "confirm": False})[0] == 400
    assert server.js("POST", "/api/diar/delete", {"id": rid, "confirm": True})[0] == 200
    assert not (lab["lab"] / "runs" / rid).exists()
    assert server.js("POST", "/api/diar/delete", {"id": rid, "confirm": True})[0] == 404


def test_audio_file_range(server, lab):
    rid = server.run({"audio": "Chào_buổi_sáng.wav", "start": 0, "end": 2})
    server.wait(rid)
    status, headers, body = server.call("GET", f"/api/diar/audio_file?id={rid}")
    assert status == 200
    full_len = len(body)
    assert full_len == os.path.getsize(lab["lab"] / "runs" / rid / "input.wav")
    status, headers, body = server.call("GET", f"/api/diar/audio_file?id={rid}", headers={"Range": "bytes=0-9"})
    assert status == 206 and len(body) == 10
    assert headers["Content-Range"] == f"bytes 0-9/{full_len}"
    status, headers, body = server.call("GET", f"/api/diar/audio_file?id={rid}",
                                        headers={"Range": "bytes=40-"})
    assert status == 206 and len(body) == full_len - 40
    status, headers, _ = server.call("GET", f"/api/diar/audio_file?id={rid}",
                                     headers={"Range": "bytes=999999999-"})
    assert status == 416 and headers["Content-Range"] == f"bytes */{full_len}"
    status, _, _ = server.call("GET", f"/api/diar/audio_file?id={rid}", headers={"Range": "bytes=5-2"})
    assert status == 416


def test_compare_permutation_and_miss(server):
    r1 = server.run({"audio": "Chào_buổi_sáng.wav", "start": 0, "end": 4})
    server.wait(r1)
    r2 = server.run({"audio": "Chào_buổi_sáng.wav", "start": 0, "end": 4,
                     "params": {"models.nemotron.batch_size": 9}})
    server.wait(r2)
    r3 = server.run({"audio": "Chào_buổi_sáng.wav", "start": 0, "end": 4,
                     "params": {"models.nemotron.vad_threshold": 0.9}})
    server.wait(r3)

    status, same = server.js("GET", f"/api/diar/compare?a={r1}&b={r2}")
    assert status == 200
    assert same["der_like"] == 0.0 and same["miss_s"] == 0 and same["false_alarm_s"] == 0
    assert same["confusion_s"] == 0 and same["disagreement_s"] == 0
    assert same["mapping"] == {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"}
    assert same["lanes"] == []

    _, miss = server.js("GET", f"/api/diar/compare?a={r1}&b={r3}")
    assert miss["miss_s"] == pytest.approx(2.0)
    assert miss["false_alarm_s"] == 0 and miss["confusion_s"] == 0
    assert miss["der_like"] == pytest.approx(0.5)            # 2 s missing out of 4 s of speaker time
    assert miss["lanes"] == [{"start": 2.0, "end": 4.0, "kind": "miss"}]
    assert miss["a"]["n_speakers"] == 2 and miss["b"]["n_speakers"] == 1
    assert miss["mapping"] == {"SPEAKER_00": "SPEAKER_00"}


def test_compare_pure_confusion_and_overlap_cases():
    # X is matched to P (2 s overlap), Y overlaps P for 1 s only: that second is confusion,
    # and the last second has Y alone (miss).
    a = [{"start": 0.0, "end": 2.0, "speaker": "X"}, {"start": 2.0, "end": 4.0, "speaker": "Y"}]
    b = [{"start": 0.0, "end": 3.0, "speaker": "P"}]
    out = job_diar.compare_rows(a, b)
    assert out["mapping"] == {"X": "P"}
    assert out["confusion_s"] == pytest.approx(1.0)
    assert out["miss_s"] == pytest.approx(1.0) and out["false_alarm_s"] == 0
    assert out["der_like"] == pytest.approx(0.5)
    assert out["lanes"] == [{"start": 2.0, "end": 3.0, "kind": "conf"},
                            {"start": 3.0, "end": 4.0, "kind": "miss"}]

    a2 = [{"start": 0.0, "end": 4.0, "speaker": "X"}, {"start": 1.0, "end": 3.0, "speaker": "Y"}]
    b2 = [{"start": 0.0, "end": 4.0, "speaker": "X"}]
    out2 = job_diar.compare_rows(a2, b2)
    assert out2["mapping"] == {"X": "X"}
    assert out2["miss_s"] == pytest.approx(2.0)          # Y is unmatched while B has only X there
    assert out2["false_alarm_s"] == 0 and out2["confusion_s"] == 0
    assert out2["der_like"] == pytest.approx(2.0 / 6.0, abs=1e-4)  # speaker time of A is 4 + 2


def test_compare_needs_finished_runs(server):
    rid = server.run({"audio": "Chào_buổi_sáng.wav", "start": 0, "end": 2,
                      "params": {"models.nemotron.batch_size": 7}})
    server.wait(rid, states=("running",))
    try:
        assert server.js("GET", f"/api/diar/compare?a={rid}&b={rid}")[0] == 409
    finally:
        server.js("POST", "/api/diar/stop", {"id": rid})


def test_hungarian_matches_brute_force_on_random_matrices():
    import itertools
    import random
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(1, 4)
        m = rng.randint(n, 5)
        w = [[rng.randint(0, 9) for _ in range(m)] for _ in range(n)]
        pairs = job_diar._assign([[-x for x in row] for row in w], n, m)
        got = sum(w[i][j] for i, j in pairs)
        best = max(sum(w[i][p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))
        assert got == best


def test_unknown_path_returns_false_and_500_is_json(server, monkeypatch):
    class Dummy:
        pass

    assert job_diar.handle(Dummy(), "/listen", {}, {}) is False
    assert job_diar.handle_post(Dummy(), "/api/review", {}) is False
    monkeypatch.setattr(job_diar, "_schema", lambda cfg: 1 / 0)
    status, doc = server.js("GET", "/api/diar/params")
    assert status == 500 and "error" in doc
