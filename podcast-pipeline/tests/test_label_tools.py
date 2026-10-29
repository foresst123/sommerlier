import json
import os
import sys
import threading
import urllib.error
import urllib.request
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from tools import label_server as LS  # noqa: E402
from tools import label_store as S  # noqa: E402


def make_sample(arms_dir, name="01_demo", seconds=6):
    d = arms_dir / name
    d.mkdir(parents=True)
    t = np.arange(int(16000 * seconds)) / 16000
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(d / "clip.wav"), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    (d / "segments_diarizen_auto.json").write_text(json.dumps({"segments": [
        {"start": 0.5, "end": 2.0, "speaker": "SPEAKER_00"}, {"start": 2.5, "end": 4.0, "speaker": "SPEAKER_01"}]}))
    return d


def test_validate_canonicalises_and_rejects_bad_input():
    doc = S.validate({"segments": [{"start": 2.0004, "end": 3, "speaker": "B"}, {"start": 0, "end": 1, "speaker": "A"}],
                      "speakers": [{"id": "A", "name": "Nam"}], "tolerance_s": 0.3, "ignore": [{"start": 5, "end": 6}]}, "s1")
    assert [s["speaker"] for s in doc["segments"]] == ["A", "B"]                 # sorted by time
    assert [sp["id"] for sp in doc["speakers"]] == ["A", "B"]                    # B was used, so it is declared
    assert doc["segments"][1]["start"] == 2.0 and doc["tolerance_s"] == 0.3 and doc["ignore"][0]["note"] == ""
    for bad in ({"segments": [{"start": 3, "end": 1, "speaker": "A"}]}, {"segments": [{"start": 0, "end": 1}]},
                {"segments": [{"start": "x", "end": 1, "speaker": "A"}]}, {"tolerance_s": 99}, [],
                {"speakers": [{"id": "A"}, {"id": "A"}]}, {"segments": [{"start": -1, "end": 1, "speaker": "A"}]}):
        with pytest.raises(ValueError):
            S.validate(bad, "s1")


def test_save_bumps_the_revision_keeps_history_and_refuses_a_stale_write(tmp_path):
    make_sample(tmp_path)
    a = {"segments": [{"start": 0, "end": 1, "speaker": "A"}]}
    r1 = S.save(str(tmp_path), "01_demo", a, base_rev=0)
    assert r1["rev"] == 1 and os.path.isfile(tmp_path / "01_demo" / "labels.json")
    r2 = S.save(str(tmp_path), "01_demo", {"segments": [{"start": 0, "end": 2, "speaker": "A"}]}, base_rev=1)
    assert r2["rev"] == 2
    hist = sorted(os.listdir(tmp_path / "01_demo" / "labels_history"))
    assert hist == ["labels_000001.json"] and json.load(open(tmp_path / "01_demo" / "labels_history" / hist[0]))["segments"][0]["end"] == 1
    with pytest.raises(S.Conflict) as err:
        S.save(str(tmp_path), "01_demo", a, base_rev=1)                           # another tab already wrote rev 2
    assert err.value.current["rev"] == 2
    assert S.save(str(tmp_path), "01_demo", a, base_rev=1, force=True)["rev"] == 3
    assert S.load(str(tmp_path), "01_demo")["segments"][0]["end"] == 1


def test_history_is_pruned(tmp_path, monkeypatch):
    make_sample(tmp_path)
    monkeypatch.setattr(S, "HISTORY_KEEP", 3)
    for i in range(8):
        S.save(str(tmp_path), "01_demo", {"segments": [{"start": 0, "end": 1 + i, "speaker": "A"}]})
    assert len(os.listdir(tmp_path / "01_demo" / "labels_history")) == 3


def test_sample_names_cannot_leave_the_folder(tmp_path):
    make_sample(tmp_path)
    for name in ("..", "../01_demo", "a/b", ""):
        with pytest.raises(ValueError):
            S.load(str(tmp_path), name)
    with pytest.raises(FileNotFoundError):
        S.load(str(tmp_path), "nope")


def test_list_samples_reports_arms_and_label_state(tmp_path):
    make_sample(tmp_path, "01_demo")
    (tmp_path / "02_noclip").mkdir()
    S.save(str(tmp_path), "01_demo", {"segments": [{"start": 0, "end": 1, "speaker": "A"}]})
    rows = S.list_samples(str(tmp_path))
    assert [r["name"] for r in rows] == ["01_demo"]
    assert rows[0]["arms"] == ["diarizen_auto"] and rows[0]["segments"] == 1 and rows[0]["rev"] == 1


@pytest.fixture
def server(tmp_path):
    arms = tmp_path / "diar_arms"
    arms.mkdir()
    make_sample(arms)
    srv = LS.LabelServer(("127.0.0.1", 0), tmp_path, arms)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", arms
    srv.shutdown()
    srv.server_close()


def call(url, method="GET", body=None):
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_serves_the_page_the_files_and_the_label_api(server):
    base, arms = server
    code, page = call(base + "/label")
    assert code == 200 and "Đánh nhãn diarization".encode() in page
    code, wav = call(base + "/diar_arms/01_demo/clip.wav")
    assert code == 200 and wav[:4] == b"RIFF"
    code, body = call(base + "/api/label/samples")
    assert json.loads(body)["samples"][0]["name"] == "01_demo"
    code, body = call(base + "/api/label/01_demo")
    assert code == 200 and json.loads(body)["rev"] == 0
    labels = {"speakers": [{"id": "S1", "name": "Nam"}], "segments": [{"start": 0.5, "end": 2.0, "speaker": "S1"}]}
    code, body = call(base + "/api/label/01_demo", "PUT", {"labels": labels, "base_rev": 0})
    assert code == 200 and json.loads(body)["rev"] == 1
    code, body = call(base + "/api/label/01_demo", "PUT", {"labels": labels, "base_rev": 0})
    assert code == 409 and json.loads(body)["current"]["rev"] == 1                  # a second tab with the old revision
    code, body = call(base + "/api/label/01_demo", "PUT", {"labels": {"segments": [{"start": 2, "end": 1, "speaker": "S1"}]}})
    assert code == 400
    assert call(base + "/api/label/nope")[0] == 404 and call(base + "/api/label/..%2F..")[0] in (400, 404)
    assert json.loads((arms / "01_demo" / "labels.json").read_text())["segments"][0]["end"] == 2.0
