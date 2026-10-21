import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.compare_lib import learned_mos as L
from tools.compare_lib import metrics as M
from tools.compare_lib.build import _summary

SR = 24000
ENV = ("UNIVERSA_REPO", "UNIVERSA_MODEL", "UNIVERSA_CONFIG")


def wav(path, amp, sec=1.0):
    t = np.arange(int(SR * sec)) / SR
    sf.write(str(path), (amp * np.sin(2 * np.pi * 220 * t)).astype("float32"), SR)
    return str(path)


def test_none_when_unconfigured(monkeypatch, tmp_path):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    assert L.score_wav(np.zeros(16000, "float32"), 16000) is None
    assert L.score_wav(wav(tmp_path / "a.wav", 0.1), None) is None
    monkeypatch.setenv("UNIVERSA_MODEL", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("UNIVERSA_CONFIG", str(tmp_path / "missing.yaml"))
    assert L.score_wav(np.zeros(16000, "float32"), 16000) is None


def test_none_when_deps_missing(monkeypatch, tmp_path):
    (tmp_path / "m.pt").write_bytes(b"x")
    (tmp_path / "c.yaml").write_text("a: 1")
    cfg = {"model": str(tmp_path / "m.pt"), "config": str(tmp_path / "c.yaml"),
           "repo": str(tmp_path / "no_repo")}
    monkeypatch.setitem(sys.modules, "urgent2026_sqa", None)  # import raises
    monkeypatch.setitem(sys.modules, "urgent2026_sqa.infer", None)
    assert L.load_predictor(cfg) is None
    assert L.score_wav(np.ones(16000, "float32"), 16000, cfg) is None


def test_import_does_not_pull_torch():
    import subprocess
    code = ("import sys; sys.path.insert(0, '.'); import tools.compare_lib.metrics; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=os.path.join(os.path.dirname(__file__), ".."))
    assert r.returncode == 0, r.stderr


def test_chunk_weighting_with_fake_model():
    calls = []

    def predict(x):
        calls.append(len(x))
        return {k: float(len(x)) for k in L.KEYS}  # score == chunk length

    x = np.zeros(int(16000 * 45), "float32")  # 45 s at 20 s chunks -> 2 chunks
    res = L.score_chunks(x, predict, chunk_s=20.0)
    assert len(calls) == 2 and sum(calls) == len(x)
    want = sum(c * c for c in calls) / len(x)  # length-weighted mean
    assert res["mos"] == pytest.approx(want) and set(res) == set(L.KEYS)
    assert L.score_chunks(np.zeros(0, "float32"), predict) is None
    assert L.score_chunks(x[:100], lambda c: {k: float("nan") for k in L.KEYS}) is None


def test_score_wav_resamples_and_uses_predictor(monkeypatch, tmp_path):
    seen = []

    def predict(x):
        seen.append(len(x))
        return {k: 3.0 for k in L.KEYS}

    monkeypatch.setattr(L, "load_predictor", lambda cfg=None: predict)
    res = L.score_wav(wav(tmp_path / "a.wav", 0.1, 2.0), None)
    assert res == {k: 3.0 for k in L.KEYS} and seen == [32000]  # 24 kHz -> 16 kHz
    res = L.score_wav(np.zeros(24000, "float32"), 24000)
    assert seen[-1] == 16000 and res["utmos"] == 3.0


def fake_scores(monkeypatch, per_amp):
    def score(path, sr=None, cfg=None):
        x, _ = sf.read(path)
        return per_amp(float(np.abs(x).max()))
    monkeypatch.setattr(L, "score_wav", score)
    M._RAW_MOS.clear()


def test_audio_metrics_delta(monkeypatch, tmp_path):
    raw, var = wav(tmp_path / "raw.wav", 0.5), wav(tmp_path / "var.wav", 0.25)
    fake_scores(monkeypatch, lambda a: {k: (2.0 if a > 0.4 else 3.5) for k in L.KEYS})
    m = M.audio_metrics(raw, var)
    assert m["learned_mos"]["mos"] == 3.5
    assert m["learned_mos_delta"] == {k: pytest.approx(1.5) for k in L.KEYS}
    b = M.audio_metrics(raw, raw, same_as_raw=True)
    assert b["learned_mos"]["utmos"] == 2.0
    assert b["learned_mos_delta"] == {k: 0.0 for k in L.KEYS}
    off = M.audio_metrics(raw, var, embed=False)
    assert off["learned_mos"] is None and off["learned_mos_delta"] is None


def test_audio_metrics_null_when_unavailable(monkeypatch, tmp_path):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    M._RAW_MOS.clear()
    raw, var = wav(tmp_path / "raw.wav", 0.5), wav(tmp_path / "var.wav", 0.25)
    m = M.audio_metrics(raw, var)
    assert "learned_mos" in m and "learned_mos_delta" in m
    assert m["learned_mos"] is None and m["learned_mos_delta"] is None
    assert M.audio_metrics(None, None)["learned_mos_delta"] is None


def test_summary_flattens_nested_means():
    mets = {"f1": {"v": {"audio": {"learned_mos_delta": {"mos": 1.0, "utmos": None},
                                   "learned_mos": None, "ecapa_cos": 0.9}}},
            "f2": {"v": {"audio": {"learned_mos_delta": {"mos": 3.0, "utmos": None},
                                   "learned_mos": None, "ecapa_cos": 0.7}}}}
    s = _summary([{"id": "v"}], mets)["v"]
    assert s["audio.learned_mos_delta.mos"] == pytest.approx(2.0)
    assert s["audio.ecapa_cos"] == pytest.approx(0.8)
    assert "audio.learned_mos_delta.utmos" not in s and "audio.learned_mos" not in s
