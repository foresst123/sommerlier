import json
import os
import pickle
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import make_mix_dataset as mmd  # noqa: E402

sf = pytest.importorskip("soundfile")
SR = 24000


def _tone(seconds, freq, amp=0.2):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype("float32")


def test_looks_like_speech_rejects_silence_and_clipping():
    assert mmd.looks_like_speech(_tone(3, 200), SR)
    assert not mmd.looks_like_speech(np.zeros(3 * SR, "float32"), SR)
    assert not mmd.looks_like_speech(np.ones(3 * SR, "float32"), SR)
    half = np.concatenate([_tone(1, 200), np.zeros(3 * SR, "float32")])
    assert not mmd.looks_like_speech(half, SR)


@pytest.mark.parametrize("seed", range(20))
def test_mix_is_exact_sum_with_requested_level_and_real_overlap(seed):
    rng = random.Random(seed)
    a, b = _tone(4, 200), _tone(3, 330)
    placed = mmd.place(rng, len(a), len(b), SR)
    A, B, M, ov = mmd.mix(a, b, placed, db_b=-6.0)
    assert len(A) == len(B) == len(M)
    assert np.allclose(A + B, M, atol=1e-6)
    assert np.max(np.abs(M)) <= 0.9501
    assert ov >= int(0.1 * SR) - 1
    both = (np.abs(A) > 0) & (np.abs(B) > 0)
    assert both.sum() >= ov * 0.9
    ra, rb = mmd.dbfs(A[np.abs(A) > 0]), mmd.dbfs(B[np.abs(B) > 0])
    assert rb - ra == pytest.approx(-6.0, abs=0.6)


def _cache(tmp_path, name, speakers=("A", "B")):
    from schemas.segment import SpeechSegment
    job = tmp_path / "cache" / name
    (job / "separation").mkdir(parents=True)
    (job / "processed_audio" / "ns").mkdir(parents=True)
    wave, rows, t = [], [], 0.0
    for i in range(8):
        spk = speakers[i % 2]
        wave.append(_tone(6, 180 + 90 * (i % 2)))
        rows.append(SpeechSegment(index=str(i), start=t, end=t + 6.0, speaker=spk))
        t += 7.0
        wave.append(np.zeros(SR, "float32"))
    np.save(str(job / "processed_audio" / "ns" / "result.npy"), np.concatenate(wave))
    (job / "separation" / "result.pkl").write_bytes(pickle.dumps(rows))


def test_build_writes_aligned_stems_and_splits_by_recording(tmp_path):
    for n in ("default_r1", "default_r2", "default_r3"):
        _cache(tmp_path, n)
    import collect_overlap_samples as cos
    recs = cos.cache_recordings(str(tmp_path / "cache"))
    rows = mmd.build(recs, str(tmp_path / "out"), n=12, seed=1, val_share=0.34)
    assert len(rows) == 12
    d = tmp_path / "out" / rows[0]["id"]
    A, sr = sf.read(str(d / "clean_A.wav"))
    B, _ = sf.read(str(d / "clean_B.wav"))
    M, _ = sf.read(str(d / "mixture.wav"))
    assert sr == SR and np.allclose(A + B, M, atol=1e-6)
    by_rec = {}
    for r in rows:
        by_rec.setdefault(r["recording"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_rec.values())
    assert {r["split"] for r in rows} == {"train", "val"}
    assert all(r["A"]["speaker"] != r["B"]["speaker"] for r in rows)
    lines = (tmp_path / "out" / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == 12 and json.loads(lines[0])["split"] in ("train", "val")
