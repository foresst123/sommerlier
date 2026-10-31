import os
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from tools import eval_verification as V  # noqa: E402
from tools import train_plda as T  # noqa: E402

FREQS = {"spk1": 300.0, "spk2": 600.0, "spk3": 900.0}


def write_wav(path, freq, seconds=1.0, amp=0.3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(int(16000 * seconds)) / 16000
    pcm = (amp * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(pcm.tobytes())


class SpeakerAdapter:
    """One-hot on the speaker whose tone is the loudest peak of the wave, plus a little offset per amplitude."""

    def load(self):
        return self

    def embed(self, wave_):
        spec = np.abs(np.fft.rfft(wave_))
        freq = np.fft.rfftfreq(len(wave_), 1 / 16000)[int(np.argmax(spec))]
        spk = min(FREQS, key=lambda s: abs(FREQS[s] - freq))
        vec = np.zeros(len(FREQS))
        vec[list(FREQS).index(spk)] = 1.0
        vec[0] += 0.01 * float(np.std(wave_))
        return vec


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "wav"
    files = {}
    for spk, f in FREQS.items():
        for k in range(3):
            rel = f"{spk}/vid/u{k}.wav"
            write_wav(str(root / rel), f, amp=0.2 + 0.05 * k)
            files.setdefault(spk, []).append(rel)
    trials = tmp_path / "trials.txt"
    lines = []
    for spk, rels in files.items():
        lines.append(f"1 {rels[0]} {rels[1]}")
    spks = list(files)
    for i in range(len(spks)):
        for j in range(i + 1, len(spks)):
            lines.append(f"0 {files[spks[i]][0]} {files[spks[j]][0]}")
    trials.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root, trials


def test_read_trials_parses_and_rejects_bad_labels(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("1 a/b.wav c/d.wav\n0 a/b.wav e/f.wav\n", encoding="utf-8")
    assert V.read_trials(str(good)) == [(1, "a/b.wav", "c/d.wav"), (0, "a/b.wav", "e/f.wav")]
    bad = tmp_path / "bad.txt"
    bad.write_text("2 a b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        V.read_trials(str(bad))


def test_cosine_eer_is_zero_for_separable_speakers(corpus):
    root, trials_path = corpus
    trials = V.read_trials(str(trials_path))
    files = sorted({p for _, a, b in trials for p in (a, b)})
    vecs, failed = V.embed_files(files, str(root), SpeakerAdapter())
    assert failed == []
    res = V.evaluate(trials, vecs, None)
    assert res["eer_cosine"] == pytest.approx(0.0)
    assert res["eer_plda"] is None
    assert res["targets"] == 3 and res["nontargets"] == 3


def test_missing_files_are_counted_not_silently_dropped(corpus):
    root, trials_path = corpus
    trials = V.read_trials(str(trials_path)) + [(1, "spk1/vid/missing.wav", "spk1/vid/u0.wav")]
    files = sorted({p for _, a, b in trials for p in (a, b)})
    vecs, failed = V.embed_files(files, str(root), SpeakerAdapter())
    assert "spk1/vid/missing.wav" in failed
    res = V.evaluate(trials, vecs, None)
    assert res["dropped"] == 1 and res["trials"] == 7


def test_plda_scores_with_a_model_trained_on_the_same_embedder(tmp_path):
    rng = np.random.default_rng(0)
    centres = np.eye(6) * 3.0
    X = np.vstack([centres[s] + 0.3 * rng.standard_normal((12, 6)) for s in range(6)])
    y = np.repeat(np.arange(6), 12)
    model = T.fit_all(X, y, 3)
    plda_dir = tmp_path / "plda"
    T.write_model(str(plda_dir), model, {"embedder_id": "toy"})

    loaded = V.load_plda(str(plda_dir))
    assert loaded["mean1"].shape[0] == 6
    vecs = {f"u{i}": X[i] for i in range(len(X))}
    trials = [(1, f"u{i}", f"u{i + 1}") for i in range(0, 11)] + [(0, f"u{i}", f"u{i + 12}") for i in range(0, 12)]
    res = V.evaluate(trials, vecs, loaded)
    assert res["eer_plda"] is not None and 0.0 <= res["eer_plda"] <= 1.0
    assert res["eer_cosine"] is not None


def test_plda_dimension_mismatch_is_refused(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 6))
    y = np.repeat(np.arange(3), 10)
    model = T.fit_all(X, y, 2)
    plda_dir = tmp_path / "plda"
    T.write_model(str(plda_dir), model, {})
    loaded = V.load_plda(str(plda_dir))
    vecs = {"a": rng.standard_normal(4), "b": rng.standard_normal(4)}
    with pytest.raises(ValueError):
        V.score_trials([(1, "a", "b")], vecs, loaded)
