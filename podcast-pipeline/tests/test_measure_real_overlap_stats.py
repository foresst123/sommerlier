import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import measure_real_overlap_stats as m  # noqa: E402

SR = 24000


def test_classify_patterns():
    assert m.classify((0, 10), (0.1, 9.9)) == "full"
    assert m.classify((0, 10), (3, 5)) == "contained"
    assert m.classify((0, 10), (8, 15)) == "partial"
    assert m.classify((0, 10), (8, 10.1)) == "contained"


def test_speech_level_ignores_silence_and_tracks_amplitude():
    rng = np.random.default_rng(0)
    loud = (0.2 * rng.standard_normal(3 * SR)).astype("float32")
    half = (0.1 * rng.standard_normal(3 * SR)).astype("float32")
    pad = np.zeros(3 * SR, "float32")
    a = m.speech_level_db(np.concatenate([loud, pad]), SR)
    b = m.speech_level_db(np.concatenate([half, pad]), SR)
    assert a - b == pytest.approx(6.02, abs=0.3)
    assert m.speech_level_db(np.zeros(SR, "float32"), SR) is None


def _rec(amp_a=0.2, amp_b=0.1, noise_amp=0.001):
    """A talks 0-30, B 20-50 (partial overlap 20-30); gaps 50-70 and 80-100 hold faint noise."""
    rng = np.random.default_rng(1)
    audio = (noise_amp * rng.standard_normal(120 * SR)).astype("float32")
    audio[0:30 * SR] += amp_a * rng.standard_normal(30 * SR).astype("float32")
    audio[20 * SR:50 * SR] += amp_b * rng.standard_normal(30 * SR).astype("float32")
    audio[70 * SR:80 * SR] += amp_a * rng.standard_normal(10 * SR).astype("float32")
    segs = [{"index": "0", "start": 0.0, "end": 30.0, "speaker": "A"},
            {"index": "1", "start": 20.0, "end": 50.0, "speaker": "B"},
            {"index": "2", "start": 70.0, "end": 80.0, "speaker": "A"}]
    return {"path": "cache/default_fake", "segments": lambda: segs, "audio": lambda: (audio, SR)}


def test_measure_recovers_pattern_level_and_noise():
    r = m.measure_recording(_rec(), merge_gap=1.0, min_overlap=0.1)
    assert len(r["pairs"]) == 1 and r["pairs"][0]["pattern"] == "partial"
    assert r["pairs"][0]["overlap"] == pytest.approx(10.0)
    # B (later) is half the amplitude of A: about -6 dB
    assert r["pairs"][0]["delta_db"] == pytest.approx(-6.0, abs=1.0)
    assert len(r["events"]) == 1 and r["events"][0]["duration"] == pytest.approx(10.0)
    assert r["noise_db"] is None or r["noise_db"] < -50   # faint noise (needs >=5 gaps, only 2 here)


def test_summary_and_outputs(tmp_path):
    results = [m.measure_recording(_rec(noise_amp=0.001 * (i + 1)), 1.0, 0.1) for i in range(5)]
    for i, r in enumerate(results):
        r["name"] = f"rec{i}"
    S = m.summarize(results, max_per_file=50, seed=0, boot=50)
    assert S["recordings"] == 5 and S["events_total"] == 5
    assert S["pattern"]["partial"]["share"] == 1.0
    assert S["event_duration_s"]["p50"] == pytest.approx(10.0)
    assert S["later_minus_earlier_speaker_db"]["p50"] == pytest.approx(-6.0, abs=1.0)
    m.write_outputs(results, S, str(tmp_path))
    for f in ("events.csv", "pairs.csv", "recordings.csv", "summary.json", "summary.md"):
        assert (tmp_path / f).exists()
    assert "pattern partial" in (tmp_path / "summary.md").read_text()
