"""MP3 and separated-wav export fan out over threads; results stay identical."""

import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.export_service import ExportService


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, *_, **__):
        pass

    def warning(self, message, **_):
        self.warnings.append(message)


class _FakeMp3Segment:
    """Records how many exports were in flight at once and on which threads."""

    def __init__(self, stats, fail_on=None):
        self.stats, self.fail_on = stats, fail_on

    def __getitem__(self, window):
        self.window = window
        return self

    def export(self, path, format=None):
        stats = self.stats
        with stats["lock"]:
            stats["active"] += 1
            stats["peak"] = max(stats["peak"], stats["active"])
            stats["threads"].add(threading.current_thread().name)
        try:
            time.sleep(0.05)
            if self.fail_on and self.fail_on in path:
                raise RuntimeError("ffmpeg failed")
            with open(path, "wb") as handle:
                handle.write(b"mp3")
        finally:
            with stats["lock"]:
                stats["active"] -= 1


def _stats():
    return {"lock": threading.Lock(), "active": 0, "peak": 0, "threads": set()}


def _segments(count):
    return [SimpleNamespace(index=f"{i:05d}", speaker="SPEAKER_00",
                            start=float(i), end=float(i) + 0.5)
            for i in range(count)]


def _audio(stats, **kwargs):
    return SimpleNamespace(audio_segment=_FakeMp3Segment(stats, **kwargs),
                           waveform=None, sample_rate=16000)


def test_mp3_segments_are_encoded_concurrently(tmp_path):
    stats = _stats()
    segments = _segments(12)
    ExportService(workers=4).export_mp3_segments(segments, _audio(stats), str(tmp_path), "clip")

    assert stats["peak"] >= 2
    for seg in segments:
        assert (tmp_path / "clip" / f"{seg.index}_{seg.speaker}.mp3").read_bytes() == b"mp3"


def test_a_single_worker_encodes_one_at_a_time(tmp_path):
    stats = _stats()
    ExportService(workers=1).export_mp3_segments(
        _segments(6), _audio(stats), str(tmp_path), "clip")
    assert stats["peak"] == 1


def test_an_mp3_failure_still_fails_the_export(tmp_path):
    stats = _stats()
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        ExportService(workers=4).export_mp3_segments(
            _segments(6), _audio(stats, fail_on="00003_"), str(tmp_path), "clip")


def _speech_segments():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        audio = rng.uniform(-0.1, 0.1, 800).astype(np.float32)
        rows.append(SimpleNamespace(index=f"{i:05d}", speaker=f"SPEAKER_0{i % 2}",
                                    start=i * 0.05, end=i * 0.05 + 0.05, audio=audio))
    return rows


def test_separated_wavs_are_written_and_the_stitched_audio_is_unchanged(tmp_path):
    import soundfile as sf
    rows = _speech_segments()
    ExportService(workers=4).export_separated_audio(rows, 16000, str(tmp_path))

    for row in rows:
        chunk, rate = sf.read(str(tmp_path / "separation" / f"{row.index}_{row.speaker}_separated.wav"))
        assert rate == 16000 and len(chunk) == len(row.audio)

    expected = np.zeros(int(rows[-1].end * 16000), dtype=np.float32)
    for row in rows:
        start = int(row.start * 16000)
        expected[start:start + len(row.audio)] += row.audio
    stitched, _ = sf.read(str(tmp_path / "after_separation.wav"), dtype="float32")
    assert len(stitched) == len(expected)
    assert np.allclose(stitched, expected, atol=1e-3)     # PCM_16 rounding


def test_a_failed_chunk_write_is_logged_and_does_not_stop_the_others(tmp_path, monkeypatch):
    import soundfile as sf
    real_write = sf.write

    def flaky_write(path, *args, **kwargs):
        if "00002_" in str(path):
            raise OSError("disk full")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(sf, "write", flaky_write)
    logger = _Logger()
    ExportService(logger=logger, workers=4).export_separated_audio(
        _speech_segments(), 16000, str(tmp_path))

    written = sorted(os.listdir(tmp_path / "separation"))
    assert len(written) == 5 and not any(name.startswith("00002_") for name in written)
    assert any("00002" in message and "disk full" in message for message in logger.warnings)
    assert (tmp_path / "after_separation.wav").exists()
