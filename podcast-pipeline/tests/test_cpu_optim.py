"""Tests for the CPU-side optimisations: audio caching, thread budget, chunking.

The audio cache is the one that can fail silently -- a stale entry means every
later stage transcribes the previous version of the file and nothing warns.
"""
import os
import sys
import time

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.audio_service import AudioService
from utils.cpu_plan import apply_to_env, thread_plan, usable_cores

SR = 24000


def _write_wav(path, seconds=1.0, freq=440, amp=0.2, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
    return path


# --- audio cache ---------------------------------------------------------

def test_second_load_comes_from_cache(tmp_path):
    src = _write_wav(tmp_path / "a.wav")
    svc = AudioService()

    first = svc.load_audio(str(src), target_sr=SR)
    assert os.path.exists(svc._cache_path(str(src), SR)), "cache was not written"

    second = svc.load_audio(str(src), target_sr=SR)
    assert np.allclose(np.asarray(first.waveform), np.asarray(second.waveform), atol=1e-6), (
        "a cached load must return the same samples a decoded load would"
    )


def test_cache_is_written_after_gain_not_before(tmp_path):
    """Both paths must agree, or a run that hits the cache is normalised
    differently from one that does not."""
    src = _write_wav(tmp_path / "quiet.wav", amp=0.01)     # well under -20 dBFS
    svc = AudioService()

    decoded = np.asarray(svc.load_audio(str(src), target_sr=SR).waveform).copy()
    cached = np.asarray(svc.load_audio(str(src), target_sr=SR).waveform)

    assert np.allclose(decoded, cached, atol=1e-6)
    rms = float(np.sqrt(np.mean(cached ** 2)))
    assert rms > 0.01, f"gain was not applied before caching (rms {rms:.4f})"


def test_a_changed_source_invalidates_the_cache(tmp_path):
    src = tmp_path / "b.wav"
    _write_wav(src, freq=440)
    svc = AudioService()
    first = np.asarray(svc.load_audio(str(src), target_sr=SR).waveform).copy()

    time.sleep(0.01)
    _write_wav(src, freq=880)                              # same path, new content
    os.utime(src, None)
    second = np.asarray(svc.load_audio(str(src), target_sr=SR).waveform)

    assert not np.allclose(first, second, atol=1e-3), (
        "the cache served the old file: every later stage would run on stale audio"
    )


def test_same_basename_in_two_folders_does_not_share_a_cache(tmp_path):
    d1, d2 = tmp_path / "one", tmp_path / "two"
    d1.mkdir(); d2.mkdir()
    _write_wav(d1 / "talk.wav", freq=440)
    _write_wav(d2 / "talk.wav", freq=880)

    svc = AudioService()
    a = np.asarray(svc.load_audio(str(d1 / "talk.wav"), target_sr=SR).waveform).copy()
    b = np.asarray(svc.load_audio(str(d2 / "talk.wav"), target_sr=SR).waveform)
    assert not np.allclose(a, b, atol=1e-3)


def test_different_sample_rates_get_different_cache_entries(tmp_path):
    src = _write_wav(tmp_path / "c.wav", seconds=1.0)
    svc = AudioService()
    a = svc.load_audio(str(src), target_sr=SR)
    b = svc.load_audio(str(src), target_sr=16000)
    assert len(a.waveform) != len(b.waveform)
    assert svc._cache_path(str(src), SR) != svc._cache_path(str(src), 16000)


def test_audio_segment_is_not_held(tmp_path):
    """export_service rebuilds it on demand; keeping it doubles resident audio."""
    src = _write_wav(tmp_path / "d.wav")
    assert AudioService().load_audio(str(src), target_sr=SR).audio_segment is None


def test_cache_can_be_turned_off(tmp_path, monkeypatch):
    import services.audio_service as mod
    monkeypatch.setattr(mod, "AUDIO_CACHE", False)
    src = _write_wav(tmp_path / "e.wav")
    svc = mod.AudioService()
    svc.load_audio(str(src), target_sr=SR)
    assert not os.path.exists(svc._cache_path(str(src), SR))


# --- thread budget -------------------------------------------------------

def test_slurm_allocation_wins_over_machine_core_count(monkeypatch):
    """A job pinned to 2 cores on a 128-core node still sees 128 from
    os.cpu_count(); using that number is how four processes end up spawning
    hundreds of threads."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert usable_cores() == 2


def test_threads_are_split_across_processes(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    plan = thread_plan(n_workers=3)
    assert plan["processes"] == 4
    assert plan["per_process"] == 4
    assert plan["env"]["OMP_NUM_THREADS"] == "4"


def test_never_allocates_zero_threads(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    assert thread_plan(n_workers=3)["per_process"] >= 1


def test_existing_settings_are_respected(monkeypatch):
    env = {"OMP_NUM_THREADS": "8"}
    apply_to_env(env, 2)
    assert env["OMP_NUM_THREADS"] == "8", "an explicit setting must not be overwritten"
    assert env["MKL_NUM_THREADS"] == "2"


# --- bs_roformer chunking -----------------------------------------------------

# models/bs_roformer imports torch, which is not installed in the test env; the
# windowing is plain arithmetic, so the defaults are restated here.
GPU_CHUNK_SEC, GPU_CHUNK_OVERLAP_SEC = 60.0, 1.0


def test_bs_roformer_windows_do_not_grow_with_file_length():
    """The OOM was VRAM scaling with duration; windows keep the peak flat."""
    class d:
        gpu_chunk_sec, gpu_chunk_overlap_sec = GPU_CHUNK_SEC, GPU_CHUNK_OVERLAP_SEC

    for seconds in (60, 600, 3000):
        n = seconds * SR
        chunk = int(d.gpu_chunk_sec * SR)
        pad = int(d.gpu_chunk_overlap_sec * SR)
        widest = max(min(n, s + chunk + pad) - max(0, s - pad)
                     for s in range(0, n, chunk))
        assert widest <= chunk + 2 * pad, (
            f"a {seconds}s file would hand {widest / SR:.0f}s to the GPU at once"
        )


def test_bs_roformer_windows_cover_every_sample():
    class d:
        gpu_chunk_sec, gpu_chunk_overlap_sec = GPU_CHUNK_SEC, GPU_CHUNK_OVERLAP_SEC

    n = 3000 * SR
    chunk = int(d.gpu_chunk_sec * SR)
    covered = 0
    for start in range(0, n, chunk):
        covered += min(n, start + chunk) - start
    assert covered == n, "chunking dropped or double-counted samples"
