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


# --- bs_roformer peak memory --------------------------------------------------

# The OOM these replace was memory scaling with file duration, fixed at the
# time by chunking inside the Demucs wrapper. audio-separator has that same
# control built in as `chunk_duration`, and leaves it off by default: without
# it the overlap-add result and counter buffers are allocated for the whole
# track. On CUDA `should_accumulate_on_device` puts those on the host, so an
# unbounded run is a RAM OOM on a Kaggle box rather than a VRAM one -- which is
# why it did not look like the same bug the second time.


def _bs_roformer_profiles():
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    return {name: profile["models"]["bs_roformer"]
            for name, profile in config["environments"].items()}


# 2 instruments x 2 channels x 4 bytes, for the result and the counter both.
BUFFER_BYTES_PER_SECOND = 2 * 2 * 44100 * 4 * 2


def test_every_profile_bounds_its_own_buffers():
    for name, block in _bs_roformer_profiles().items():
        assert block.get("chunk_duration"), name


def test_the_bound_leaves_room_on_the_box_it_names():
    """A 50-minute recording unbounded is ~4.2GB of float32 buffers. These are
    the numbers that make that a fixed cost instead."""
    budgets = {"kaggle": 1.0, "a100": 2.0}          # GiB the stage may hold
    for name, block in _bs_roformer_profiles().items():
        if name not in budgets:
            continue
        gib = block["chunk_duration"] * BUFFER_BYTES_PER_SECOND / 1024 ** 3
        assert gib <= budgets[name], f"{name}: {gib:.2f}GiB > {budgets[name]}GiB"


def test_the_smaller_box_does_not_ask_for_more_than_the_bigger_one():
    """kaggle shares its RAM with DiariZen, the embedder, TSE and ASR."""
    profiles = _bs_roformer_profiles()
    if {"kaggle", "a100"} <= set(profiles):
        assert profiles["kaggle"]["chunk_duration"] <= profiles["a100"]["chunk_duration"]


def test_a_segment_size_is_only_honoured_when_the_override_is_asked_for():
    """audio-separator ignores segment_size unless override_model_segment_size
    is set, so a profile naming one without the flag is a knob that looks live
    and is not -- exactly the failure the Demucs block was."""
    for name, block in _bs_roformer_profiles().items():
        if "segment_size" in block:
            assert block.get("override_model_segment_size") is True, name
