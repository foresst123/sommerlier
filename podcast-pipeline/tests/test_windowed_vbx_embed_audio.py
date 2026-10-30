import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import windowed_vbx_stage as V  # noqa: E402


def test_without_embed_audio_the_embedder_gets_the_audio_itself():
    audio = np.arange(32000, dtype=np.float32)
    assert V.embedder_waveform(audio, 16000, None, None) is audio


def test_with_embed_audio_the_embedder_cuts_its_windows_from_that_file():
    audio = np.zeros(2 * 16000, dtype=np.float32)
    vocals = np.full(2 * 16000, 0.5, dtype=np.float32)
    chosen = V.embedder_waveform(audio, 16000, vocals, 16000)
    seen = []

    def fake_embedder(clip, mask):             # records the waveform it was given
        seen.append(np.asarray(clip))
        return np.ones(4)

    # the stage calls the embedder once per window span of a piece that starts at 0.5 s
    fake_embedder(V.crop_for_embedder(chosen, 0.5, (0.0, 1.0)), np.ones(16000))
    fake_embedder(V.crop_for_embedder(chosen, 0.5, (0.25, 1.25)), np.ones(16000))
    assert len(seen) == 2 and all(len(c) == 16000 for c in seen)
    assert all(np.all(c == 0.5) for c in seen)                   # the vocals, never the zeros of --audio


def test_crop_matches_the_slice_the_stage_used_before():
    wave = np.arange(3 * 16000, dtype=np.float32)
    assert np.array_equal(V.crop_for_embedder(wave, 1.0, (0.5, 1.5)), wave[24000:40000])


def test_embed_audio_of_another_length_is_refused():
    with pytest.raises(ValueError, match="same length"):
        V.embedder_waveform(np.zeros(32000), 16000, np.zeros(31999), 16000)


def test_embed_audio_of_another_rate_is_refused():
    with pytest.raises(ValueError, match="must match"):
        V.embedder_waveform(np.zeros(32000), 16000, np.zeros(32000), 8000)
