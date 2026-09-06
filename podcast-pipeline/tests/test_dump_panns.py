"""Writing PANNs' raw output out so the thresholds can stop being guesses.

Six groups out of 527 labels, four thresholds, none of them chosen against a
measured distribution. These pin the two things that would make the dump
useless: losing information while claiming not to, and describing a signal the
pipeline never actually routed on.

Run:  python -m pytest tests/test_dump_panns.py -q   (from podcast-pipeline/)
"""
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.dump_panns import _write_top_csv, block_reduce, derive_flags

BLOCK = 32
LABELS = 527


def _repeated(blocks, labels=LABELS, block=BLOCK):
    """What the model actually emits: one decision, repeated across the block."""
    decisions = np.random.default_rng(0).random((blocks, labels)).astype(np.float32)
    return np.repeat(decisions, block, axis=0), decisions


# --- the 32x reduction has to be lossless, not approximately lossless --------

def test_a_repeated_block_reduces_to_one_row_and_loses_nothing():
    framewise, decisions = _repeated(10)
    reduced, spread = block_reduce(framewise, BLOCK)
    assert reduced.shape == (10, LABELS)
    assert spread == 0.0, "a genuinely constant block must report zero spread"
    assert np.array_equal(reduced, decisions)


def test_a_block_that_is_not_constant_is_reported_not_hidden():
    """If panns_inference ever stops repeating, the dump must say so rather
    than silently keep the first frame of each block."""
    framewise, _ = _repeated(4)
    framewise[5, 3] += 0.25
    _, spread = block_reduce(framewise, BLOCK)
    assert spread == pytest.approx(0.25, abs=1e-6)


def test_the_last_partial_block_is_kept():
    """It is real audio at the end of the recording, not padding."""
    framewise, _ = _repeated(3)
    framewise = np.concatenate([framewise, framewise[:7]], axis=0)
    reduced, _ = block_reduce(framewise, BLOCK)
    assert len(reduced) == 4


def test_audio_shorter_than_one_block_still_produces_a_row():
    framewise, _ = _repeated(1)
    reduced, _ = block_reduce(framewise[:9], BLOCK)
    assert len(reduced) == 1


def test_an_empty_matrix_reduces_to_an_empty_matrix():
    reduced, spread = block_reduce(np.zeros((0, LABELS), np.float32), BLOCK)
    assert len(reduced) == 0 and spread == 0.0


def test_float16_resolves_far_finer_than_any_threshold_here():
    """The default dtype must not blur the numbers a threshold sits on. The
    loosest threshold in the pipeline is music at 0.10."""
    values = np.array([0.10, 0.20, 0.35, 0.5001], dtype=np.float32)
    assert np.allclose(values.astype(np.float16).astype(np.float32), values, atol=5e-4)


# --- the decisions in the dump must be the pipeline's, not a copy of them ----

def test_the_routing_flags_match_music_map_exactly():
    """Reimplementing the rule here and getting it subtly different would make
    every analysis of the dump wrong in a way nothing would catch."""
    from utils import music_map as mm

    rng = np.random.default_rng(1)
    scores = {k: rng.random(400).astype(np.float32)
              for k in ("speech", "singing", "music")}
    config = {"music_threshold": mm.MUSIC_THRESHOLD,
              "singing_threshold": mm.SINGING_THRESHOLD,
              "singing_margin": mm.SINGING_MARGIN,
              "speech_present": mm.SPEECH_PRESENT}
    flags = derive_flags(scores, config)

    speech, singing, music = scores["speech"], scores["singing"], scores["music"]
    expect_singing = ((singing >= mm.SINGING_THRESHOLD)
                      & (singing >= speech + mm.SINGING_MARGIN))
    expect_loud = (music >= mm.MUSIC_THRESHOLD) & ~expect_singing
    assert np.array_equal(flags["is_singing"], expect_singing)
    assert np.array_equal(flags["is_song"], expect_loud & (speech < mm.SPEECH_PRESENT))
    assert np.array_equal(flags["is_music"], expect_loud & ~(speech < mm.SPEECH_PRESENT))


def test_the_three_decisions_never_overlap_on_one_frame():
    """A frame is cut, stripped, or left alone -- not two of those."""
    from utils import music_map as mm
    rng = np.random.default_rng(2)
    scores = {k: rng.random(2000).astype(np.float32)
              for k in ("speech", "singing", "music")}
    flags = derive_flags(scores, {"music_threshold": mm.MUSIC_THRESHOLD,
                                  "singing_threshold": mm.SINGING_THRESHOLD,
                                  "singing_margin": mm.SINGING_MARGIN,
                                  "speech_present": mm.SPEECH_PRESENT})
    stacked = np.stack([flags["is_singing"], flags["is_song"], flags["is_music"]])
    assert stacked.sum(axis=0).max() <= 1


def test_the_dump_reads_the_same_preprocessing_the_pipeline_routes_on():
    """tag_framewise and the dump must share one path; two paths drift, and
    then the analysis describes a signal that was never routed on."""
    import inspect
    from models.panns import PANNSDetector
    source = inspect.getsource(PANNSDetector.tag_framewise)
    assert "self.framewise_raw(" in source
    dump = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tools", "dump_panns.py"), encoding="utf-8").read()
    assert "detector.framewise_raw(" in dump


def test_the_scale_that_was_applied_is_carried_out_of_the_detector():
    """Normalising to a fixed peak over the whole file means one loud
    transient lowers every score in it. A threshold set on these numbers is
    meaningless without knowing what they are relative to."""
    import inspect
    from models.panns import PANNSDetector
    source = inspect.getsource(PANNSDetector.framewise_raw)
    assert "return self._sed_framewise(audio), self.SED_FPS, scale" in source


# --- the readable slice ------------------------------------------------------

def test_the_csv_names_the_strongest_labels_with_their_times():
    matrix = np.zeros((2, 5), dtype=np.float32)
    matrix[0, 3] = 0.9
    matrix[0, 1] = 0.5
    matrix[1, 0] = 0.7
    labels = ["Speech", "Music", "Dog", "Singing", "Wind"]

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_top_tmp.csv")
    try:
        _write_top_csv(path, matrix, labels, 0.32, top=2)
        rows = list(csv.reader(open(path, encoding="utf-8")))
    finally:
        if os.path.exists(path):
            os.remove(path)

    assert rows[0][:2] == ["start_s", "end_s"]
    assert rows[1][:4] == ["0.0", "0.32", "Singing", "0.9"]
    assert rows[2][:4] == ["0.32", "0.64", "Speech", "0.7"]


# --- neither 312MB checkpoint may load unless it is actually used ------------

def test_the_label_list_costs_no_checkpoint():
    """Reading `.labels` off a loaded model is what made the clip tagger load
    on every run -- 312MB of download and VRAM for a list of names that
    panns_inference already holds."""
    import inspect
    from models.panns import PANNSDetector
    source = inspect.getsource(PANNSDetector.labels.fget)
    assert "from panns_inference.config import labels" in source


def test_both_taggers_are_lazy():
    """The clip tagger answers only detect_music, the per-segment fallback. A
    run with a music map never calls it and must not pay for it."""
    import inspect
    from models.panns import PANNSDetector
    source = inspect.getsource(PANNSDetector.__init__)
    assert "self._tagger = None" in source and "self._sed = None" in source
    assert "AudioTagging(" not in source, "the clip tagger must not load eagerly"
    assert "SoundEventDetection(" not in source


def test_the_dump_never_touches_the_clip_tagger():
    """It needs frame-level scores; loading the clip model would double the
    download for nothing."""
    dump = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tools", "dump_panns.py"), encoding="utf-8").read()
    assert "detector.model" not in dump, "reading .model would load the clip tagger"
    assert "detector.labels" in dump
