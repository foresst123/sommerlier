"""DiarizationService's raw/post split -- see
docs/superpowers/specs/2026-09-23-diarization-async-postprocess-design.md

Run:  python -m pytest tests/test_diarization_service.py -q   (from podcast-pipeline/)
"""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pyannote.audio is a heavy real dependency (requirements.txt) not installed
# in every dev sandbox; services/diarization_service.py imports it at module
# scope, so this whole file skips cleanly here instead of erroring, and runs
# for real wherever the dependency is actually present.
pyannote_core = pytest.importorskip("pyannote.core")
Annotation = pyannote_core.Annotation
PyannoteSegment = pyannote_core.Segment

from schemas.audio import AudioData
from services.diarization_service import DiarizationService


class _FakeDiarizer:
    """Returns a fixed pyannote Annotation, ignoring speaker-bound kwargs."""

    def __init__(self, tracks):
        self._tracks = tracks  # [(start, end, speaker), ...]
        self.calls = 0

    def diarize(self, audio_input, **kwargs):
        self.calls += 1
        annotation = Annotation()
        for start, end, speaker in self._tracks:
            annotation[PyannoteSegment(start, end)] = speaker
        return annotation


def _audio(duration=20.0, sr=16000):
    return AudioData(
        waveform=np.zeros(int(duration * sr), dtype=np.float32),
        sample_rate=sr, name="test", audio_segment=None, duration=duration)


def _args(**kw):
    a = types.SimpleNamespace(dia3=False, vad=False, merge_gap=0.5,
                              max_segment_length=30.0, bridge_gap=3.0)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _service(tracks):
    return DiarizationService(diarizer=_FakeDiarizer(tracks))


def test_diarize_raw_does_not_run_any_cpu_postprocessing():
    """The GPU-only half must not touch filter/VAD/merge/split at all."""
    import services.diarization_service as mod
    patched = ("filter_diarizer_noise", "cut_by_speaker_label",
               "bridge_interrupted_speaker_turns", "split_long_segments")
    originals = {name: getattr(mod, name) for name in patched}
    calls = []

    def _make_tracker(name, original):
        def _tracker(*a, **kw):
            calls.append(name)
            return original(*a, **kw)
        return _tracker

    for name in patched:
        setattr(mod, name, _make_tracker(name, originals[name]))
    try:
        svc = _service([(0.0, 2.0, "A"), (1.5, 3.0, "B")])
        raw = svc.diarize_raw([], _audio(), _args())
    finally:
        for name in patched:
            setattr(mod, name, originals[name])
    assert calls == [], f"diarize_raw must be GPU-only, but it ran: {calls}"
    assert raw.method == "diarizen"
    assert len(raw.combined_df) == 2


def test_diarize_postprocess_reproduces_run_diarization_output():
    """diarize_postprocess(diarize_raw(...)) must equal today's run_diarization()."""
    tracks = [(0.0, 2.0, "A"), (1.5, 3.0, "B"), (5.0, 5.05, "A")]
    svc_a = _service(tracks)
    svc_b = _service(tracks)
    audio = _audio()
    args = _args()

    raw = svc_a.diarize_raw([], audio, args)
    via_split = svc_a.diarize_postprocess(raw, audio, args)
    via_monolith = svc_b.run_diarization([], audio, args)

    assert [(s.start, s.end, s.speaker) for s in via_split.segments] == \
           [(s.start, s.end, s.speaker) for s in via_monolith.segments]
    assert via_split.num_speakers == via_monolith.num_speakers
    assert via_split.method == via_monolith.method == "diarizen"


def test_submit_postprocess_runs_in_the_background_and_resolves():
    svc = _service([(0.0, 2.0, "A"), (1.5, 3.0, "B")])
    audio = _audio()
    args = _args()
    raw = svc.diarize_raw([], audio, args)

    future = svc.submit_postprocess(raw, audio, args)
    result = future.result(timeout=5)
    assert len(result.segments) >= 1
    svc.close_postprocess_pool()


def test_close_postprocess_pool_is_a_safe_no_op_before_any_submit_and_when_called_twice():
    svc = _service([])
    svc.close_postprocess_pool()
    svc.close_postprocess_pool()
