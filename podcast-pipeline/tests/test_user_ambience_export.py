"""The export's user channel keeps the room (user_ambience.mode), checked through fakes.

utils/ambience_bed.py is still a contract, so its four functions are replaced by fakes with known answers. These
tests are about what the exporter passes them, what it does with the answer, and what it writes when they fail.
"""
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.conversation_export_service import (  # noqa: E402
    STRICT_TRACKS, TIME_GATED, ConversationExportService)
from utils.ambience_bed import BedResult  # noqa: E402
from utils.conversation_events import RoleResult  # noqa: E402
from test_conversation_export_service import (  # noqa: E402
    A, B, SR, FakeLLM, FakeSeparation, Logger, _good, _run, _speech, _talk, _only, _wave)

C = "SPEAKER_02"                       # a third person in the recording
WAVE = _wave(60)                       # 60 s of noise at SR
GAP = (2.0, 8.0)                       # the fake quiet gap of the recording


def _segments():
    return [
        _speech(A, 10.0, 13.0),        # the user
        _speech(B, 12.0, 15.0),        # the assistant, speaking over the user from 12 to 13
        _speech(C, 16.0, 17.0),        # a third person
        _speech(A, 18.0, 19.0),
    ]


def _cand(last=3):
    return SimpleNamespace(first=0, last=last, pad_start=10.0, pad_end=20.0,
                           speakers=(A, B), start=10.0, end=19.0)


def _roles(user=A, assistant=B):
    return RoleResult(assistant=assistant, user=user, role="locked", role_conf=None,
                      violations=[], swappable=False, source="behaviour")


def _svc(mode, logger=None):
    return ConversationExportService(FakeLLM(_good()), logger=logger,
                                     user_ambience=({"mode": mode} if mode else None))


def _install(monkeypatch, calls, fail=None):
    """Replace the four functions by fakes that record their arguments; `fail` names one that raises."""
    import utils.ambience_bed as ab

    def wrap(name, fn):
        def fake(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            if fail == name:
                raise RuntimeError("boom")
            return fn(*args, **kwargs)
        return fake

    monkeypatch.setattr(ab, "find_quiet_gaps", wrap(
        "find_quiet_gaps", lambda spans, lo, hi, params, waveform=None, sr=None: [GAP]))
    monkeypatch.setattr(ab, "build_bed", wrap(
        "build_bed",
        lambda waveform, sr, gaps, spans, start_s, end_s, params, focus_s=None: BedResult(
            audio=np.full(round((end_s - start_s) * sr), 0.01, dtype=np.float32),
            tier="local", pieces=2, repeats=0)))
    monkeypatch.setattr(ab, "safe_mask", wrap(
        "safe_mask",
        lambda n, sr, start_s, other, third, bad, overlap, params, tail_ms=None: np.arange(n) < n // 2))
    monkeypatch.setattr(ab, "compose_user_channel", wrap(
        "compose_user_channel",
        lambda raw, user_track, bed, safe, sr, params: np.where(
            safe, raw, user_track + bed).astype(np.float32)))


_DEFAULT = object()


def _render(svc, roles=None, separation=_DEFAULT, segs=None, speech=None):
    """Cut the excerpt. `separation=None` means no separation stage (the time-gated fallback)."""
    segs = segs if segs is not None else _segments()
    speech = speech if speech is not None else segs
    fake = FakeSeparation() if separation is _DEFAULT else separation
    return svc._render(SimpleNamespace(segs=segs), _cand(), WAVE, SR, speech, fake, roles=roles)


def _mid(out):
    return len(out.mixture) // 2


def test_none_is_the_old_behaviour_and_calls_nothing(monkeypatch):
    calls = {}
    _install(monkeypatch, calls)
    svc = _svc(None)
    out = _render(svc, roles=_roles())
    assert calls == {}
    assert out.ambience is None
    assert out.method == STRICT_TRACKS
    assert out.left[_mid(out)] == pytest.approx(0.25, abs=1e-3)
    assert out.right[_mid(out)] == pytest.approx(-0.25, abs=1e-3)
    # The same answer without roles: the new argument changes nothing while the mode is off.
    again = _render(svc, roles=None)
    assert np.array_equal(again.left, out.left) and np.array_equal(again.right, out.right)


def test_raw_mask_puts_the_room_on_the_user_channel_and_keeps_the_assistant(monkeypatch):
    calls = {}
    _install(monkeypatch, calls)
    out = _render(_svc("raw_mask"), roles=_roles())
    n = len(out.mixture)
    mid = _mid(out)

    # Unsafe part (second half): the user's voice on the bed. Safe part (first half): the recording itself.
    assert out.left[mid + 10] == pytest.approx(0.26, abs=1e-3)
    assert out.left[n // 4] == pytest.approx(out.mixture[n // 4], abs=1e-6)
    # The assistant channel is the separated track as it was.
    assert out.right[mid] == pytest.approx(-0.25, abs=1e-3)

    assert out.ambience["mode"] == "raw_mask"
    assert out.ambience["applied"] is True
    assert out.ambience["bed_tier"] == "local"
    assert out.ambience["bed_pieces"] == 2 and out.ambience["bed_repeats"] == 0
    assert out.ambience["safe_fraction"] == pytest.approx(0.5)


def test_the_arguments_are_the_right_speakers_spans(monkeypatch):
    calls = {}
    _install(monkeypatch, calls)
    out = _render(_svc("raw_mask"), roles=_roles())
    lo_s, hi_s = out.lo / SR, out.hi / SR

    (spans, lo, hi, _params, waveform, sr), _ = calls["find_quiet_gaps"][0]
    assert spans == [(10.0, 13.0), (12.0, 15.0), (16.0, 17.0), (18.0, 19.0)]
    assert (lo, hi) == (0.0, len(WAVE) / SR) and waveform is not None and sr == SR

    (waveform, sr, gaps, spans, start_s, end_s, _params), kw = calls["build_bed"][0]
    assert gaps == [GAP] and len(spans) == 4
    assert start_s == pytest.approx(lo_s) and end_s == pytest.approx(hi_s)
    assert kw["focus_s"] == pytest.approx((lo_s + hi_s) / 2)

    (n, sr, start_s, other, third, bad, overlap, _params), _ = calls["safe_mask"][0]
    assert n == len(out.mixture) and start_s == pytest.approx(lo_s)
    assert other == [(12.0, 15.0)], "the other speaker is the assistant"
    assert third == [(16.0, 17.0)], "the third speaker"
    assert bad == []
    assert overlap == [(12.0, 13.0)], "where user and assistant both speak"


def test_when_the_user_is_the_second_speaker_the_right_channel_is_replaced(monkeypatch):
    calls = {}
    _install(monkeypatch, calls)
    out = _render(_svc("raw_mask"), roles=_roles(user=B, assistant=A))
    mid = _mid(out)
    assert out.right[mid + 10] == pytest.approx(-0.24, abs=1e-3)   # -0.25 + 0.01 bed
    assert out.left[mid] == pytest.approx(0.25, abs=1e-3)
    (_, _, _, other, third, _, overlap, _), _ = calls["safe_mask"][0]
    assert other == [(10.0, 13.0), (18.0, 19.0)], "the other speaker is now A"
    assert third == [(16.0, 17.0)]
    assert overlap == [(12.0, 13.0)]


@pytest.mark.parametrize("failing", ["find_quiet_gaps", "build_bed", "safe_mask", "compose_user_channel"])
def test_a_failure_keeps_the_old_track_and_says_why(monkeypatch, failing):
    calls = {}
    _install(monkeypatch, calls, fail=failing)
    logger = Logger()
    out = _render(_svc("raw_mask", logger=logger), roles=_roles())
    assert out.left[_mid(out)] == pytest.approx(0.25, abs=1e-3), "the track is the old one"
    assert out.right[_mid(out)] == pytest.approx(-0.25, abs=1e-3)
    assert out.ambience["mode"] == "raw_mask"
    assert out.ambience["applied"] is False
    assert "boom" in out.ambience["error"]
    assert logger.warnings, "the failure is logged"


def test_a_time_gated_excerpt_is_not_given_the_room(monkeypatch):
    calls = {}
    _install(monkeypatch, calls)
    segs = _segments()
    out = _render(_svc("raw_mask"), roles=_roles(), separation=None, segs=segs)
    assert out.method == TIME_GATED
    assert out.ambience == {"mode": "raw_mask", "applied": False, "reason": "not_separated_tracks"}
    assert calls == {}


def test_without_roles_nothing_is_changed_and_the_reason_is_kept(monkeypatch):
    calls = {}
    _install(monkeypatch, calls)
    out = _render(_svc("raw_mask"), roles=None)
    assert out.ambience == {"mode": "raw_mask", "applied": False, "reason": "no_roles"}
    assert out.left[_mid(out)] == pytest.approx(0.25, abs=1e-3)
    assert calls == {}


@pytest.mark.parametrize("mode, present", [("none", False), ("raw_mask", True)])
def test_conversation_json_records_the_user_ambience_only_when_it_is_on(tmp_path, monkeypatch, mode, present):
    calls = {}
    _install(monkeypatch, calls)
    segs = _talk(30)
    speech = [_speech(seg.speaker, seg.start, seg.end) for seg in segs]
    _run(tmp_path, FakeLLM(_good()), segs=segs, max_candidates=1,
         user_ambience={"mode": mode}, run_kwargs={"speech_segments": speech,
                                                  "separation_service": FakeSeparation()})
    _, meta = _only(tmp_path)
    if present:
        assert meta["user_ambience"]["mode"] == "raw_mask"
        assert meta["user_ambience"]["applied"] is True
        assert calls["compose_user_channel"], "the user channel was built with the room"
    else:
        assert "user_ambience" not in meta
        assert calls == {}
    json.dumps(meta)                       # still plain JSON


def test_the_safe_mask_gets_the_reverb_tail_cap_of_the_params(monkeypatch):
    calls = {}
    _install(monkeypatch, calls)
    svc = _svc("raw_mask")
    _render(svc, roles=_roles())
    (_, _, _, _, _, _, _, _), kw = calls["safe_mask"][0]
    assert kw["tail_ms"] == svc.user_ambience.reverb_tail_max_ms == 400.0


def test_an_assistant_span_the_speech_list_lacks_still_makes_the_user_channel_unsafe(monkeypatch):
    calls = {}
    _install(monkeypatch, calls)
    # The separation stage left the assistant out of speech_segments; the ASR segments still have it.
    speech = [_speech(A, 10.0, 13.0), _speech(C, 16.0, 17.0), _speech(A, 18.0, 19.0)]
    _render(_svc("raw_mask"), roles=_roles(), speech=speech)
    (_, _, _, other, third, _, _, _), _ = calls["safe_mask"][0]
    assert other == [(12.0, 15.0)], "the assistant's ASR span, once"
    assert third == [(16.0, 17.0)]


def test_a_third_speaker_only_the_asr_segments_know_of_is_a_third_span(monkeypatch):
    calls = {}
    _install(monkeypatch, calls)
    speech = [_speech(A, 10.0, 13.0), _speech(B, 12.0, 15.0), _speech(A, 18.0, 19.0)]
    _render(_svc("raw_mask"), roles=_roles(), speech=speech)
    (_, _, _, other, third, _, _, _), _ = calls["safe_mask"][0]
    assert other == [(12.0, 15.0)]
    assert third == [(16.0, 17.0)], "the ASR segment of speaker C, which speech_segments lacks"


def test_four_render_workers_find_the_quiet_gaps_once(monkeypatch):
    import utils.ambience_bed as ab
    calls = []

    def slow_gaps(spans, lo, hi, params, waveform=None, sr=None):
        calls.append(threading.get_ident())
        time.sleep(0.2)                    # long enough that the other threads arrive while it runs
        return [GAP]

    monkeypatch.setattr(ab, "find_quiet_gaps", slow_gaps)
    svc = _svc("raw_mask")
    speech = _segments()
    spans = [(float(s.start), float(s.end)) for s in speech]
    barrier = threading.Barrier(4)
    results = [None] * 4

    def work(i):
        barrier.wait()
        results[i] = svc._quiet_gaps(speech, spans, WAVE, SR)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(calls) == 1, "one computation for the recording, the others wait for it"
    assert all(r is results[0] for r in results) and results[0] == [GAP]
    # The cache still holds one entry, and a later call is a hit too.
    assert svc._quiet_gaps(speech, spans, WAVE, SR) is results[0]
    assert len(calls) == 1
