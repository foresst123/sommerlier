"""Transcript-review flags in conversation selection, and the select/write split.

Run:  python -m pytest tests/test_conversation_review_flags.py -q     (from podcast-pipeline/)
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sf = pytest.importorskip("soundfile")

from services.conversation_export_service import ConversationExportService
from utils.conversation_selection import (
    ConversationSelectionConfig, ConversationSelectionFinder, foreign_seconds)
from utils.excise import TimelineMap
from utils.noise_map import KINDS, NoiseTrack

A, B = "SPEAKER_00", "SPEAKER_01"
SR = 8000


def _seg(i, speaker, start, end, flags=None, words=False):
    seg = SimpleNamespace(
        index=f"{i:05d}", speaker=speaker, start=start, end=end,
        text=f"đoạn {i} bàn về chuyện số {i} rất dài dòng", bss=False, unseparated=None)
    if flags is not None:
        seg.review_flags = flags
    if words:
        n = len(seg.text.split())
        step = (end - start) / n
        seg.words = [{"word": w, "start": start + k * step, "end": start + (k + 1) * step}
                     for k, w in enumerate(seg.text.split())]
    return seg


def _talk(n=30, words=False):
    return [_seg(k, (A, B)[k % 2], k * 8.0, k * 8.0 + 7.0, words=words) for k in range(n)]


def _quiet():
    return NoiseTrack({k: np.full(60000, 0.001, dtype=np.float32) for k in KINDS}, fps=100.0)


def _finder(segs, **cfg):
    return ConversationSelectionFinder(
        segs, TimelineMap(), _quiet(), None, ConversationSelectionConfig(**cfg))


def flag(kind, phrase=None):
    return {"kind": kind, "phrase": phrase, "conf": 0.9, "why": "test"}


# --- validator ----------------------------------------------------------------

@pytest.mark.parametrize("kind", ["speaker_turn", "asr_hallucination", "asr_garbage"])
def test_a_hard_flag_rejects_the_range_with_its_kind(kind):
    segs = _talk()
    segs[10].review_flags = [flag(kind)]
    finder = _finder(segs)
    cand, why = finder.evaluate(0, 14)
    assert cand is None and why == f"review_flag_{kind}"
    cand, why = finder.evaluate(11, 25)
    assert cand is not None, why


def test_hard_reject_can_be_turned_off():
    segs = _talk()
    segs[10].review_flags = [flag("speaker_turn")]
    cand, _ = _finder(segs, review_hard_reject=False).evaluate(0, 14)
    assert cand is not None


def test_soft_flags_lower_the_score_and_are_counted():
    clean = _finder(_talk()).evaluate(0, 14)[0]
    segs = _talk()
    segs[3].review_flags = [flag("speaker_phrase", "đúng rồi")]
    segs[6].review_flags = [flag("speaker_phrase", "ừ")]
    soft = _finder(segs).evaluate(0, 14)[0]
    assert soft is not None
    assert soft.metrics["review_soft_flags"] == 2
    assert clean.metrics["review_soft_flags"] == 0
    assert soft.score == pytest.approx(clean.score - 10.0, abs=0.11)


def test_none_and_empty_flags_are_clean():
    segs = _talk()
    segs[2].review_flags = []
    segs[4].review_flags = None
    cand, why = _finder(segs).evaluate(0, 14)
    assert cand is not None, why
    assert cand.metrics["review_soft_flags"] == 0


def test_a_hard_flag_shows_in_the_segment_quality_hard_flags():
    segs = _talk()
    segs[1].review_flags = [flag("asr_garbage")]
    assert "review_flag_asr_garbage" in _finder(segs).segment_quality(1)["hard_flags"]


def test_word_alignment_no_longer_gates_selection():
    cand, why = _finder(_talk(), require_word_alignment=True).evaluate(0, 14)
    assert cand is not None, why


# --- the finder line ------------------------------------------------------------

def _line(seg_flags):
    segs = _talk()
    segs[0].review_flags = seg_flags
    svc = ConversationExportService(SimpleNamespace(count_tokens=lambda t: len(t.split())))
    return svc._finder_line(_finder(segs), 0, 1, {A: "A", B: "B"})


def test_finder_line_shows_a_hard_flag_and_a_truncated_phrase():
    assert "review=speaker_turn" in _line([flag("speaker_turn")])
    line = _line([flag("speaker_phrase", "đúng rồi")])
    assert 'review=speaker_phrase:"đúng rồi"' in line
    long = _line([flag("speaker_phrase", "x" * 60)])
    assert "x" * 60 not in long and 'speaker_phrase:"' in long


def test_finder_line_omits_review_when_clean():
    assert "review=" not in _line(None)
    assert "review=" not in _line([])


# --- select / write ----------------------------------------------------------------

class FinderLLM:
    model_name = "fake/model"
    batch_size = 100
    max_batch_tokens = 0

    def ensure_loaded(self):
        return True

    def count_tokens(self, text):
        return len(text.split())

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None, **extra):
        reply = json.dumps([{"start_line": 1, "end_line": 30, "keep": True,
                             "self_contained": 5, "topic": "chủ đề thử", "reason": "ok"}])
        return True, [reply for _ in user_messages]


def _wave():
    return (np.random.default_rng(0).standard_normal(int(300 * SR)) * 0.1).astype(np.float32)


def _service(**settings):
    return ConversationExportService(FinderLLM(), **settings)


def _kw():
    return dict(timeline=TimelineMap(), noise=_quiet(), music_map=None, waveform=_wave(),
                sample_rate=SR, base_name="ep")


def _files(root):
    return [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]


def test_select_writes_nothing_and_lists_segments_in_order(tmp_path):
    segs = _talk()
    selection = _service().select(segs, **_kw())
    assert selection.report["skipped"] is None, selection.report
    assert selection.final
    assert _files(tmp_path) == []
    aligned = selection.segments_to_align()
    assert [s.index for s in aligned] == sorted(s.index for s in aligned)
    assert len({id(s) for s in aligned}) == len(aligned)
    first, last = selection.final[0].first, selection.final[0].last
    assert len(aligned) >= last - first + 1


def test_write_drops_unaligned_excerpts_when_required(tmp_path):
    svc = _service(require_word_alignment=True)
    selection = svc.select(_talk(words=False), **_kw())
    assert selection.final
    run = svc.write(selection, waveform=_wave(), sample_rate=SR,
                    out_dir=str(tmp_path), base_name="ep")
    assert run.exports == []
    assert run.report["dropped_unaligned"] == len(selection.final)


def test_write_keeps_aligned_excerpts_when_required(tmp_path):
    svc = _service(require_word_alignment=True)
    selection = svc.select(_talk(words=True), **_kw())
    run = svc.write(selection, waveform=_wave(), sample_rate=SR,
                    out_dir=str(tmp_path), base_name="ep")
    assert run.report["dropped_unaligned"] == 0
    assert run.exports


def test_write_keeps_unaligned_excerpts_when_not_required(tmp_path):
    svc = _service()
    selection = svc.select(_talk(words=False), **_kw())
    run = svc.write(selection, waveform=_wave(), sample_rate=SR,
                    out_dir=str(tmp_path), base_name="ep")
    assert run.exports and "dropped_unaligned" not in run.report


def test_run_equals_write_of_select(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    svc = _service()
    run = svc.run(_talk(), out_dir=str(a), **_kw())
    selection = svc.select(_talk(), **_kw())
    split = svc.write(selection, waveform=_wave(), sample_rate=SR,
                      out_dir=str(b), base_name="ep")
    strip = lambda r: [{k: v for k, v in e.items() if k != "files"} for e in r.exports]
    assert strip(run) == strip(split) and run.exports
    assert run.report["exported"] == split.report["exported"]


def test_conversation_json_rows_carry_nonempty_review_flags(tmp_path):
    segs = _talk()
    segs[5].review_flags = [flag("speaker_phrase", "đúng rồi")]
    segs[6].review_flags = []
    svc = _service()
    run = svc.run(segs, out_dir=str(tmp_path), **_kw())
    assert run.exports
    doc = json.loads(next(
        open(p, encoding="utf-8").read() for p in _files(tmp_path)
        if p.endswith("conversation.json")))
    rows = {r["index"]: r for r in doc["conversation"]}
    assert rows["00005"]["review_flags"][0]["kind"] == "speaker_phrase"
    assert "review_flags" not in rows["00006"]
    assert "review_flags" not in rows["00004"]


# --- not_vietnamese: rejected by share, not by presence ---------------------------

def test_a_short_foreign_line_does_not_sink_an_excerpt():
    segs = _talk()
    segs[10].review_flags = [flag("not_vietnamese")]       # 7 s of 120 s, about 6%
    cand, why = _finder(segs).evaluate(5, 20)
    assert cand is not None, why
    assert cand.metrics["foreign_share"] == pytest.approx(7.0 / (cand.end - cand.start), abs=1e-3)


def test_too_much_foreign_text_rejects_the_range():
    segs = _talk()
    for k in (8, 9, 10):                                   # 21 s of about 120 s, over 10 %
        segs[k].review_flags = [flag("not_vietnamese")]
    cand, why = _finder(segs).evaluate(5, 20)
    assert cand is None and why == "not_vietnamese"


def test_the_foreign_share_limit_is_configurable():
    segs = _talk()
    for k in (8, 9, 10):
        segs[k].review_flags = [flag("not_vietnamese")]
    cand, why = _finder(segs, max_foreign_share=0.5).evaluate(5, 20)
    assert cand is not None, why


def test_foreign_share_at_the_limit_is_allowed():
    segs = _talk()
    # The selected span is 127 seconds; make the flagged duration exactly 10%.
    # The gate is intentionally strict only above the configured maximum.
    segs[5].end = segs[5].start + 12.7
    segs[5].review_flags = [flag("not_vietnamese")]
    cand, why = _finder(segs).evaluate(0, 15)
    assert cand is not None, why
    assert cand.metrics["foreign_share"] == pytest.approx(0.10)


def test_zero_duration_foreign_segment_has_zero_share():
    segs = _talk()
    segs[3].end = segs[3].start
    segs[3].review_flags = [flag("not_vietnamese")]
    assert foreign_seconds(segs[3]) == 0.0


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_max_foreign_share_must_be_a_finite_fraction(value):
    with pytest.raises(ValueError):
        ConversationSelectionConfig(max_foreign_share=value)


def test_not_vietnamese_is_not_a_hard_flag():
    segs = _talk()
    segs[3].review_flags = [flag("not_vietnamese")]
    assert _finder(segs).segment_quality(3)["hard_flags"] == []


def test_finder_line_shows_not_vietnamese_and_the_prompt_no_longer_owns_language():
    from services.conversation_export_service import CONVERSATION_FINDER_SYSTEM_PROMPT as P
    assert "review=not_vietnamese" in _line([flag("not_vietnamese")])
    assert "không phải hội thoại tiếng Việt" not in P
    assert "review=not_vietnamese" in P
