"""Transcript review: the LLM may flag unreasonable labels/text, and change nothing.

Run:  python -m pytest tests/test_transcript_review.py -q     (from podcast-pipeline/)
"""
import dataclasses
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.transcript import TranscriptSegment
from services.transcript_review_service import (
    FLAG_KINDS, HARD_FLAG_KINDS, REVIEW_PROMPT_VERSION, TranscriptReviewService,
    apply_review_flags, find_phrase, parse_flags, review_system_prompt)

A, B = "SPEAKER_00", "SPEAKER_01"


def _seg(i, speaker, text=None, bss=False):
    start = i * 5.0
    return TranscriptSegment(
        index=f"{i:05d}", start=start, end=start + 4.0, speaker=speaker,
        text=text or f"câu số {i} nói gì đó",
        text_whisper="w", text_phowhisper="p", text_qwen3="q",
        language="vi", bs_roformer=False, bss=bss, gap_before=0.5)


def _conversation(n=10):
    return [_seg(i, A if i % 2 == 0 else B) for i in range(n)]


def _flag(position, kind="speaker_turn", phrase=None, conf=0.9, why="x"):
    line = position + 1 if isinstance(position, int) else position
    return {"i": line, "kind": kind, "phrase": phrase, "conf": conf, "why": why}


def _json(*flags):
    return json.dumps(list(flags), ensure_ascii=False)


class FakeLLM:
    model_name = "fake/model"

    def __init__(self, reply="[]", loaded=True, batch_size=100):
        self.reply = reply
        self.loaded = loaded
        self.batch_size = batch_size
        self.max_batch_tokens = 0
        self.calls = []
        self.system_prompts = []

    def ensure_loaded(self):
        return self.loaded

    def count_tokens(self, text):
        return len(text.split())

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None, **extra):
        self.calls.append(list(user_messages))
        self.system_prompts.append(system_prompt)
        out = [self.reply(m) if callable(self.reply) else self.reply
               for m in user_messages]
        return True, out


def _small_window(lines=15):
    """A token budget that fits the prompt plus about `lines` transcript lines."""
    svc = TranscriptReviewService(FakeLLM())
    return svc.prompt_overhead(["A", "B"]) + lines * 14


def _review(segments, llm, **kw):
    kw.setdefault("window_tokens", 100000)
    return TranscriptReviewService(llm, **kw).review(segments)


# --- contract constants --------------------------------------------------------

def test_constants_match_the_contract():
    assert REVIEW_PROMPT_VERSION == "review-v3"
    assert FLAG_KINDS == ("speaker_phrase", "speaker_turn", "asr_hallucination", "asr_garbage",
                          "not_vietnamese")
    assert HARD_FLAG_KINDS == ("speaker_phrase", "speaker_turn", "asr_hallucination", "asr_garbage")


# --- parsing --------------------------------------------------------------------

def test_parse_reads_five_keys_and_drops_the_rest():
    raw = _json({**_flag(2, "asr_garbage"), "text": "hack", "start": 1.0})
    assert parse_flags(raw) == [
        {"i": "3", "kind": "asr_garbage", "phrase": None, "conf": 0.9, "why": "x"}]


def test_parse_survives_fences_think_and_garbage():
    raw = "<think>hm</think>\n```json\n" + _json(_flag(0)) + "\n```"
    assert len(parse_flags(raw)) == 1
    assert parse_flags("") == [] and parse_flags("không có gì") == []


# --- valid flags -----------------------------------------------------------------

def test_valid_flags_of_every_kind_are_kept():
    segs = _conversation()
    segs[3].text = "người ta điềm tĩnh đúng rồi và cẩn thận"
    segs[5].text = "hẹn gặp lại zzz blah"
    reply = _json(
        _flag(1, "speaker_turn"),
        _flag(3, "speaker_phrase", "đúng rồi"),
        _flag(5, "asr_hallucination", "zzz blah"),
        _flag(7, "asr_garbage"))
    result = _review(segs, FakeLLM(reply))
    assert result.skipped is None and result.rejected == {}
    f = result.flags_by_index
    assert f["00001"] == [{"kind": "speaker_turn", "phrase": None, "conf": 0.9, "why": "x"}]
    assert f["00003"][0]["phrase"] == "đúng rồi"
    assert f["00005"][0]["kind"] == "asr_hallucination"
    assert f["00007"][0]["phrase"] is None
    assert set(f) == {s.index for s in segs}           # every reviewed index present
    assert f["00000"] == []
    report = result.to_report()
    assert report["flagged_segments"] == 4 and report["flags"] == 4
    assert report["by_kind"]["speaker_turn"] == 1


def test_speaker_turn_ignores_a_phrase_the_model_adds():
    result = _review(_conversation(), FakeLLM(_json(_flag(1, "speaker_turn", "không có"))))
    assert result.flags_by_index["00001"][0]["phrase"] is None


# --- rejections ------------------------------------------------------------------

def test_unknown_kind_is_rejected():
    result = _review(_conversation(), FakeLLM(_json(_flag(1, "bad_kind"))))
    assert result.rejected == {"unknown_kind": 1}
    assert result.flags_by_index["00001"] == []


def test_line_outside_the_window_is_rejected():
    result = _review(_conversation(10), FakeLLM(_json(_flag(11), _flag(0, conf=0.9) | {"i": 0})))
    assert result.rejected == {"unknown_index": 2}
    assert all(v == [] for v in result.flags_by_index.values())


def test_phrase_must_be_a_substring_after_folding():
    segs = _conversation()
    segs[2].text = "Anh  ấy   rất Điềm Tĩnh   đúng rồi nhé"
    ok = _flag(2, "speaker_phrase", "điềm tĩnh ĐÚNG  RỒI")
    bad = _flag(4, "asr_garbage", "không hề có")
    result = _review(segs, FakeLLM(_json(ok, bad)))
    assert result.rejected == {"phrase_not_found": 1}
    stored = result.flags_by_index["00002"][0]["phrase"]
    assert stored == "Điềm Tĩnh   đúng rồi"            # the segment's own spelling
    assert stored in segs[2].text


def test_find_phrase_folds_case_and_whitespace():
    assert find_phrase("Xin  Chào các bạn", "xin chào") == "Xin  Chào"
    assert find_phrase("Xin chào", "tạm biệt") is None
    assert find_phrase("abc", "   ") is None


def test_low_confidence_is_rejected():
    result = _review(_conversation(), FakeLLM(_json(_flag(1, conf=0.5))),
                     min_confidence=0.7)
    assert result.rejected == {"low_confidence": 1}


def test_bad_confidence_is_rejected():
    result = _review(_conversation(), FakeLLM(_json(_flag(1, conf="high"))))
    assert result.rejected == {"bad_confidence": 1}


def test_speaker_phrase_without_phrase_is_rejected():
    result = _review(_conversation(), FakeLLM(_json(_flag(1, "speaker_phrase"))))
    assert result.rejected == {"missing_phrase": 1}


def test_copying_the_prompt_example_is_refused():
    example = re.findall(r'\[\{"i": 902.*\}\]\s*$', review_system_prompt(), re.S)[0]
    reply = example
    result = _review(_conversation(), FakeLLM(reply))
    assert not any(result.flags_by_index.values())
    assert result.rejected.get("unknown_index") == 4      # the four example answers


# --- never changes data; locked may be flagged ------------------------------------

def test_review_changes_nothing_on_the_segments():
    segs = _conversation()
    before = [dataclasses.asdict(s) for s in segs]
    result = _review(segs, FakeLLM(_json(_flag(1), _flag(3, "asr_garbage"))))
    assert [dataclasses.asdict(s) for s in segs] == before     # review() is pure
    apply_review_flags(segs, result.flags_by_index)
    after = [dataclasses.asdict(s) for s in segs]
    for b, a in zip(before, after):
        a = dict(a)
        a.pop("review_flags")
        b = dict(b)
        b.pop("review_flags")
        assert a == b


def test_a_locked_segment_can_be_flagged():
    segs = _conversation()
    segs[1].bss = True
    result = _review(segs, FakeLLM(_json(_flag(1, "speaker_turn"))))
    assert result.rejected == {}
    assert len(result.flags_by_index["00001"]) == 1


# --- windows ---------------------------------------------------------------------

def test_overlapping_windows_do_not_duplicate_flags():
    segs = _conversation(40)

    def everything(message):
        # Flag every visible line: each window sees overlap lines the other does too.
        n = int(re.search(r"Số dòng: 1 đến (\d+)", message).group(1))
        return _json(*[_flag(k, "asr_garbage") for k in range(n)])

    llm = FakeLLM(everything)
    result = _review(segs, llm, window_tokens=_small_window(), overlap_segments=6)
    assert result.windows > 1
    assert len(result.flags_by_index) == 40
    for flags in result.flags_by_index.values():
        assert len(flags) == 1                      # one window decided each segment
    assert result.rejected.get("outside_window_core", 0) > 0


def test_a_flag_for_a_line_outside_the_core_is_not_applied_twice():
    segs = _conversation(40)
    llm = FakeLLM(lambda m: _json(_flag(0, "asr_garbage")))   # line 1 of every window
    result = _review(segs, llm, window_tokens=_small_window(), overlap_segments=6)
    flagged = [i for i, f in result.flags_by_index.items() if f]
    assert len(flagged) == len(set(flagged))


def test_unanswered_or_unreadable_windows_leave_segments_unreviewed():
    segs = _conversation()
    result = _review(segs, FakeLLM("Tôi nghĩ mọi thứ đều ổn."))
    assert result.flags_by_index == {}
    assert result.to_report()["unreadable_windows"] == 1


def test_skips():
    assert _review([], FakeLLM()).skipped == "no_segments"
    assert _review(_conversation(), FakeLLM(loaded=False)).skipped == "llm_unavailable"


def test_an_empty_array_marks_everything_clean():
    result = _review(_conversation(), FakeLLM("[]"))
    assert all(v == [] for v in result.flags_by_index.values())
    assert len(result.flags_by_index) == 10


# --- apply_review_flags ------------------------------------------------------------

def test_apply_sets_flags_empty_list_and_leaves_unreviewed_none():
    segs = _conversation(4)
    flags = {"00000": [], "00001": [{"kind": "speaker_turn", "phrase": None,
                                     "conf": 0.9, "why": "x"}], "00002": []}
    assert apply_review_flags(segs, flags) == 1
    assert segs[0].review_flags == []
    assert segs[1].review_flags[0]["kind"] == "speaker_turn"
    assert segs[3].review_flags is None
    assert apply_review_flags(segs, flags) == 1       # idempotent


def test_default_review_flags_is_none():
    assert _seg(0, A).review_flags is None


# --- checkpoint namespace ------------------------------------------------------------

def test_checkpoint_namespace_tracks_version_and_settings(monkeypatch):
    base = TranscriptReviewService(FakeLLM()).checkpoint_namespace
    assert REVIEW_PROMPT_VERSION in base
    assert re.fullmatch(r"[A-Za-z0-9._-]+", base)
    for kw in ({"window_tokens": 2000}, {"overlap_segments": 5},
               {"min_confidence": 0.9}, {"thinking": True},
               {"max_window_segments": 40}, {"max_new_tokens": 300}):
        assert TranscriptReviewService(FakeLLM(), **kw).checkpoint_namespace != base
    import services.transcript_review_service as mod
    monkeypatch.setattr(mod, "REVIEW_PROMPT_VERSION", "review-v99")
    assert TranscriptReviewService(FakeLLM()).checkpoint_namespace != base


def test_prompt_carries_the_confidence_floor():
    llm = FakeLLM()
    _review(_conversation(), llm, min_confidence=0.85)
    assert "0.85" in llm.system_prompts[0] and "{min_confidence}" not in llm.system_prompts[0]


def test_find_phrase_returns_the_segments_own_spelling_for_decomposed_text():
    import unicodedata
    from services.transcript_review_service import find_phrase
    text = unicodedata.normalize("NFD", "rất điềm tĩnh Đúng Rồi và cẩn thận")
    found = find_phrase(text, "đúng rồi")
    assert found is not None and found in text


# --- not_vietnamese ------------------------------------------------------------

def test_not_vietnamese_is_a_whole_line_flag_even_if_the_model_quotes_a_phrase():
    segs = _conversation()
    segs[4].text = "Yeah I think that is really interesting, let me explain why"
    result = _review(segs, FakeLLM(_json(_flag(4, "not_vietnamese", phrase="really interesting"))))
    assert result.flags_by_index[segs[4].index][0]["kind"] == "not_vietnamese"
    assert result.flags_by_index[segs[4].index][0]["phrase"] is None
    assert "not_vietnamese" not in HARD_FLAG_KINDS      # judged by share, not presence


def test_not_vietnamese_still_requires_confidence_floor():
    result = _review(_conversation(), FakeLLM(
        _json(_flag(4, "not_vietnamese", conf=0.69))), min_confidence=0.7)
    assert result.flags_by_index["00004"] == []
    assert result.rejected == {"low_confidence": 1}


def test_the_prompt_asks_for_not_vietnamese_and_keeps_borrowed_words_unflagged():
    llm = FakeLLM()
    _review(_conversation(), llm)
    prompt = llm.system_prompts[0]
    assert "not_vietnamese" in prompt and "NĂM LOẠI CỜ" in prompt
    assert "BÌNH THƯỜNG" in prompt
    assert "một câu tiếng nước ngoài có nghĩa" in prompt
