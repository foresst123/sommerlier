"""Speaker relabel: Qwen may say which speaker a segment belongs to, and nothing else.

The invariant these tests exist to protect: after the pass, every field of every
segment except `speaker` (and the trace field `speaker_original`) is exactly what
it was. Timestamps and text have no way through -- the parser reads four keys,
and the apply step writes one attribute.

Run:  python -m pytest tests/test_speaker_relabel.py -q     (from podcast-pipeline/)
"""
import copy
import dataclasses
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.transcript import TranscriptSegment
from services.speaker_relabel_service import (
    RELABEL_PROMPT_VERSION, SpeakerRelabelService, apply_relabels,
    parse_proposals, relabel_system_prompt, speaker_names)

A, B = "SPEAKER_00", "SPEAKER_01"          # the diarizer's labels
LETTER = {A: "A", B: "B"}                  # what the model is told they are


# --- helpers -----------------------------------------------------------------

def _seg(i, speaker, text=None, bss=False, unseparated=None, gap=0.5):
    start = i * 5.0
    return TranscriptSegment(
        index=f"{i:05d}", start=start, end=start + 4.0, speaker=speaker,
        text=text or f"câu số {i} nói gì đó",
        text_whisper="w", text_phowhisper="p", text_qwen3="q",
        language="vi", bs_roformer=False, bss=bss, unseparated=unseparated,
        gap_before=gap)


def _conversation(n=10, wrong=None):
    """Alternating A/B turns; positions in `wrong` carry the opposite label."""
    wrong = set(wrong or ())
    segs = []
    for i in range(n):
        speaker = A if i % 2 == 0 else B
        if i in wrong:
            speaker = B if speaker == A else A
        segs.append(_seg(i, speaker))
    return segs


def _proposal(position, speaker, conf=0.9, why="trả lời câu hỏi"):
    """A proposal as the model writes it: a line number and a letter.

    `position` is the segment's place in the file, and `speaker` the diarizer's
    label; with one window covering the file, line = position + 1.
    """
    line = position + 1 if isinstance(position, int) else position
    return {"i": line, "speaker": LETTER.get(speaker, speaker), "conf": conf, "why": why}


class FakeLLM:
    """Stands in for DiarizationRefinementService's model-facing surface."""

    model_name = "fake/model"

    def __init__(self, reply="[]", max_batch_tokens=0, batch_size=100,
                 loaded=True, fail_multi=False, fail_all=False):
        self.reply = reply
        self.max_batch_tokens = max_batch_tokens
        self.batch_size = batch_size
        self.loaded = loaded
        self.fail_multi = fail_multi
        self.fail_all = fail_all
        self.calls = []
        self.system_prompts = []

    def ensure_loaded(self):
        return self.loaded

    def count_tokens(self, text):
        return len(text.split())

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None):
        self.calls.append(list(user_messages))
        self.system_prompts.append(system_prompt)
        if self.fail_all or (self.fail_multi and len(user_messages) > 1):
            return False, []
        out = [self.reply(m) if callable(self.reply) else self.reply
               for m in user_messages]
        return True, out


def _json(*proposals):
    return json.dumps(list(proposals), ensure_ascii=False)


def _flip_everything_visible(message):
    """A model that disagrees with every line it is shown."""
    proposals = []
    for number, letter in re.findall(r"^\[(\d+)\] \S+ ([A-Z])\b", message, re.M):
        proposals.append({"i": int(number), "speaker": "B" if letter == "A" else "A",
                          "conf": 0.9, "why": "x"})
    return json.dumps(proposals)


def _relabel(segments, llm, **kw):
    kw.setdefault("window_tokens", 100000)
    return SpeakerRelabelService(llm, **kw).relabel(segments)


# --- the parser reads four keys and survives messy output ---------------------

def test_a_plain_json_list_is_parsed():
    out = parse_proposals(_json(_proposal(4, B)))
    assert out == [{"i": "5", "speaker": "B", "conf": 0.9, "why": "trả lời câu hỏi"}]


def test_a_fenced_block_and_think_tags_are_stripped():
    raw = "<think>hmm</think>\n```json\n" + _json(_proposal(4, B)) + "\n```"
    assert [p["speaker"] for p in parse_proposals(raw)] == ["B"]


def test_an_empty_list_and_garbage_both_mean_no_changes():
    assert parse_proposals("[]") == []
    assert parse_proposals("") == []
    assert parse_proposals("Không có đoạn nào sai.") == []


def test_output_cut_off_mid_object_keeps_the_complete_objects():
    whole = _json(_proposal(4, B), _proposal(6, B))
    cut = whole[: whole.rindex('{') + 12]          # second object is truncated
    assert [p["i"] for p in parse_proposals(cut)] == ["5"]


def test_a_wrapper_object_is_unwrapped():
    raw = json.dumps({"changes": [_proposal(4, B)]})
    assert [p["i"] for p in parse_proposals(raw)] == ["5"]


def test_a_numeric_line_is_kept_as_text_and_read_as_a_number_later():
    assert parse_proposals(json.dumps([{"i": 12, "speaker": B, "conf": 1}]))[0]["i"] == "12"


def test_only_four_keys_survive_parsing():
    """Text and timestamps have no path through: they are dropped here."""
    raw = json.dumps([{"i": 5, "speaker": "B", "conf": 0.9, "why": "x",
                       "text": "bịa", "start": 1.0, "end": 2.0}])
    (proposal,) = parse_proposals(raw)
    assert set(proposal) == {"i", "speaker", "conf", "why"}


# --- the invariant: only `speaker` changes -----------------------------------

def test_a_relabelled_segment_changes_speaker_and_nothing_else():
    segs = _conversation(wrong={4})
    before = copy.deepcopy(segs)
    result = _relabel(segs, FakeLLM(_json(_proposal(4, A))))
    apply_relabels(segs, result.mapping)

    assert segs[4].speaker == A and segs[4].speaker_original == B
    for old, new in zip(before, segs):
        a, b = dataclasses.asdict(old), dataclasses.asdict(new)
        for key in a:
            if key in ("speaker", "speaker_original"):
                continue
            assert a[key] == b[key], key


def test_a_reply_that_carries_text_and_times_cannot_change_them():
    segs = _conversation(wrong={4})
    before = copy.deepcopy(segs)
    reply = json.dumps([{"i": 5, "speaker": "A", "conf": 0.9, "why": "x",
                         "text": "lời bịa đặt", "start": 99.0, "end": 100.0}])
    result = _relabel(segs, FakeLLM(reply))
    apply_relabels(segs, result.mapping)
    assert (segs[4].text, segs[4].start, segs[4].end) == (
        before[4].text, before[4].start, before[4].end)
    assert segs[4].speaker == A


def test_relabel_itself_does_not_touch_the_segments():
    segs = _conversation(wrong={4})
    before = copy.deepcopy(segs)
    _relabel(segs, FakeLLM(_json(_proposal(4, A))))
    assert segs == before


def test_speech_segments_get_the_same_speaker_and_keep_their_audio():
    segs = _conversation(wrong={4})
    marker = object()
    speech = [type("S", (), {})() for _ in segs]
    for s, t in zip(speech, segs):
        s.index, s.speaker, s.audio = t.index, t.speaker, marker
    result = _relabel(segs, FakeLLM(_json(_proposal(4, A))))
    apply_relabels(segs, result.mapping, speech)
    assert speech[4].speaker == A
    assert all(s.audio is marker for s in speech)
    assert [s.speaker for i, s in enumerate(speech) if i != 4] == [
        t.speaker for i, t in enumerate(segs) if i != 4]


def test_applying_twice_keeps_the_first_original_label():
    segs = _conversation(wrong={4})
    result = _relabel(segs, FakeLLM(_json(_proposal(4, A))))
    apply_relabels(segs, result.mapping)
    apply_relabels(segs, result.mapping)
    assert segs[4].speaker == A and segs[4].speaker_original == B


# --- guards ------------------------------------------------------------------

def _reasons(result):
    return [r["reason"] for r in result.rejected]


def test_a_label_the_file_does_not_have_is_rejected():
    result = _relabel(_conversation(wrong={4}),
                      FakeLLM(_json(_proposal(4, "SPEAKER_07"))))
    assert result.mapping == {} and "unknown_label" in _reasons(result)


def test_a_letter_is_matched_ignoring_case_and_padding():
    result = _relabel(_conversation(wrong={4}),
                      FakeLLM(_json(_proposal(4, " a "))))
    assert result.mapping == {"00004": A}


def test_the_diarizers_own_label_is_not_an_answer_the_model_may_give():
    """It was told letters; the raw label of a bare-digit file reads like a line number."""
    result = _relabel(_conversation(wrong={4}),
                      FakeLLM(json.dumps([{"i": 5, "speaker": A, "conf": 0.9}])))
    assert result.mapping == {} and "unknown_label" in _reasons(result)


def test_low_confidence_is_rejected():
    result = _relabel(_conversation(wrong={4}),
                      FakeLLM(_json(_proposal(4, A, conf=0.5))))
    assert result.mapping == {} and "low_confidence" in _reasons(result)


def test_a_missing_or_unusable_confidence_counts_as_low():
    for conf in (None, "cao", -1, 7):
        reply = json.dumps([{"i": 5, "speaker": "A", "conf": conf}])
        result = _relabel(_conversation(wrong={4}), FakeLLM(reply))
        assert result.mapping == {}, conf


def test_a_segment_from_overlap_separation_is_locked():
    for kwargs in ({"bss": True}, {"unseparated": [{"start": 1.0, "end": 2.0}]}):
        segs = _conversation(wrong={4})
        segs[4] = _seg(4, B, **kwargs)
        result = _relabel(segs, FakeLLM(_json(_proposal(4, A))))
        assert result.mapping == {} and "locked_segment" in _reasons(result), kwargs


def test_a_locked_segment_is_shown_but_marked():
    seen = []
    segs = _conversation()
    segs[3] = _seg(3, B, bss=True)
    _relabel(segs, FakeLLM(lambda m: seen.append(m) or "[]"))
    line = next(l for l in seen[0].splitlines() if l.startswith("[4] "))
    assert "[cố định]" in line


def test_a_line_that_is_not_in_the_window_is_rejected():
    for bad in (99999, 0, -3, 11, "đầu", 2.5):
        reply = json.dumps([{"i": bad, "speaker": "A", "conf": 0.9}])
        result = _relabel(_conversation(wrong={4}), FakeLLM(reply))
        assert result.mapping == {} and _reasons(result) == ["unknown_index"], bad


def test_proposing_the_label_it_already_has_is_a_no_change():
    result = _relabel(_conversation(), FakeLLM(_json(_proposal(4, A))))
    assert result.mapping == {} and "no_change" in _reasons(result)


def test_the_same_segment_twice_is_applied_once():
    reply = _json(_proposal(4, A), _proposal(4, A, conf=0.99))
    result = _relabel(_conversation(wrong={4}), FakeLLM(reply))
    assert result.mapping == {"00004": A} and "duplicate" in _reasons(result)


def test_the_ways_a_model_writes_the_same_line_all_name_it():
    for written in (5, "5", "#5", "[5]", 5.0, "dòng 5"):
        reply = json.dumps([{"i": written, "speaker": "A", "conf": 0.9}])
        result = _relabel(_conversation(wrong={4}), FakeLLM(reply))
        assert result.mapping == {"00004": A}, written


# --- the cap -----------------------------------------------------------------

def test_a_model_that_disagrees_with_most_of_the_file_is_discarded_whole():
    segs = _conversation(n=20)
    result = _relabel(segs, FakeLLM(_flip_everything_visible))
    assert result.discarded == "over_cap"
    assert result.mapping == {} and result.applied == []
    assert "over_cap" in _reasons(result)


def test_a_few_changes_are_under_the_cap():
    segs = _conversation(n=40, wrong={7, 22})
    reply = _json(_proposal(7, B), _proposal(22, A))
    result = _relabel(segs, FakeLLM(reply))
    assert result.discarded is None and set(result.mapping) == {"00007", "00022"}


def test_a_tiny_file_may_still_change_a_couple_of_segments():
    segs = _conversation(n=6, wrong={2})
    result = _relabel(segs, FakeLLM(_json(_proposal(2, A))))
    assert result.mapping == {"00002": A} and result.discarded is None


# --- skips -------------------------------------------------------------------

def test_a_single_speaker_file_is_skipped_without_calling_the_model():
    segs = [_seg(i, A) for i in range(6)]
    llm = FakeLLM("[]")
    result = _relabel(segs, llm)
    assert result.skipped == "single_speaker" and llm.calls == []


def test_an_unavailable_model_is_skipped_not_an_error():
    llm = FakeLLM("[]", loaded=False)
    result = _relabel(_conversation(), llm)
    assert result.skipped == "llm_unavailable" and llm.calls == []


def test_no_segments_is_skipped():
    assert _relabel([], FakeLLM("[]")).skipped == "no_segments"


# --- windows -----------------------------------------------------------------

def _small_windows(llm, overhead_lines=8):
    """A window budget of about `overhead_lines` lines for the FakeLLM's word count."""
    probe = SpeakerRelabelService(llm, window_tokens=10 ** 6)
    per_line = len(_conversation(1)[0].text.split()) + 4
    return probe.prompt_overhead([A, B]) + per_line * overhead_lines


def test_a_long_transcript_is_read_in_several_windows():
    llm = FakeLLM("[]")
    segs = _conversation(n=30)
    result = _relabel(segs, llm, window_tokens=_small_windows(llm),
                      overlap_segments=4)
    assert result.windows > 1
    assert sum(len(c) for c in llm.calls) == result.windows


def test_each_segment_is_decided_by_the_one_window_that_owns_it():
    llm = FakeLLM(_flip_everything_visible)
    segs = _conversation(n=30)
    result = _relabel(segs, llm, window_tokens=_small_windows(llm),
                      overlap_segments=4, max_change_fraction=1.0)
    assert result.windows > 1
    assert sorted(a["index"] for a in result.applied) == [s.index for s in segs]
    assert "outside_window_core" in _reasons(result)


def test_windows_share_context_but_not_decisions():
    """The overlap shows a segment on both sides of a boundary; one window decides."""
    llm = FakeLLM(_flip_everything_visible)
    segs = _conversation(n=30)
    result = _relabel(segs, llm, window_tokens=_small_windows(llm),
                      overlap_segments=6, max_change_fraction=1.0)
    indexes = [a["index"] for a in result.applied]
    assert len(indexes) == len(set(indexes))


def test_windows_are_batched_up_to_the_models_token_limit():
    llm = FakeLLM("[]", max_batch_tokens=10 ** 6)
    segs = _conversation(n=30)
    result = _relabel(segs, llm, window_tokens=_small_windows(llm),
                      overlap_segments=4)
    assert len(llm.calls) < result.windows


def test_a_batch_that_does_not_fit_is_halved_until_it_does():
    llm = FakeLLM(_flip_everything_visible, max_batch_tokens=10 ** 6, fail_multi=True)
    segs = _conversation(n=30)
    result = _relabel(segs, llm, window_tokens=_small_windows(llm),
                      overlap_segments=4, max_change_fraction=1.0)
    assert result.failed_windows == 0
    assert len(result.applied) == len(segs)


def test_a_window_the_model_cannot_answer_leaves_labels_alone():
    result = _relabel(_conversation(wrong={4}), FakeLLM("[]", fail_all=True))
    assert result.failed_windows == result.windows >= 1
    assert result.mapping == {}


# --- bookkeeping -------------------------------------------------------------

def test_the_report_says_what_changed_and_what_was_refused():
    segs = _conversation(n=40, wrong={7})
    reply = _json(_proposal(7, B), _proposal(9, "SPEAKER_09"))
    report = _relabel(segs, FakeLLM(reply)).to_report()
    assert report["prompt_version"] == RELABEL_PROMPT_VERSION
    assert report["applied"][0]["index"] == "00007"
    assert report["applied"][0]["from"] == A and report["applied"][0]["to"] == B
    assert report["rejected"][0]["reason"] == "unknown_label"
    assert report["rejected"][0]["line"] == 10 and report["rejected"][0]["index"] == "00009"
    assert report["applied"][0]["line"] == 8
    assert report["names"] == {"A": A, "B": B}
    assert report["segments"] == 40 and report["changed"] == 1


def test_the_checkpoint_namespace_names_prompt_and_model_and_is_a_safe_path():
    ns = SpeakerRelabelService(FakeLLM()).checkpoint_namespace
    assert RELABEL_PROMPT_VERSION in ns and "fake" in ns
    assert "/" not in ns and " " not in ns


@pytest.mark.parametrize("bad", [
    {"min_confidence": 0.7, "surprise": 1},
])
def test_an_unknown_setting_is_an_error_not_silently_dropped(bad):
    with pytest.raises(TypeError):
        SpeakerRelabelService(FakeLLM(), **bad)


# --- what the model actually said ----------------------------------------------

def test_a_reply_in_prose_is_counted_apart_from_an_empty_answer():
    prose = _relabel(_conversation(wrong={4}), FakeLLM("Không có đoạn nào sai nhãn."))
    empty = _relabel(_conversation(wrong={4}), FakeLLM("[]"))
    assert prose.unreadable_windows == prose.windows >= 1 and prose.proposed == 0
    assert empty.unreadable_windows == 0 and empty.proposed == 0
    assert prose.mapping == empty.mapping == {}


def test_every_window_keeps_the_raw_reply_and_how_many_proposals_it_held():
    reply = _json(_proposal(4, B), _proposal(6, A))
    result = _relabel(_conversation(wrong={4, 6}), FakeLLM(reply))
    assert result.proposed == 2
    row = result.replies[0]
    assert row["raw"] == reply and row["proposals"] == 2 and row["readable"] is True
    assert (row["first_index"], row["last_index"], row["lines"]) == ("00000", "00009", 10)
    report = result.to_report()
    assert report["proposed"] == 2 and report["replies"] == result.replies


def test_a_window_the_model_could_not_be_run_on_has_no_reply_and_is_not_unreadable():
    result = _relabel(_conversation(wrong={4}), FakeLLM("[]", fail_all=True))
    row = result.replies[0]
    assert row["answered"] is False and row["raw"] is None
    assert result.failed_windows >= 1 and result.unreadable_windows == 0


def test_the_prompt_example_names_lines_no_window_has():
    prompt = relabel_system_prompt(0.7)
    assert '"i": 903' in prompt and "[903]" in prompt
    assert "{min_confidence}" not in prompt and "0.7" in prompt
    assert "SPEAKER_" not in prompt and "00012" not in prompt


def test_the_confidence_floor_in_the_prompt_is_the_one_the_guards_apply():
    llm = FakeLLM("[]")
    _relabel(_conversation(), llm, min_confidence=0.85)
    assert llm.system_prompts and all("0.85" in p for p in llm.system_prompts)


def test_a_copied_prompt_example_names_no_real_segment_and_changes_nothing():
    copied = _json({"i": 903, "speaker": "A", "conf": 0.9,
                    "why": "câu hỏi, người trả lời vừa nói xong"})
    result = _relabel(_conversation(wrong={4}), FakeLLM(copied))
    assert result.mapping == {} and "unknown_index" in _reasons(result)


# --- what the model is shown ---------------------------------------------------

def test_speakers_are_lettered_in_a_stable_order_whatever_the_diarizer_called_them():
    assert speaker_names(["3", "0", "2"]) == {"0": "A", "2": "B", "3": "C"}
    assert speaker_names(["SPEAKER_01", "SPEAKER_00"]) == {"SPEAKER_00": "A", "SPEAKER_01": "B"}


def test_the_model_reads_numbered_lines_and_letters_never_the_raw_labels():
    seen = []
    segs = [_seg(i, "3" if i % 2 else "0") for i in range(10)]
    _relabel(segs, FakeLLM(lambda m: seen.append(m) or "[]"))
    message = seen[0]
    assert "Người nói: A, B" in message and "Số dòng: 1 đến 10" in message
    lines = [l for l in message.splitlines() if re.match(r"\[\d+\] ", l)]
    assert len(lines) == 10 and lines[0].startswith("[1] 00:00.0-00:04.0 A ")
    assert lines[1].startswith("[2] 00:05.0-00:09.0 B ")
    assert "#0000" not in message


def test_a_letter_answer_is_turned_back_into_the_files_own_label():
    segs = [_seg(i, "3" if i % 2 else "0") for i in range(10)]
    segs[4] = _seg(4, "3")                       # wrong: should be "0"
    reply = json.dumps([{"i": 5, "speaker": "A", "conf": 0.9, "why": "x"}])
    result = _relabel(segs, FakeLLM(reply))
    assert result.mapping == {"00004": "0"} and result.names == {"A": "0", "B": "3"}


def test_every_window_is_numbered_from_one_and_a_line_maps_to_its_own_segment():
    llm = FakeLLM(_flip_everything_visible)
    segs = _conversation(n=30)
    probe = SpeakerRelabelService(llm, window_tokens=10 ** 6)
    per_line = len(segs[0].text.split()) + 4
    budget = probe.prompt_overhead(["A", "B"]) + per_line * 8
    result = _relabel(segs, llm, window_tokens=budget, overlap_segments=4,
                      max_change_fraction=1.0)
    assert result.windows > 2
    for message in (m for call in llm.calls for m in call):
        numbers = [int(n) for n in re.findall(r"^\[(\d+)\] ", message, re.M)]
        assert numbers == list(range(1, len(numbers) + 1))
    # Every applied change names the segment whose own line it was, in its window.
    assert all(a["index"] == f"{a['index']}" and a["line"] >= 1 for a in result.applied)
    assert sorted(a["index"] for a in result.applied) == [s.index for s in segs]
