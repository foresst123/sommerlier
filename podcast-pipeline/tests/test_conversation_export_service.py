"""The service that judges candidate clips with the LLM and writes the keepers.

The model is a gate: it scores a candidate and may name where to start and stop,
by segment index. What is pinned here is that it can only ever narrow what the
scan already accepted -- never widen a clip, invent a time, or pass one the scan
refused -- and that the audio and metadata on disk describe the clip honestly.

Run:  python -m pytest tests/test_conversation_export_service.py -q     (from podcast-pipeline/)
"""
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sf = pytest.importorskip("soundfile")

from services.conversation_export_service import (
    CONVERSATION_EXPORT_PROMPT_VERSION, ConversationExportService, _EXAMPLE_BAD, _EXAMPLE_GOOD,
    cut_excerpt, parse_verdict)
from utils.excise import TimelineMap
from utils.noise_map import KINDS, NoiseTrack

A, B = "SPEAKER_00", "SPEAKER_01"
SR = 8000


# --- builders ----------------------------------------------------------------

def _seg(i, speaker, start, end):
    return SimpleNamespace(
        index=f"{i:05d}", speaker=speaker, start=start, end=end,
        text=f"đoạn {i} bàn về chuyện số {i} rất dài dòng", bss=False,
        unseparated=None)


def _talk(n, start=0.0, seg=7.0, gap=1.0, first=0):
    out, t = [], start
    for k in range(n):
        out.append(_seg(first + k, (A, B)[k % 2], t, t + seg))
        t += seg + gap
    return out


def _quiet(total=600):
    return NoiseTrack({k: np.full(int(total * 100), 0.001, dtype=np.float32)
                       for k in KINDS}, fps=100.0)


def _wave(seconds=300.0):
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(seconds * SR)) * 0.1).astype(np.float32)


class FakeLLM:
    model_name = "fake/model"
    batch_size = 100
    max_batch_tokens = 0

    def __init__(self, reply, loaded=True):
        self.reply, self.loaded = reply, loaded
        self.messages = []
        self.system_prompts = []
        self.thinking_seen = []
        self.budgets = []

    def ensure_loaded(self):
        return self.loaded

    def count_tokens(self, text):
        return len(text.split())

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None, **extra):
        self.messages.extend(user_messages)
        self.system_prompts.append(system_prompt)
        self.thinking_seen.append(extra.get("thinking", False))
        self.budgets.append(max_new_tokens)
        return True, [self.reply(m) if callable(self.reply) else self.reply
                      for m in user_messages]


def _good(score=5, topic="chủ đề thử"):
    return json.dumps({"self_contained": score, "topic": topic})


def _run(tmp_path, llm, segs=None, noise="quiet", timeline=None, run_kwargs=None,
         logger=None, **settings):
    segs = segs if segs is not None else _talk(30)
    wave = _wave()
    svc = ConversationExportService(llm, logger=logger, **settings)
    result = svc.run(
        segs, timeline=timeline or TimelineMap(),
        noise=_quiet() if noise == "quiet" else NoiseTrack(),
        music_map=None, waveform=wave, sample_rate=SR,
        out_dir=str(tmp_path), base_name="ep 01", **(run_kwargs or {}))
    return svc, result, wave


def _folders(root):
    """The conversation folders under an export root: best tier first, in number order."""
    out = []
    for tier in sorted(p for p in root.glob("tier_*") if p.is_dir()):
        out.extend(sorted((c for c in tier.glob("conversation_*") if c.is_dir()),
                          key=lambda c: int(c.name.rsplit("_", 1)[1])))
    return out


def _doc_of(folder):
    return json.loads((folder / "conversation.json").read_text(encoding="utf-8"))


def _only(root):
    """(folder, conversation.json) of the one conversation that was written."""
    (folder,) = _folders(root)
    return folder, _doc_of(folder)


# --- reading the model's verdict -------------------------------------------------

def test_a_plain_verdict_is_parsed():
    v = parse_verdict('{"topic": "làm bếp", "reason": "mở và kết tự nhiên", '
                      '"self_contained": 4, "start_line": 3, "end_line": 9}')
    assert v == {"self_contained": 4, "topic": "làm bếp", "reason": "mở và kết tự nhiên",
                 "start_line": 3, "end_line": 9, "trim_garbled": False}


def test_a_fenced_reply_with_reasoning_is_parsed():
    raw = "<think>hmm</think>```json\n{\"self_contained\": 5}\n```"
    assert parse_verdict(raw)["self_contained"] == 5


def test_a_score_outside_one_to_five_or_unreadable_is_no_score():
    for bad in (0, 6, "cao", None, True, -3):
        assert parse_verdict(json.dumps({"self_contained": bad}))["self_contained"] is None


def test_a_reply_with_no_json_is_no_verdict():
    assert parse_verdict("Đoạn này khá ổn.") is None
    assert parse_verdict("") is None


def test_unknown_keys_in_the_verdict_are_dropped():
    v = parse_verdict(json.dumps({"self_contained": 4, "start": 1.0, "text": "bịa"}))
    assert set(v) == {"self_contained", "topic", "reason", "start_line", "end_line",
                      "trim_garbled"}


def test_a_trim_written_in_any_of_the_usual_ways_is_the_same_line():
    for start in (3, 3.0, "3", "#3", "[3]", "dòng 3"):
        v = parse_verdict(json.dumps({"self_contained": 4, "start_line": start}))
        assert (v["start_line"], v["trim_garbled"]) == (3, False), start


def test_no_trim_is_null_or_absent_or_empty_and_is_not_garbled():
    for body in ({}, {"start_line": None, "end_line": None},
                 {"start_line": "", "end_line": "null"}):
        v = parse_verdict(json.dumps({"self_contained": 4, **body}))
        assert (v["start_line"], v["end_line"], v["trim_garbled"]) == (None, None, False), body


def test_a_trim_that_names_no_line_is_garbled_not_absent():
    for bad in ("đầu", 0, -2, 1.5, True, "hết"):
        v = parse_verdict(json.dumps({"self_contained": 4, "start_line": bad}))
        assert v["trim_garbled"] is True and v["start_line"] is None, bad


# --- what the model sees ------------------------------------------------------------

def test_the_model_sees_numbered_turns_as_a_and_b_without_raw_labels_or_segment_ids(tmp_path):
    llm = FakeLLM(_good())
    _run(tmp_path, llm, max_candidates=2)
    assert llm.messages
    message = llm.messages[0]
    assert "SPEAKER_" not in message and "#0" not in message
    assert re.search(r"^\[1\] 0:00 [AB]: ", message, re.M)
    assert re.search(r"^\[2\] \d+:\d{2} [AB]: ", message, re.M)
    assert re.search(r"dài \d+ giây, \d+ dòng", message)
    numbers = [int(n) for n in re.findall(r"^\[(\d+)\] ", message, re.M)]
    assert numbers == list(range(1, len(numbers) + 1))


def test_only_the_shortlist_is_sent_to_the_model(tmp_path):
    llm = FakeLLM(_good())
    _, result, _ = _run(tmp_path, llm, max_candidates=3)
    assert len(llm.messages) <= 3
    assert result.report["shortlisted"] == len(llm.messages)


# --- accepting and rejecting -----------------------------------------------------------

def test_a_self_contained_clip_is_written_as_a_folder_of_audio_and_metadata(tmp_path):
    llm = FakeLLM(_good(5, "chuyện nấu ăn"))
    _, result, wave = _run(tmp_path, llm)
    assert result.exports
    folders = _folders(tmp_path)
    assert folders and len(folders) == len(result.exports)
    assert folders[0].name == "conversation_1"

    info = sf.info(str(folders[0] / "mixture.wav"))
    assert info.samplerate == SR and info.channels == 1
    doc = _doc_of(folders[0])
    assert doc["id"].startswith("ep_01_conversation_000001")
    assert doc["topic"] == "chuyện nấu ăn" and doc["semantic_score"] == 5
    assert doc["speakers"] == ["A", "B"]
    assert set(doc["speaker_ids"].values()) == {A, B}
    assert 60.0 <= doc["source_end"] - doc["source_start"] <= 241.0
    assert info.duration == pytest.approx(doc["duration"], abs=0.01)
    assert doc["conversation"][0]["start"] >= 0.0
    assert {row["speaker"] for row in doc["conversation"]} == {"A", "B"}
    assert doc["noise"]["dominant_kind"] == "clean"
    assert doc["orig_spans"]


def test_clean_clip_metadata_keeps_final_word_times_relative_to_written_audio(tmp_path):
    segs = _talk(30)
    by_index = {seg.index: seg for seg in segs}
    for seg in segs:
        tokens = seg.text.split()
        width = (seg.end - seg.start) / len(tokens)
        seg.words = [
            {"word": token,
             "start": seg.start + number * width,
             "end": seg.start + (number + 1) * width,
             "score": 0.9}
            for number, token in enumerate(tokens)
        ]

    _, result, _ = _run(
        tmp_path, FakeLLM(_good()), segs=segs, require_word_alignment=True)
    assert result.exports
    doc = _doc_of(_folders(tmp_path)[0])
    row = doc["conversation"][0]
    source = by_index[row["index"]]
    assert [word["word"] for word in row["words"]] == source.text.split()
    assert row["words"][0]["start"] == pytest.approx(
        source.words[0]["start"] - doc["source_start"], abs=0.001)


def test_a_low_score_rejects_the_clip(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good(3)))
    assert result.exports == []
    assert result.report["judged_rejected"]["not_self_contained"] >= 1
    assert _folders(tmp_path) == []


def test_no_answer_is_not_a_yes_unless_the_requirement_is_off(tmp_path):
    _, strict, _ = _run(tmp_path / "strict", FakeLLM("Đoạn này ổn."))
    assert strict.exports == []
    assert strict.report["judged_rejected"]["no_verdict"] >= 1

    _, lax, _ = _run(tmp_path / "lax", FakeLLM("Đoạn này ổn."), require_semantic=False)
    assert lax.exports
    assert all(c["semantic_score"] is None for c in lax.exports)


def test_an_unavailable_model_skips_the_pass_or_passes_unjudged_if_allowed(tmp_path):
    llm = FakeLLM(_good(), loaded=False)
    _, strict, _ = _run(tmp_path / "strict", llm)
    assert strict.exports == [] and strict.report["skipped"] == "llm_unavailable"

    _, lax, _ = _run(tmp_path / "lax", FakeLLM(_good(), loaded=False),
                     require_semantic=False)
    assert lax.exports


def test_unmeasured_noise_writes_nothing_and_never_asks_the_model(tmp_path):
    llm = FakeLLM(_good())
    _, result, _ = _run(tmp_path, llm, noise="none")
    assert result.exports == [] and llm.messages == []
    assert result.report["skipped"] == "noise_not_measured"


def test_clips_that_are_kept_do_not_overlap_each_other(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good()), segs=_talk(60), max_candidates=40)
    spans = sorted((c["source_start"], c["source_end"]) for c in result.exports)
    assert len(spans) > 1
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        # Padding may share a breath of silence; the turns themselves never do.
        assert prev_end - next_start <= 0.5


# --- the model can only narrow a clip --------------------------------------------------

def _trim(first, last):
    """Reply that asks to keep only lines `first`..`last` of whatever it is shown."""
    def reply(message):
        numbers = [int(n) for n in re.findall(r"^\[(\d+)\] ", message, re.M)]
        body = {"topic": "t", "reason": "r", "self_contained": 5}
        if first is not None:
            body["start_line"] = numbers[first]
        if last is not None:
            body["end_line"] = numbers[last]
        return json.dumps(body)
    return reply


def _only_row(result):
    (row,) = result.report["replies"]
    return row


def test_a_trim_inside_the_candidate_shortens_it_and_is_marked(tmp_path):
    # Drop the first and the last line of whatever candidate is shown.
    _, result, _ = _run(tmp_path, FakeLLM(_trim(1, -2)), max_candidates=1)
    assert result.exports and result.exports[0]["trimmed_by_model"] is True
    assert _only_row(result)["trim_ignored"] is None


def test_a_trim_keeps_exactly_the_lines_it_names(tmp_path):
    """Line 2 to the one before the last: the excerpt is those lines, no more, no fewer."""
    shown = {}

    def reply(message):
        shown["rows"] = re.findall(r"^\[(\d+)\] \d+:\d{2} [AB]: (.*)$", message, re.M)
        return json.dumps({"topic": "t", "self_contained": 5,
                           "start_line": int(shown["rows"][1][0]),
                           "end_line": int(shown["rows"][-2][0])})
    _run(tmp_path, FakeLLM(reply), max_candidates=1)
    _, meta = _only(tmp_path)
    assert [row["text"] for row in meta["conversation"]] == [
        text for _, text in shown["rows"][1:-1]]


def test_a_trim_that_leaves_less_than_the_minimum_is_ignored_and_the_excerpt_kept_whole(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_trim(0, 3)), max_candidates=1)
    assert result.exports and result.exports[0]["trimmed_by_model"] is False
    assert _only_row(result)["trim_ignored"] == "trim_duration"
    assert result.report["judged_rejected"] == {}


def test_a_trim_to_a_line_that_does_not_exist_is_ignored_and_the_excerpt_kept_whole(tmp_path):
    def outside(message):
        return json.dumps({"topic": "t", "self_contained": 5,
                           "start_line": 99999, "end_line": 88888})
    _, result, _ = _run(tmp_path, FakeLLM(outside), max_candidates=1)
    assert result.exports and result.exports[0]["trimmed_by_model"] is False
    assert _only_row(result)["trim_ignored"] == "trim_out_of_range"
    assert result.report["judged_rejected"] == {}
    assert result.report["trims_ignored"] == {"trim_out_of_range": 1}


def test_a_trim_back_to_front_is_ignored(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_trim(-3, 2)), max_candidates=1)
    assert result.exports
    assert _only_row(result)["trim_ignored"] == "trim_out_of_range"


def test_a_trim_that_names_no_line_is_ignored(tmp_path):
    def prose(message):
        return json.dumps({"topic": "t", "self_contained": 5, "start_line": "đầu"})
    _, result, _ = _run(tmp_path, FakeLLM(prose), max_candidates=1)
    assert result.exports and _only_row(result)["trim_ignored"] == "trim_unreadable"


def test_a_trim_cannot_widen_a_clip_past_the_candidate(tmp_path):
    """One line past the last is not in the candidate, whatever the recording holds."""
    def widen(message):
        numbers = [int(n) for n in re.findall(r"^\[(\d+)\] ", message, re.M)]
        return json.dumps({"topic": "t", "self_contained": 5, "end_line": numbers[-1] + 1})
    _, result, _ = _run(tmp_path, FakeLLM(widen), max_candidates=1)
    assert result.exports and result.exports[0]["trimmed_by_model"] is False
    assert _only_row(result)["trim_ignored"] == "trim_out_of_range"


def test_null_trims_are_no_trim_and_nothing_is_ignored(tmp_path):
    reply = json.dumps({"topic": "t", "self_contained": 5, "start_line": None, "end_line": None})
    _, result, _ = _run(tmp_path, FakeLLM(reply), max_candidates=1)
    assert result.exports and result.exports[0]["trimmed_by_model"] is False
    assert _only_row(result)["trim_ignored"] is None and result.report["trims_ignored"] == {}


def test_an_ignored_trim_is_written_into_the_metadata(tmp_path):
    def outside(message):
        return json.dumps({"topic": "t", "reason": "vì sao", "self_contained": 5,
                           "start_line": 99999})
    _run(tmp_path, FakeLLM(outside), max_candidates=1)
    _, meta = _only(tmp_path)
    assert meta["trim_ignored"] == "trim_out_of_range" and meta["semantic_reason"] == "vì sao"


# --- the prompt and its examples --------------------------------------------------

def test_the_prompt_is_filled_in_with_this_runs_thresholds(tmp_path):
    llm = FakeLLM(_good())
    _run(tmp_path, llm, max_candidates=1, min_semantic=4, min_seconds=60)
    prompt = llm.system_prompts[0]
    assert not re.findall(r"\{[a-z_]+\}", prompt)
    assert "từ 4 trở lên" in prompt and "ít nhất 60 giây" in prompt


def test_the_examples_in_the_prompt_trim_nothing_so_copying_them_cannot_cut_anything(tmp_path):
    llm = FakeLLM(_good())
    _run(tmp_path, llm, max_candidates=1)
    for example in (_EXAMPLE_GOOD, _EXAMPLE_BAD):
        assert example["start_line"] is None and example["end_line"] is None
        assert json.dumps(example, ensure_ascii=False) in llm.system_prompts[0]
    assert "00012" not in llm.system_prompts[0]
    assert '"topic": "tối đa 10 từ"' not in llm.system_prompts[0]


def test_a_verdict_that_is_the_prompts_own_example_is_refused(tmp_path):
    for example in (_EXAMPLE_GOOD, _EXAMPLE_BAD):
        reply = json.dumps(example, ensure_ascii=False)
        _, result, _ = _run(tmp_path, FakeLLM(reply), max_candidates=1)
        assert result.exports == []
        assert result.report["judged_rejected"] == {"copied_example": 1}
        assert _only_row(result)["outcome"] == "copied_example"


def test_a_copied_example_is_recognised_however_it_is_spaced_or_capitalised(tmp_path):
    reply = json.dumps({"topic": "  Cách nấu canh chua cho người mới. ", "self_contained": 5})
    _, result, _ = _run(tmp_path, FakeLLM(reply), max_candidates=1)
    assert result.exports == [] and result.report["judged_rejected"] == {"copied_example": 1}


# --- the audio -------------------------------------------------------------------------

def test_the_excerpt_is_the_recording_between_its_times_and_the_source_is_untouched():
    wave = _wave(60.0)
    before = wave.copy()
    plain = cut_excerpt(wave, SR, 10.0, 20.0, fade_ms=0, zero_cross_ms=0)
    np.testing.assert_array_equal(plain, wave[10 * SR:20 * SR])

    faded = cut_excerpt(wave, SR, 10.0, 20.0, fade_ms=50, zero_cross_ms=0)
    away_from_the_fades = slice(SR, -SR)
    np.testing.assert_array_equal(faded[away_from_the_fades],
                                  wave[10 * SR:20 * SR][away_from_the_fades])
    assert faded.dtype == np.float32
    np.testing.assert_array_equal(wave, before)


def test_an_excerpt_fades_in_and_out_so_it_cannot_click(tmp_path):
    wave = np.full(10 * SR, 0.5, dtype=np.float32)
    excerpt = cut_excerpt(wave, SR, 1.0, 9.0, fade_ms=50, zero_cross_ms=0)
    assert abs(excerpt[0]) < 1e-3 and abs(excerpt[-1]) < 1e-3
    assert excerpt[len(excerpt) // 2] == pytest.approx(0.5)
    fade = int(0.05 * SR)
    assert excerpt[fade // 2] < 0.5 and excerpt[-fade // 2] < 0.5


def test_an_excerpt_starts_and_stops_near_a_zero_crossing(tmp_path):
    # 200 Hz crosses zero every 2.5 ms, so a 5 ms search always finds one; real
    # voiced speech is at least that busy.
    t = np.arange(20 * SR) / SR
    wave = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)
    excerpt = cut_excerpt(wave, SR, 3.0011, 9.0033, fade_ms=0, zero_cross_ms=5)
    assert abs(excerpt[0]) < 0.1 and abs(excerpt[-1]) < 0.1
    unsnapped = cut_excerpt(wave, SR, 3.0011, 9.0033, fade_ms=0, zero_cross_ms=0)
    assert abs(unsnapped[0]) > 0.1 or abs(unsnapped[-1]) > 0.1


def test_a_span_past_the_end_of_the_audio_is_clamped(tmp_path):
    wave = _wave(10.0)
    excerpt = cut_excerpt(wave, SR, 8.0, 15.0, fade_ms=0, zero_cross_ms=0)
    assert len(excerpt) == 2 * SR
    assert len(cut_excerpt(wave, SR, 20.0, 25.0, fade_ms=0, zero_cross_ms=0)) == 0


def test_the_written_wav_is_the_cut_excerpt(tmp_path):
    _, result, wave = _run(tmp_path, FakeLLM(_good()), max_candidates=1)
    assert result.exports
    folder, doc = _only(tmp_path)
    data, sr = sf.read(str(folder / doc["files"]["mixture"]))
    assert sr == SR
    expected = cut_excerpt(wave, SR, doc["source_start"], doc["source_end"], 50, 5)
    assert len(data) == pytest.approx(len(expected), abs=int(0.02 * SR))


# --- housekeeping ------------------------------------------------------------------------

def test_an_unknown_setting_is_an_error(tmp_path):
    with pytest.raises(TypeError):
        ConversationExportService(FakeLLM(_good()), min_secs=30)


def test_the_report_names_the_prompt_version_and_counts(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good()))
    report = result.report
    assert report["prompt_version"] == CONVERSATION_EXPORT_PROMPT_VERSION
    assert report["exported"] == len(result.exports)
    assert report["accepted"] >= report["exported"]
    assert "finder" in report and report["finder"]["noise_measured"] is True


# --- the two-channel file ---------------------------------------------------------
#
# Beside the mono recording, one more file of the same length: the first speaker
# in the left ear, the second in the right. It is a listening layout over one
# microphone, so the mono stays the recording of record.

from services.conversation_export_service import fade_edges, gate_channels  # noqa: E402

FADE_MS = 10.0


def _gate(mixture, left, right, offset=0, margin_ms=30.0, sr=SR):
    return gate_channels(mixture, sr, offset, left, right, margin_ms, FADE_MS)


def test_each_speaker_is_heard_only_in_their_own_ear():
    n = 10 * SR
    mix = np.full(n, 0.5, dtype=np.float32)
    out = _gate(mix, left=[(1.0, 3.0)], right=[(5.0, 7.0)])
    assert out.shape == (n, 2) and out.dtype == np.float32
    at = lambda t: int(t * SR)
    assert out[at(2.0), 0] == pytest.approx(0.5) and out[at(2.0), 1] == 0.0
    assert out[at(6.0), 1] == pytest.approx(0.5) and out[at(6.0), 0] == 0.0
    assert not out[at(3.5):at(4.5)].any()                 # nobody speaks: both shut


def test_where_only_one_person_speaks_the_channel_is_the_recording_itself():
    rng = np.random.default_rng(1)
    mix = rng.standard_normal(10 * SR).astype(np.float32)
    out = _gate(mix, left=[(1.0, 3.0)], right=[(5.0, 7.0)])
    inside = slice(int(1.1 * SR), int(2.9 * SR))
    np.testing.assert_array_equal(out[inside, 0], mix[inside])


def test_when_both_speak_at_once_both_ears_carry_the_mixture():
    mix = np.full(10 * SR, 0.5, dtype=np.float32)
    out = _gate(mix, left=[(1.0, 4.0)], right=[(3.0, 6.0)])
    both = int(3.5 * SR)
    assert out[both, 0] == pytest.approx(0.5) and out[both, 1] == pytest.approx(0.5)


def test_a_channel_stays_open_a_little_past_a_turn_but_never_into_the_others():
    mix = np.full(10 * SR, 1.0, dtype=np.float32)
    # Left ends at 3.0 with lots of room; right starts at 3.01, 10 ms later.
    out = _gate(mix, left=[(1.0, 3.0)], right=[(3.01, 5.0)], margin_ms=30.0)
    assert out[int(3.005 * SR), 0] > 0.0                  # the margin, partly used
    assert out[int(3.02 * SR), 0] == 0.0                  # not into the other's turn
    assert out[int(2.99 * SR), 1] == 0.0                  # right does not reach back


def test_a_turn_that_starts_inside_the_others_gets_no_margin_on_that_side():
    mix = np.full(10 * SR, 1.0, dtype=np.float32)
    out = _gate(mix, left=[(2.0, 6.0)], right=[(1.0, 3.0)], margin_ms=100.0)
    assert out[int(1.95 * SR), 0] == 0.0                  # left did not widen backwards


def test_the_gate_opens_and_closes_without_a_click():
    mix = np.ones(10 * SR, dtype=np.float32)
    out = _gate(mix, left=[(2.0, 4.0)], right=[], margin_ms=0.0)
    gain = out[:, 0]
    assert gain.min() >= 0.0 and gain.max() <= 1.0
    step = np.abs(np.diff(gain)).max()
    assert step <= 1.0 / (FADE_MS / 1000.0 * SR) + 1e-6   # a ramp, never a jump


def test_a_speaker_with_no_turns_has_a_silent_channel():
    out = _gate(np.ones(3 * SR, dtype=np.float32), left=[(0.5, 1.5)], right=[])
    assert not out[:, 1].any() and out[:, 0].any()


def test_the_offset_places_turns_correctly_in_an_excerpt_cut_from_later_in_the_recording():
    excerpt = np.ones(4 * SR, dtype=np.float32)            # the recording from 100.0 s
    out = gate_channels(excerpt, SR, 100 * SR, [(101.0, 102.0)], [(102.5, 103.5)], 0.0, FADE_MS)
    assert out[int(1.5 * SR), 0] == 1.0 and out[int(1.5 * SR), 1] == 0.0
    assert out[int(3.0 * SR), 1] == 1.0 and out[int(3.0 * SR), 0] == 0.0


def test_both_ends_of_a_two_channel_clip_fade_like_the_mono():
    stereo = np.ones((2 * SR, 2), dtype=np.float32)
    fade_edges(stereo, SR, 50)
    assert abs(stereo[0]).max() < 1e-3 and abs(stereo[-1]).max() < 1e-3
    assert stereo[SR].tolist() == [1.0, 1.0]


def _read(path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, sr


def _first_clip(tmp_path):
    """(folder, conversation.json) of the first conversation, best tier first."""
    folder = _folders(tmp_path)[0]
    return folder, _doc_of(folder)


def test_the_service_writes_the_speaker_files_and_the_stereo_beside_the_mixture(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good()))
    assert result.exports
    folder, doc = _first_clip(tmp_path)
    mono, sr_m = _read(folder / "mixture.wav")
    both, sr_s = _read(folder / "stereo_2ch.wav")
    spk_a, _ = _read(folder / "speaker_A.wav")
    spk_b, _ = _read(folder / "speaker_B.wav")
    assert mono.shape[1] == spk_a.shape[1] == spk_b.shape[1] == 1 and both.shape[1] == 2
    assert sr_m == sr_s == SR
    assert len(mono) == len(both) == len(spk_a) == len(spk_b)     # sample for sample
    assert len(both) / SR == pytest.approx(doc["duration"], abs=0.01)
    assert result.exports[0]["audio_2ch"] == f"{doc['folder']}/stereo_2ch.wav"


def test_speaker_a_is_the_left_ear_and_b_the_right_and_the_mixture_is_unchanged(tmp_path):
    _run(tmp_path, FakeLLM(_good()))
    folder, doc = _first_clip(tmp_path)
    mono, _ = _read(folder / "mixture.wav")
    both, _ = _read(folder / "stereo_2ch.wav")
    assert doc["channels_2ch"]["left"] == "A" and doc["channels_2ch"]["right"] == "B"
    checked = {"A": 0, "B": 0}
    for row in doc["conversation"]:
        lo, hi = int((row["start"] + 0.3) * SR), int((row["end"] - 0.3) * SR)
        if hi <= lo or lo < 0 or hi > len(both):
            continue
        mine, other = (0, 1) if row["speaker"] == "A" else (1, 0)
        np.testing.assert_allclose(both[lo:hi, mine], mono[lo:hi, 0], atol=1e-4)
        assert not np.abs(both[lo:hi, other]).max() > 1e-4
        checked[row["speaker"]] += 1
    assert checked["A"] >= 2 and checked["B"] >= 2


def test_the_stereo_channels_are_the_speaker_files_sample_for_sample(tmp_path):
    _run(tmp_path, FakeLLM(_good()))
    folder, _ = _first_clip(tmp_path)
    both, _ = sf.read(str(folder / "stereo_2ch.wav"), dtype="int16", always_2d=True)
    a, _ = sf.read(str(folder / "speaker_A.wav"), dtype="int16")
    b, _ = sf.read(str(folder / "speaker_B.wav"), dtype="int16")
    assert np.array_equal(both[:, 0], a) and np.array_equal(both[:, 1], b)


def test_the_two_channel_file_starts_and_ends_silent_like_the_mono(tmp_path):
    _run(tmp_path, FakeLLM(_good()))
    folder, _ = _first_clip(tmp_path)
    both, _ = _read(folder / "stereo_2ch.wav")
    assert np.abs(both[0]).max() < 1e-3 and np.abs(both[-1]).max() < 1e-3


def test_the_speakers_in_the_two_channel_file_match_the_metadata_ids(tmp_path):
    _run(tmp_path, FakeLLM(_good()))
    _, doc = _first_clip(tmp_path)
    assert doc["speaker_ids"] == {"A": A, "B": B} or doc["speaker_ids"] == {"A": B, "B": A}
    names = {row["speaker"]: row["speaker_id"] for row in doc["conversation"]}
    assert names == doc["speaker_ids"]


def test_each_clip_has_a_plain_text_transcript_naming_both_speakers(tmp_path):
    _run(tmp_path, FakeLLM(_good()))
    folder, doc = _first_clip(tmp_path)
    text = (folder / doc["transcript"]).read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l]
    assert len(lines) == len(doc["conversation"])
    assert any("] A: " in l for l in lines) and any("] B: " in l for l in lines)
    assert lines[0].startswith("[00:")
    assert doc["conversation"][0]["text"] in lines[0]


def test_switching_stereo_off_writes_only_the_mixture(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good()), stereo=False)
    assert result.exports
    folder, doc = _first_clip(tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == [
        "conversation.json", "mixture.wav", "transcript.txt"]
    assert set(doc["files"]) == {"mixture"} and "channels_2ch" not in doc
    assert result.exports[0]["audio_2ch"] is None


# --- what the model actually said ---------------------------------------------------

def test_every_reply_is_kept_beside_the_verdict_made_of_it(tmp_path):
    reply = json.dumps({"topic": "t", "reason": "vì sao", "self_contained": 5,
                        "start_line": 99999, "end_line": 88888})
    _, result, _ = _run(tmp_path, FakeLLM(reply), max_candidates=1)
    row = result.report["replies"][0]
    assert row["raw"] == reply and row["readable"] is True
    assert row["verdict"]["start_line"] == 99999 and row["verdict"]["reason"] == "vì sao"
    assert row["outcome"] == "accepted" and row["trimmed"] is False
    assert row["trim_ignored"] == "trim_out_of_range"
    assert re.fullmatch(r"\d{5}", row["first_index"]) and re.fullmatch(r"\d{5}", row["last_index"])
    assert result.report["unreadable"] == 0


def test_an_accepted_candidate_says_so_in_its_row(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good()), max_candidates=1)
    assert result.exports
    assert [r["outcome"] for r in result.report["replies"]] == ["accepted"]


def test_a_reply_in_prose_is_counted_as_unreadable(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM("Đoạn này khá ổn."), max_candidates=1)
    row = result.report["replies"][0]
    assert row["answered"] is True and row["readable"] is False and row["verdict"] is None
    assert row["outcome"] == "no_verdict"
    assert result.report["unreadable"] == 1


def test_an_unavailable_model_leaves_rows_with_no_reply(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good(), loaded=False), max_candidates=1,
                        require_semantic=False)
    assert all(r["raw"] is None and r["answered"] is False
               for r in result.report["replies"])
    assert result.report["unreadable"] == 0


# --- thinking ---------------------------------------------------------------------

def test_thinking_is_off_by_default_and_the_model_is_called_as_before(tmp_path):
    llm = FakeLLM(_good())
    _, result, _ = _run(tmp_path, llm, max_candidates=1)
    assert llm.thinking_seen == [False] and llm.budgets == [256]
    assert result.report["thinking"] is False


def test_thinking_is_passed_on_and_the_budget_covers_the_reasoning(tmp_path):
    llm = FakeLLM(_good())
    _, result, _ = _run(tmp_path, llm, max_candidates=1, thinking=True)
    assert llm.thinking_seen == [True] and llm.budgets == [2048]
    assert result.report["thinking"] is True


def test_a_larger_configured_budget_is_kept_when_thinking(tmp_path):
    llm = FakeLLM(_good())
    _run(tmp_path, llm, max_candidates=1, thinking=True, max_new_tokens=3072)
    assert llm.budgets == [3072]


def test_the_reasoning_is_stripped_and_the_verdict_after_it_is_read(tmp_path):
    reply = ('<think>Dòng đầu là câu hỏi, dòng cuối khép ý. Có thể {"self_contained": 1}.'
             '</think>\n' + _good(score=5, topic="chuyện đổi nghề"))
    _, result, _ = _run(tmp_path, FakeLLM(reply), max_candidates=1, thinking=True)
    assert result.exports and result.exports[0]["semantic_score"] == 5
    assert result.report["cut_in_thought"] == 0


def test_a_reply_that_ends_inside_its_reasoning_is_counted_and_marked(tmp_path):
    reply = "<think>Đoạn này bắt đầu giữa chừng một câu chuyện, nhưng"
    _, result, _ = _run(tmp_path, FakeLLM(reply), max_candidates=1, thinking=True)
    row = result.report["replies"][0]
    assert result.exports == [] and row["cut_in_thought"] is True and row["readable"] is False
    assert row["outcome"] == "no_verdict"
    assert result.report["cut_in_thought"] == 1 and result.report["unreadable"] == 1


# --- the two-channel file from the separated speaker tracks ---------------------------

class Logger:
    def __init__(self):
        self.warnings, self.infos = [], []

    def warning(self, msg):
        self.warnings.append(msg)

    def info(self, msg):
        self.infos.append(msg)


class FakeSeparation:
    """Lays out one constant per speaker, so which track ended up in which ear is visible."""

    def __init__(self, left=0.25, right=-0.25, fail=None):
        self.left, self.right, self.fail = left, right, fail
        self.calls = []

    def export_sdlm_dual_channel(self, speech_segments, audio_duration, sr, strict=True,
                                 speakers=None, time_range=None, log_stats=True):
        self.calls.append({"strict": strict, "speakers": speakers, "time_range": time_range,
                           "duration": audio_duration, "sr": sr})
        if self.fail is not None:
            raise self.fail
        n = round(time_range[1] * sr) - round(time_range[0] * sr)
        return (np.full(n, self.left, dtype=np.float32),
                np.full(n, self.right, dtype=np.float32))


def _speech(speaker, start=0.0, end=1.0, **kw):
    """One segment as the separation stage leaves it."""
    return SimpleNamespace(speaker=speaker, start=start, end=end, audio=None,
                           bss_spans=[], bss_failed_spans=[], **kw)


def _tracks(tmp_path, **kw):
    """Run once with separation output at hand; return (result, metadata, mono, stereo)."""
    fake = kw.pop("fake", None) or FakeSeparation()
    logger = kw.pop("logger", None)
    _, result, _ = _run(tmp_path, FakeLLM(_good()), segs=kw.pop("segs", None),
                        max_candidates=1, logger=logger,
                        run_kwargs={"speech_segments": kw.pop("speech_segments", None)
                                    or [_speech(A)], "separation_service": fake})
    folder, meta = _only(tmp_path)
    mono, _ = sf.read(str(folder / "mixture.wav"), dtype="float32")
    stereo, _ = sf.read(str(folder / "stereo_2ch.wav"), dtype="float32")
    return result, meta, mono, stereo, fake


def test_each_ear_carries_that_speakers_separated_track(tmp_path):
    _, meta, mono, stereo, fake = _tracks(tmp_path)
    mid = len(stereo) // 2
    assert stereo[mid, 0] == pytest.approx(0.25, abs=1e-3)
    assert stereo[mid, 1] == pytest.approx(-0.25, abs=1e-3)
    assert meta["channels_2ch"]["method"] == "strict_separation_tracks"
    assert "separated" in meta["channels_2ch"]["note"]


def test_the_two_channel_file_is_exactly_as_long_as_the_mono(tmp_path):
    _, _, mono, stereo, _ = _tracks(tmp_path)
    assert stereo.shape == (len(mono), 2)


def test_the_tracks_are_asked_for_strictly_for_this_pair_over_exactly_the_cut(tmp_path):
    _, meta, mono, _, fake = _tracks(tmp_path)
    # Asked twice -- once to decide the tier, once to write -- and the same way both times.
    assert fake.calls and all(c == fake.calls[0] for c in fake.calls)
    call = fake.calls[0]
    assert call["strict"] is True
    assert call["speakers"] == (meta["speaker_ids"]["A"], meta["speaker_ids"]["B"])
    lo, hi = call["time_range"]
    assert round((hi - lo) * SR) == len(mono)
    assert lo == pytest.approx(meta["source_start"], abs=1e-3)


def test_the_mono_recording_is_the_mixture_whatever_the_tracks_are(tmp_path):
    _, _, with_tracks, _, _ = _tracks(tmp_path / "with")
    _run(tmp_path / "without", FakeLLM(_good()), max_candidates=1)
    folder, _ = _only(tmp_path / "without")
    without, _ = sf.read(str(folder / "mixture.wav"), dtype="float32")
    assert np.array_equal(with_tracks, without)


def test_without_separation_output_the_mixture_is_gated_as_before(tmp_path):
    _run(tmp_path, FakeLLM(_good()), max_candidates=1)
    _, meta = _only(tmp_path)
    assert meta["channels_2ch"]["method"] == "time_gated"
    assert "separated_spans" not in meta


def test_a_silent_separated_track_falls_back_to_the_gated_mixture_and_says_so(tmp_path):
    log = Logger()
    _, meta, _, stereo, _ = _tracks(tmp_path, fake=FakeSeparation(left=0.0), logger=log)
    assert meta["channels_2ch"]["method"] == "time_gated"
    assert np.any(np.abs(stereo) > 1e-3), "the fallback is the gated mixture, not silence"
    assert any("silent" in m for m in log.warnings)


def test_a_separation_service_that_fails_does_not_lose_the_excerpt(tmp_path):
    log = Logger()
    result, meta, _, _, _ = _tracks(
        tmp_path, fake=FakeSeparation(fail=RuntimeError("boom")), logger=log)
    assert result.exports and meta["channels_2ch"]["method"] == "time_gated"
    assert any("boom" in m for m in log.warnings)


def test_a_track_shorter_than_the_cut_is_padded_not_broadcast(tmp_path):
    class Short(FakeSeparation):
        def export_sdlm_dual_channel(self, *a, **k):
            left, right = super().export_sdlm_dual_channel(*a, **k)
            return left[:-50], right[:-50]

    _, meta, mono, stereo, _ = _tracks(tmp_path, fake=Short())
    assert stereo.shape == (len(mono), 2) and meta["channels_2ch"]["method"] == "strict_separation_tracks"


def test_where_the_separator_worked_and_where_it_failed_is_written_into_the_metadata(tmp_path):
    seg = SimpleNamespace(
        speaker=A, start=0.0, end=400.0, audio=None,
        bss_spans=[(150.0, 152.0, 0.81)],
        bss_failed_spans=[(160.0, 161.0, "low_similarity", "sim 0.1")])
    _, meta, _, _, _ = _tracks(tmp_path, speech_segments=[seg])
    assert [row["similarity"] for row in meta["separated_spans"]] == [0.81]
    assert meta["failed_separation_spans"][0]["reason"] == "low_similarity"
    assert meta["failed_separation_spans"][0]["zeroed_in_strict_track"] is True


def test_with_the_real_track_builder_an_overlap_is_heard_one_voice_to_an_ear(tmp_path):
    """Two voices at once in the recording, one each in the separated tracks:
    the left ear must hold only A and the right only B where they overlap."""
    from services.separation_service import SeparationService

    segs = _talk(30)
    segs[21].start -= 3.0                # B starts 3s before A (seg 20) has finished
    speech = []
    for seg in segs:
        n = round((seg.end - seg.start) * SR)
        level = 0.1 if seg.speaker == A else 0.2
        speech.append(SimpleNamespace(
            index=seg.index, speaker=seg.speaker, start=seg.start, end=seg.end,
            audio=np.full(n, level, dtype=np.float32), bss_spans=[], bss_failed_spans=[]))
    real = SeparationService.__new__(SeparationService)
    real.logger = None

    result, meta, mono, stereo, _ = _tracks(tmp_path, segs=segs, speech_segments=speech, fake=real)
    assert meta["channels_2ch"]["method"] == "strict_separation_tracks"
    # Inside the overlap [seg 21 start, seg 20 end] both are speaking; each ear is one constant.
    src = meta["source_start"]
    assert src < segs[21].start and segs[20].end < meta["source_end"], "the excerpt must hold the overlap"
    a, b = round((segs[21].start + 0.3 - src) * SR), round((segs[20].end - 0.3 - src) * SR)
    assert b > a
    left, right = stereo[a:b, 0], stereo[a:b, 1]
    assert np.allclose(left, 0.1, atol=1e-3), "the left ear must hold only speaker A"
    assert np.allclose(right, 0.2, atol=1e-3), "the right ear must hold only speaker B"


def test_the_pass_is_given_the_separation_output_by_the_pipeline(tmp_path):
    from services.pipeline_service import PipelineService
    from utils.excise import TimelineMap as _Timeline
    from services.conversation_export_service import ConversationExportRun

    class Capture:
        def run(self, transcripts, **kwargs):
            self.kwargs = kwargs
            return ConversationExportRun(exports=[], report={})

    class Out:
        def write_conversation_exports(self, report, exports):
            pass

    pipe = PipelineService.__new__(PipelineService)
    pipe.conversation_export_svc, pipe.timeline = Capture(), _Timeline()
    pipe.noise_track, pipe.logger, pipe.separation_svc = None, None, "the-separation-service"
    audio = SimpleNamespace(waveform=np.zeros(SR, dtype=np.float32), sample_rate=SR)
    speech = [_speech(A)]
    pipe._export_conversation_exports(Out(), [], audio, None, str(tmp_path), "ep.mp3",
                                      speech_segments=speech)
    assert pipe.conversation_export_svc.kwargs["speech_segments"] is speech
    assert pipe.conversation_export_svc.kwargs["separation_service"] == "the-separation-service"


# --- the folders: tiers, numbering, overlap first ---------------------------------------

from services import conversation_export_service
from services.conversation_export_service import (
    FOLDER_ORDER, OVERLAP_BAD, OVERLAP_GOOD, cross_speaker_overlaps, tier_folder)


def test_tier_folders_are_numbered_so_the_best_one_sorts_first():
    names = [tier_folder(t) for t in FOLDER_ORDER]
    assert names == ["tier_1_overlap_good", "tier_2_S", "tier_3_A", "tier_4_B",
                     "tier_5_C", "tier_6_overlap_bad"]
    assert names == sorted(names)


def test_a_folder_that_is_not_a_known_tier_sorts_after_all_of_them():
    assert tier_folder("odd") == "tier_7_odd" and tier_folder("odd") > tier_folder(OVERLAP_BAD)


def test_overlap_is_where_two_different_speakers_are_heard_at_once():
    segs = [_speech(A, 0.0, 10.0), _speech(B, 8.0, 12.0), _speech(A, 12.5, 14.0)]
    assert cross_speaker_overlaps(segs) == [(8.0, 10.0)]


def test_one_speaker_over_themselves_and_a_touching_turn_are_not_overlap():
    assert cross_speaker_overlaps([_speech(A, 0.0, 5.0), _speech(A, 3.0, 8.0)]) == []
    assert cross_speaker_overlaps([_speech(A, 0.0, 5.0), _speech(B, 5.0, 9.0)]) == []


def test_overlaps_that_touch_or_cross_are_merged():
    segs = [_speech(A, 0.0, 10.0), _speech(B, 2.0, 4.0), _speech(B, 3.5, 6.0),
            _speech(B, 20.0, 21.0), _speech(A, 20.5, 22.0)]
    assert cross_speaker_overlaps(segs) == [(2.0, 6.0), (20.5, 21.0)]


def test_every_conversation_folder_holds_exactly_the_agreed_files(tmp_path):
    _run(tmp_path, FakeLLM(_good()))
    for folder in _folders(tmp_path):
        assert sorted(p.name for p in folder.iterdir()) == [
            "conversation.json", "mixture.wav", "speaker_A.wav", "speaker_B.wav",
            "stereo_2ch.wav", "transcript.txt"]


def test_conversations_are_numbered_from_one_within_each_tier_and_the_json_says_where_it_is(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good()), segs=_talk(60), max_candidates=40)
    assert len(result.exports) >= 2
    for tier in tmp_path.glob("tier_*"):
        numbers = sorted(int(c.name.rsplit("_", 1)[1]) for c in tier.glob("conversation_*"))
        assert numbers == list(range(1, len(numbers) + 1)), tier.name
    for folder in _folders(tmp_path):
        assert _doc_of(folder)["folder"] == f"{folder.parent.name}/{folder.name}"


def test_the_export_rows_point_at_files_that_exist_under_the_export_root(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good()), segs=_talk(60), max_candidates=40)
    for row in result.exports:
        for rel in row["files"].values():
            assert (tmp_path / rel).is_file(), rel
        assert row["verified"] is True


def test_the_report_counts_the_tier_folders(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good()), segs=_talk(60), max_candidates=40)
    counted = {t.name: len(list(t.glob("conversation_*"))) for t in tmp_path.glob("tier_*")}
    assert result.report["tiers"] == counted
    assert result.report["verification_failed"] == 0


def test_a_second_run_does_not_leave_the_first_ones_folders_behind(tmp_path):
    _run(tmp_path, FakeLLM(_good()))
    assert _folders(tmp_path)
    _run(tmp_path, FakeLLM(_good(3)))               # nothing is accepted this time
    assert _folders(tmp_path) == []


def test_a_folder_that_is_not_ours_is_left_alone_when_the_old_ones_are_cleared(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "keep.txt").write_text("mine", encoding="utf-8")
    _run(tmp_path, FakeLLM(_good()))
    assert (tmp_path / "notes" / "keep.txt").read_text(encoding="utf-8") == "mine"


# --- the read-back check ----------------------------------------------------------------

def test_every_file_is_read_back_and_the_result_is_in_the_json(tmp_path):
    _run(tmp_path, FakeLLM(_good()), max_candidates=1)
    folder, doc = _only(tmp_path)
    check = doc["verification"]
    assert check["ok"] is True and all(check["checks"].values())
    assert set(check["checks"]) >= {"same_sample_rate", "same_length", "mixture_is_mono",
                                    "speaker_files_are_mono", "stereo_is_two_channels",
                                    "stereo_left_is_speaker_A", "stereo_right_is_speaker_B"}
    frames = {v["frames"] for v in check["files"].values()}
    assert frames == {check["num_samples"]}
    assert check["files"]["stereo_2ch"]["channels"] == 2
    assert check["files"]["speaker_A"]["channels"] == 1
    assert check["sample_rate"] == SR


def test_the_recorded_hash_is_the_hash_of_the_file_on_disk(tmp_path):
    import hashlib
    _run(tmp_path, FakeLLM(_good()), max_candidates=1)
    folder, doc = _only(tmp_path)
    for key, row in doc["verification"]["files"].items():
        assert row["sha256"] == hashlib.sha256((folder / row["file"]).read_bytes()).hexdigest()


def test_a_file_that_has_drifted_from_the_others_fails_the_check(tmp_path):
    svc, _, _ = _run(tmp_path, FakeLLM(_good()), max_candidates=1)
    folder, doc = _only(tmp_path)
    b, sr = sf.read(str(folder / "speaker_B.wav"), dtype="int16")
    sf.write(str(folder / "speaker_B.wav"), np.roll(b, 5), sr, subtype="PCM_16")   # 5 samples late
    files = {k: v["file"] for k, v in doc["verification"]["files"].items()}
    check = svc._verify(str(folder), files, SR, doc["verification"]["num_samples"])
    assert check["ok"] is False and check["checks"]["stereo_right_is_speaker_B"] is False
    assert check["checks"]["same_length"] is True


def test_a_file_of_the_wrong_length_fails_the_check(tmp_path):
    svc, _, _ = _run(tmp_path, FakeLLM(_good()), max_candidates=1)
    folder, doc = _only(tmp_path)
    a, sr = sf.read(str(folder / "speaker_A.wav"), dtype="int16")
    sf.write(str(folder / "speaker_A.wav"), a[:-3], sr, subtype="PCM_16")
    files = {k: v["file"] for k, v in doc["verification"]["files"].items()}
    check = svc._verify(str(folder), files, SR, doc["verification"]["num_samples"])
    assert check["ok"] is False and check["checks"]["same_length"] is False


def test_a_failed_check_is_counted_and_reported_as_an_error(tmp_path, monkeypatch):
    log = Logger()
    log.errors = []
    log.error = log.errors.append
    monkeypatch.setattr(ConversationExportService, "_verify",
                        lambda self, *a, **k: {"ok": False, "checks": {"same_length": False},
                                               "files": {}})
    _, result, _ = _run(tmp_path, FakeLLM(_good()), max_candidates=1, logger=log)
    assert result.report["verification_failed"] == 1
    assert result.exports[0]["verified"] is False
    assert any("same_length" in m for m in log.errors)


# --- overlap folders --------------------------------------------------------------------

def _speech_for(segs, similarity=0.8, covered=1.0):
    """Separation output for `segs`: A at one level, B at another, nothing failed.

    Each segment that takes part in an overlap carries the span the separator worked
    on, with its similarity; `covered` is the share of the overlap it managed.
    """
    overlaps = cross_speaker_overlaps(segs)
    out = []
    for seg in segs:
        n = round((seg.end - seg.start) * SR)
        spans = []
        for a, b in overlaps:
            lo, hi = max(a, seg.start), min(b, seg.end)
            if hi > lo:
                spans.append((lo, lo + (hi - lo) * covered, similarity))
        out.append(SimpleNamespace(
            index=seg.index, speaker=seg.speaker, start=seg.start, end=seg.end,
            audio=np.full(n, 0.1 if seg.speaker == A else 0.2, dtype=np.float32),
            bss_spans=spans, bss_failed_spans=[]))
    return out


def _real_tracks():
    from services.separation_service import SeparationService
    real = SeparationService.__new__(SeparationService)
    real.logger = None
    return real


class _Item:
    topic, semantic, trimmed, reason, why, trim_ignored, candidate = "t", 5, False, None, "", None, None


def _planned(overlap=True, tier="B", fake=None, speech=None, speech_args=None, **settings):
    """The plan for the first candidate of a talk in which B breaks in on A (or does not)."""
    from utils.conversation_selection import ConversationSelectionFinder
    segs = _talk(30)
    if overlap:
        segs[21].start -= 3.0            # B starts 3s before A (seg 20) has finished
    svc = ConversationExportService(FakeLLM(_good()), **settings)
    finder = ConversationSelectionFinder(segs, TimelineMap(), _quiet(), None, svc.cfg)
    cand = next(c for c in finder.candidates()
                if finder.segs[c.first].index <= "00020" and finder.segs[c.last].index >= "00022")
    cand.tier = tier
    speech = speech if speech is not None else _speech_for(segs, **(speech_args or {}))
    return svc._plan(finder, cand, _Item(), _wave(), SR, speech,
                     fake if fake is not None else _real_tracks())


def test_a_cleanly_separated_overlap_is_filed_apart_whatever_its_score():
    for tier in ("S", "B"):
        plan = _planned(overlap=True, tier=tier)
        assert plan.overlap_seconds > 0.3
        assert plan.method == "strict_separation_tracks"
        assert plan.tier == OVERLAP_GOOD and plan.overlap_quality == "good"
        assert plan.overlap_reasons == []
        assert plan.separated_share == pytest.approx(1.0) and plan.min_similarity == 0.8


def test_without_overlap_the_tier_is_the_one_the_score_gives():
    plan = _planned(overlap=False, tier="B")
    assert plan.overlap_seconds == 0.0 and plan.tier == "B" and plan.overlap_quality is None


def test_the_folders_are_by_score_alone_when_overlap_first_is_off():
    plan = _planned(overlap=True, tier="B", overlap_first=False)
    assert plan.tier == "B"
    assert plan.overlap_quality == "good", "how good the overlap is is still recorded"


def test_an_overlap_shorter_than_the_threshold_is_not_an_overlap_excerpt():
    plan = _planned(overlap=True, tier="B", overlap_min_seconds=10.0)
    assert plan.tier == "B" and plan.overlap_quality is None


def test_an_overlap_that_was_only_gated_from_the_mixture_is_a_bad_one():
    """Both voices are still in both ears there."""
    class NoTracks(FakeSeparation):
        def export_sdlm_dual_channel(self, *a, **k):
            raise RuntimeError("no tracks")
    plan = _planned(overlap=True, tier="S", fake=NoTracks())
    assert plan.method == "time_gated" and plan.overlap_seconds > 0.3
    assert plan.tier == OVERLAP_BAD and plan.overlap_reasons == ["not_separated_tracks"]


def test_an_overlap_with_no_separation_output_at_all_is_a_bad_one():
    plan = _planned(overlap=True, tier="S", speech=[])
    assert plan.tier == OVERLAP_BAD and plan.overlap_reasons == ["not_separated_tracks"]


def test_an_overlap_the_separator_failed_on_is_a_bad_one():
    segs = _talk(30)
    segs[21].start -= 3.0
    speech = _speech_for(segs)
    speech[21].bss_failed_spans = [(165.0, 166.0, "low_similarity", "sim 0.1")]
    plan = _planned(overlap=True, tier="S", speech=speech)
    assert plan.tier == OVERLAP_BAD and "separation_failed" in plan.overlap_reasons


def test_an_overlap_separated_with_a_weak_match_is_a_bad_one():
    plan = _planned(overlap=True, tier="S", speech_args={"similarity": 0.3})
    assert plan.tier == OVERLAP_BAD and plan.overlap_reasons == ["low_similarity"]
    assert plan.min_similarity == 0.3


def test_an_overlap_only_partly_separated_is_a_bad_one():
    plan = _planned(overlap=True, tier="S", speech_args={"covered": 0.5})
    assert plan.tier == OVERLAP_BAD and plan.overlap_reasons == ["overlap_not_separated"]
    assert plan.separated_share == pytest.approx(0.5, abs=0.02)


def test_the_bar_for_a_good_overlap_is_configurable():
    assert _planned(overlap=True, tier="S", speech_args={"similarity": 0.3},
                    overlap_good_min_similarity=0.2).tier == OVERLAP_GOOD
    assert _planned(overlap=True, tier="S", speech_args={"covered": 0.5},
                    overlap_good_min_coverage=0.4).tier == OVERLAP_GOOD


def test_every_reason_an_overlap_is_bad_is_listed_not_just_the_first():
    segs = _talk(30)
    segs[21].start -= 3.0
    speech = _speech_for(segs, similarity=0.3, covered=0.5)
    speech[21].bss_failed_spans = [(165.0, 166.0, "low_similarity", "x")]
    plan = _planned(overlap=True, tier="S", speech=speech)
    assert set(plan.overlap_reasons) == {"separation_failed", "low_similarity",
                                         "overlap_not_separated"}


def test_the_json_says_how_good_the_overlap_is(tmp_path):
    segs = _talk(30)
    segs[21].start -= 3.0
    result, meta, _, _, _ = _tracks(tmp_path, segs=segs, speech_segments=_speech_for(segs),
                                    fake=_real_tracks())
    assert meta["tier"] == OVERLAP_GOOD and meta["folder"].startswith("tier_1_overlap_good/")
    assert meta["overlap_quality"] == "good" and meta["overlap_reasons"] == []
    assert meta["overlap_separated_share"] == pytest.approx(1.0)
    assert meta["overlap_min_similarity"] == 0.8
    assert meta["overlap_seconds"] > 0.3 and meta["overlap_spans"]
    span = meta["overlap_spans"][0]
    assert 0.0 <= span["start"] < span["end"] <= meta["duration"]
    assert meta["tier_by_score"] in ("S", "A", "B", "C")
    assert result.exports[0]["overlap_quality"] == "good"


def test_the_json_says_why_a_bad_overlap_is_bad(tmp_path):
    segs = _talk(30)
    segs[21].start -= 3.0
    result, meta, _, _, _ = _tracks(tmp_path, segs=segs,
                                    speech_segments=_speech_for(segs, similarity=0.3),
                                    fake=_real_tracks())
    assert meta["tier"] == OVERLAP_BAD and meta["folder"].startswith("tier_6_overlap_bad/")
    assert meta["overlap_quality"] == "bad" and meta["overlap_reasons"] == ["low_similarity"]
    assert result.exports[0]["overlap_reasons"] == ["low_similarity"]


def test_conversations_with_overlap_leave_the_score_tiers_and_the_rest_stay(tmp_path):
    segs = _talk(60)
    segs[41].start -= 3.0                          # an overlap far into the recording
    _, result, _ = _run(tmp_path, FakeLLM(_good()), segs=segs, max_candidates=40,
                        max_seconds=100.0,
                        run_kwargs={"speech_segments": _speech_for(segs),
                                    "separation_service": _real_tracks()})
    with_overlap = [r for r in result.exports if r["overlap_seconds"] > 0]
    without = [r for r in result.exports if r["overlap_seconds"] == 0]
    assert with_overlap and without
    assert {r["tier"] for r in with_overlap} == {OVERLAP_GOOD}
    assert not {r["tier"] for r in without} & {OVERLAP_GOOD, OVERLAP_BAD}
    names = sorted(p.name for p in tmp_path.glob("tier_*"))
    assert names[0] == "tier_1_overlap_good"


def test_the_bad_overlaps_sort_after_every_score_tier(tmp_path):
    segs = _talk(60)
    segs[41].start -= 3.0
    _, result, _ = _run(tmp_path, FakeLLM(_good()), segs=segs, max_candidates=40,
                        max_seconds=100.0,
                        run_kwargs={"speech_segments": _speech_for(segs, similarity=0.2),
                                    "separation_service": _real_tracks()})
    names = sorted(p.name for p in tmp_path.glob("tier_*"))
    assert names[-1] == "tier_6_overlap_bad" and len(names) >= 2


def test_the_report_counts_the_overlap_folders_too(tmp_path):
    segs = _talk(60)
    segs[41].start -= 3.0
    _, result, _ = _run(tmp_path, FakeLLM(_good()), segs=segs, max_candidates=40,
                        max_seconds=100.0,
                        run_kwargs={"speech_segments": _speech_for(segs),
                                    "separation_service": _real_tracks()})
    assert result.report["tiers"].get("tier_1_overlap_good", 0) >= 1


def _refiled(monkeypatch, assign):
    """Give the picked excerpts the tier and score `assign(k, candidate)` says."""
    real = conversation_export_service.pick_non_overlapping

    def picked(candidates):
        chosen = real(candidates)
        for k, cand in enumerate(chosen):
            cand.tier, cand.score = assign(k, cand)
        return chosen

    monkeypatch.setattr(conversation_export_service, "pick_non_overlapping", picked)


def test_each_tier_folder_is_numbered_from_one_and_ids_follow_the_folder_order(tmp_path, monkeypatch):
    tiers = ["A", "S", "B", "A", "B", "B", "S"]
    _refiled(monkeypatch, lambda k, c: (tiers[k % len(tiers)], 90.0 - k))
    _, result, _ = _run(tmp_path, FakeLLM(_good()), segs=_talk(60), max_candidates=40,
                        max_seconds=100.0)
    assert len(result.exports) >= 4
    folders = sorted(p.name for p in tmp_path.glob("tier_*"))
    assert len(folders) >= 2 and folders == sorted(folders)
    for tier in tmp_path.glob("tier_*"):
        numbers = sorted(int(c.name.rsplit("_", 1)[1]) for c in tier.glob("conversation_*"))
        assert numbers == list(range(1, len(numbers) + 1)), tier.name
        for conv in tier.glob("conversation_*"):
            assert _doc_of(conv)["tier"] == tier.name.split("_", 2)[2]
    ranks = [tier_folder(row["tier"]) for row in result.exports]
    assert ranks == sorted(ranks), "the best folder is written and numbered first"
    ids = [row["id"] for row in result.exports]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
