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
    CONVERSATION_EXPORT_PROMPT_VERSION, ConversationExportService, cut_excerpt, parse_verdict)
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

    def ensure_loaded(self):
        return self.loaded

    def count_tokens(self, text):
        return len(text.split())

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None):
        self.messages.extend(user_messages)
        return True, [self.reply(m) if callable(self.reply) else self.reply
                      for m in user_messages]


def _good(score=5, topic="chủ đề thử"):
    return json.dumps({"self_contained": score, "topic": topic})


def _run(tmp_path, llm, segs=None, noise="quiet", timeline=None, **settings):
    segs = segs if segs is not None else _talk(30)
    wave = _wave()
    svc = ConversationExportService(llm, **settings)
    result = svc.run(
        segs, timeline=timeline or TimelineMap(),
        noise=_quiet() if noise == "quiet" else NoiseTrack(),
        music_map=None, waveform=wave, sample_rate=SR,
        out_dir=str(tmp_path), base_name="ep 01")
    return svc, result, wave


def _files(tmp_path, sub, ext):
    """Files in `sub` ending in `ext`; the two-channel companions are not the mono."""
    d = tmp_path / sub
    names = sorted(p.name for p in d.iterdir()) if d.exists() else []
    return [n for n in names if n.endswith(ext) and not n.endswith("_2ch.wav")]


# --- reading the model's verdict -------------------------------------------------

def test_a_plain_verdict_is_parsed():
    v = parse_verdict('{"self_contained": 4, "topic": "làm bếp", '
                      '"start_index": "00012", "end_index": "00040"}')
    assert v == {"self_contained": 4, "topic": "làm bếp",
                 "start_index": "00012", "end_index": "00040"}


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
    assert set(v) == {"self_contained", "topic", "start_index", "end_index"}


# --- what the model sees ------------------------------------------------------------

def test_the_model_sees_turns_as_a_and_b_by_index_without_raw_labels_or_times(tmp_path):
    llm = FakeLLM(_good())
    _run(tmp_path, llm, max_candidates=2)
    assert llm.messages
    message = llm.messages[0]
    assert "SPEAKER_" not in message
    assert re.search(r"^#\d{5} [AB]: ", message, re.M)
    assert not re.search(r"\d{2}:\d{2}", message)


def test_only_the_shortlist_is_sent_to_the_model(tmp_path):
    llm = FakeLLM(_good())
    _, result, _ = _run(tmp_path, llm, max_candidates=3)
    assert len(llm.messages) <= 3
    assert result.report["shortlisted"] == len(llm.messages)


# --- accepting and rejecting -----------------------------------------------------------

def test_a_self_contained_clip_is_written_as_audio_and_metadata(tmp_path):
    llm = FakeLLM(_good(5, "chuyện nấu ăn"))
    _, result, wave = _run(tmp_path, llm)
    assert result.exports
    audio = _files(tmp_path, "audio", ".wav")
    meta = _files(tmp_path, "metadata", ".json")
    assert audio and len(audio) == len(meta) == len(result.exports)
    assert audio[0].startswith("ep_01_conversation_000001")

    info = sf.info(str(tmp_path / "audio" / audio[0]))
    assert info.samplerate == SR and info.channels == 1
    doc = json.loads((tmp_path / "metadata" / meta[0]).read_text(encoding="utf-8"))
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
    meta_name = _files(tmp_path, "metadata", ".json")[0]
    doc = json.loads(
        (tmp_path / "metadata" / meta_name).read_text(encoding="utf-8"))
    row = doc["conversation"][0]
    source = by_index[row["index"]]
    assert [word["word"] for word in row["words"]] == source.text.split()
    assert row["words"][0]["start"] == pytest.approx(
        source.words[0]["start"] - doc["source_start"], abs=0.001)


def test_a_low_score_rejects_the_clip(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good(3)))
    assert result.exports == []
    assert result.report["judged_rejected"]["not_self_contained"] >= 1
    assert _files(tmp_path, "audio", ".wav") == []


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
        ids = re.findall(r"^#(\d{5}) ", message, re.M)
        lo = ids[first] if first is not None else None
        hi = ids[last] if last is not None else None
        body = {"self_contained": 5, "topic": "t"}
        if lo:
            body["start_index"] = lo
        if hi:
            body["end_index"] = hi
        return json.dumps(body)
    return reply


def test_a_trim_inside_the_candidate_shortens_it_and_is_marked(tmp_path):
    # Drop the first and the last line of whatever candidate is shown.
    _, result, _ = _run(tmp_path, FakeLLM(_trim(1, -2)), max_candidates=1)
    assert result.exports and result.exports[0]["trimmed_by_model"] is True


def test_a_trim_that_leaves_less_than_the_minimum_is_rejected(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_trim(0, 3)), max_candidates=1)
    assert result.exports == []
    assert result.report["judged_rejected"].get("trim_duration", 0) >= 1


def test_a_trim_outside_the_candidate_rejects_it(tmp_path):
    def outside(message):
        return json.dumps({"self_contained": 5, "start_index": "99999",
                           "end_index": "88888"})
    _, result, _ = _run(tmp_path, FakeLLM(outside), max_candidates=1)
    assert result.exports == []
    assert result.report["judged_rejected"]["bad_trim"] >= 1


def test_a_trim_back_to_front_rejects_it(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_trim(-3, 2)), max_candidates=1)
    assert result.exports == []
    assert result.report["judged_rejected"]["bad_trim"] >= 1


def test_a_trim_cannot_widen_a_clip_past_the_candidate(tmp_path):
    """Indexes just outside the candidate exist in the recording but not in it."""
    def widen(message):
        ids = [int(i) for i in re.findall(r"^#(\d{5}) ", message, re.M)]
        return json.dumps({"self_contained": 5, "start_index": f"{ids[0] - 1:05d}",
                           "end_index": f"{ids[-1] + 1:05d}"})
    _, result, _ = _run(tmp_path, FakeLLM(widen), max_candidates=1)
    assert result.exports == []
    assert result.report["judged_rejected"] == {"bad_trim": 1}


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
    doc = json.loads(next((tmp_path / "metadata").glob("*.json")).read_text(encoding="utf-8"))
    data, sr = sf.read(str(tmp_path / doc["audio"]))
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


def _read(tmp_path, rel):
    data, sr = sf.read(str(tmp_path / rel), dtype="float32", always_2d=True)
    return data, sr


def _first_clip(tmp_path):
    doc = json.loads(next((tmp_path / "metadata").glob("*.json")).read_text(encoding="utf-8"))
    return doc


def test_the_service_writes_a_two_channel_file_beside_each_mono(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good()))
    assert result.exports
    doc = _first_clip(tmp_path)
    mono, sr_m = _read(tmp_path, doc["audio"])
    both, sr_s = _read(tmp_path, doc["audio_2ch"])
    assert mono.shape[1] == 1 and both.shape[1] == 2
    assert sr_m == sr_s == SR
    assert len(mono) == len(both)                          # equal length, sample for sample
    assert len(both) / SR == pytest.approx(doc["duration"], abs=0.01)
    assert result.exports[0]["audio_2ch"] == doc["audio_2ch"]


def test_speaker_a_is_the_left_ear_and_b_the_right_and_the_mono_is_unchanged(tmp_path):
    _run(tmp_path, FakeLLM(_good()))
    doc = _first_clip(tmp_path)
    mono, _ = _read(tmp_path, doc["audio"])
    both, _ = _read(tmp_path, doc["audio_2ch"])
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


def test_the_two_channel_file_starts_and_ends_silent_like_the_mono(tmp_path):
    _run(tmp_path, FakeLLM(_good()))
    both, _ = _read(tmp_path, _first_clip(tmp_path)["audio_2ch"])
    assert np.abs(both[0]).max() < 1e-3 and np.abs(both[-1]).max() < 1e-3


def test_the_speakers_in_the_two_channel_file_match_the_metadata_ids(tmp_path):
    _run(tmp_path, FakeLLM(_good()))
    doc = _first_clip(tmp_path)
    assert doc["speaker_ids"] == {"A": A, "B": B} or doc["speaker_ids"] == {"A": B, "B": A}
    names = {row["speaker"]: row["speaker_id"] for row in doc["conversation"]}
    assert names == doc["speaker_ids"]


def test_each_clip_has_a_plain_text_transcript_naming_both_speakers(tmp_path):
    _run(tmp_path, FakeLLM(_good()))
    doc = _first_clip(tmp_path)
    text = (tmp_path / doc["transcript"]).read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l]
    assert len(lines) == len(doc["conversation"])
    assert any("] A: " in l for l in lines) and any("] B: " in l for l in lines)
    assert lines[0].startswith("[00:")
    assert doc["conversation"][0]["text"] in lines[0]


def test_switching_stereo_off_writes_only_the_mono(tmp_path):
    _, result, _ = _run(tmp_path, FakeLLM(_good()), stereo=False)
    assert result.exports
    assert not [p for p in (tmp_path / "audio").iterdir() if p.name.endswith("_2ch.wav")]
    doc = _first_clip(tmp_path)
    assert "audio_2ch" not in doc and "channels_2ch" not in doc
    assert result.exports[0]["audio_2ch"] is None


def test_one_json_describes_both_audio_files_and_the_one_transcript(tmp_path):
    _run(tmp_path, FakeLLM(_good()))
    docs = list((tmp_path / "metadata").glob("*.json"))
    audio = [p for p in (tmp_path / "audio").iterdir() if p.suffix == ".wav"]
    assert len(audio) == 2 * len(docs)
