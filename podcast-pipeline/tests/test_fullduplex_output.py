"""Full-duplex selection and output: finder fields, gates, categories, roles, files.

Fake LLM and synthetic word-aligned segments only.

Run:  python -m pytest tests/test_fullduplex_output.py -q     (from podcast-pipeline/)
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

from services import conversation_export_service as svc_mod
from services.conversation_export_service import (
    CONVERSATION_EXPORT_PROMPT_VERSION, ConversationExportService, cross_speaker_overlaps,
    parse_finder_reply)
from utils.conversation_selection import (
    ConversationSelectionConfig, ConversationSelectionFinder)
from utils.excise import TimelineMap
from utils.noise_map import KINDS, NoiseTrack

A, B = "SPEAKER_00", "SPEAKER_01"
SR = 8000


# --- builders ------------------------------------------------------------------

def _words(text, start, end):
    tokens = text.split()
    step = (end - start) / len(tokens)
    return [{"word": w, "start": round(start + k * step, 3), "end": round(start + (k + 1) * step, 3)}
            for k, w in enumerate(tokens)]


def _seg(index, speaker, start, end, text=None, flags=None):
    text = text or f"đoạn {index} bàn về chuyện số {index} rất dài dòng"
    seg = SimpleNamespace(index=f"{index:05d}", speaker=speaker, start=start, end=end,
                          text=text, bss=False, unseparated=None,
                          words=_words(text, start, end))
    if flags is not None:
        seg.review_flags = flags
    return seg


def _talk(n=30):
    return [_seg(k, (A, B)[k % 2], k * 8.0, k * 8.0 + 7.0) for k in range(n)]


def _cut_in(segs, k, overlap):
    """Segment k starts `overlap` seconds before the one before it ends."""
    old = segs[k]
    segs[k] = _seg(k, old.speaker, segs[k - 1].end - overlap, old.end, old.text)


def _backchannel(segs, number, speaker, at, text="ừ"):
    segs.append(_seg(100 + number, speaker, at, at + 0.3, text))


def _quiet(total=600):
    return NoiseTrack({k: np.full(int(total * 100), 0.001, dtype=np.float32) for k in KINDS},
                      fps=100.0)


def _wave(seconds=300.0):
    return (np.random.default_rng(0).standard_normal(int(seconds * SR)) * 0.1).astype(np.float32)


def _region(segs, **fields):
    n = len(segs)
    region = {"start_line": 1, "end_line": n, "keep": True, "self_contained": 5,
              "topic": "chủ đề thử", "reason": "ok", "assistant": "B", "role_conf": 0.9,
              "role_why": "B đáp", "interruptions": [], "summary": "Hai người trao đổi."}
    region.update(fields)
    return region


def _line_of(segs, target):
    ordered = sorted(segs, key=lambda s: (s.start, s.end))
    return ordered.index(target) + 1


class FinderLLM:
    model_name = "fake/model"
    batch_size = 100
    max_batch_tokens = 0

    def __init__(self, reply):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)

    def ensure_loaded(self):
        return True

    def count_tokens(self, text):
        return len(text.split())

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None, **extra):
        return True, [self.reply for _ in user_messages]


class FakeSeparation:
    """One constant per speaker, so which track ended up where is visible."""

    def __init__(self, left=0.25, right=-0.25):
        self.left, self.right = left, right

    def export_sdlm_dual_channel(self, speech_segments, audio_duration, sr, strict=True,
                                 speakers=None, time_range=None, log_stats=True):
        n = round(time_range[1] * sr) - round(time_range[0] * sr)
        return (np.full(n, self.left, dtype=np.float32), np.full(n, self.right, dtype=np.float32))


def _real_tracks():
    from services.separation_service import SeparationService
    real = SeparationService.__new__(SeparationService)
    real.logger = None
    return real


def _speech_for(segs, similarity=0.8, covered=1.0):
    """Separation output for `segs`: A at one level, B at another, nothing failed."""
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


def _run(tmp_path, segs, reply=None, separate=True, similarity=0.8, covered=1.0, tracks=None,
         **settings):
    settings.setdefault("min_score", 0)
    reply = reply if reply is not None else [_region(segs)]
    svc = ConversationExportService(FinderLLM(reply), **settings)
    kw = {}
    if separate:
        kw = {"speech_segments": _speech_for(segs, similarity, covered),
              "separation_service": tracks or _real_tracks()}
    result = svc.run(sorted(segs, key=lambda s: (s.start, s.end)), timeline=TimelineMap(),
                     noise=_quiet(), music_map=None, waveform=_wave(), sample_rate=SR,
                     out_dir=str(tmp_path), base_name="ep 01", **kw)
    return svc, result


def _docs(root):
    return {p.parent: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(root.rglob("conversation.json"))}


def _only(root):
    (folder, doc), = _docs(root).items()
    return folder, doc


def _interrupted_talk():
    segs = _talk()
    _cut_in(segs, 11, 1.0)                  # B cuts in over A's last second; A yields
    return segs


# --- finder reply parsing ---------------------------------------------------------

def _reply(**fields):
    base = {"start_line": 2, "end_line": 9, "keep": True, "self_contained": 4,
            "topic": "t", "reason": "r"}
    base.update(fields)
    return json.dumps([base])


def test_the_new_role_fields_are_parsed():
    (r,) = parse_finder_reply(_reply(
        assistant="b", role_conf=0.8, role_why="B đáp",
        interruptions=[{"line": 5, "intent": "Clarify"}],
        summary="Người hỏi mở chuyện. Người đáp trả lời."))
    assert r["assistant"] == "B" and r["role_conf"] == 0.8 and r["role_why"] == "B đáp"
    assert r["interruptions"] == [{"line": 5, "intent": "clarify"}]
    assert r["summary"] == "Người hỏi mở chuyện. Người đáp trả lời."


def test_an_old_reply_without_the_new_fields_still_parses():
    (r,) = parse_finder_reply(_reply())
    assert r["assistant"] is None and r["role_conf"] is None and r["role_why"] == ""
    assert r["interruptions"] == [] and r["summary"] is None
    assert r["start_line"] == 2 and r["keep"] is True


def test_role_conf_percent_is_scaled_and_junk_is_none():
    assert parse_finder_reply(_reply(role_conf=85))[0]["role_conf"] == 0.85    # percent
    assert parse_finder_reply(_reply(role_conf="90"))[0]["role_conf"] == 0.9
    assert parse_finder_reply(_reply(role_conf=1))[0]["role_conf"] == 1.0
    for junk in ("cao", None, True, 250, -1):
        assert parse_finder_reply(_reply(role_conf=junk))[0]["role_conf"] is None


def test_interruptions_outside_the_region_or_with_unknown_intent_are_dropped_and_counted():
    (r,) = parse_finder_reply(_reply(interruptions=[
        {"line": 1, "intent": "clarify"}, {"line": 10, "intent": "clarify"},
        {"line": 4, "intent": "shout"}, {"line": 4, "intent": "disagree"},
        {"line": 4, "intent": "complete"}, "bad", {"line": "x", "intent": "clarify"}]))
    assert r["interruptions"] == [{"line": 4, "intent": "disagree"}]
    assert r["interruptions_dropped"] == {"outside_region": 3, "unknown_intent": 1, "bad_entry": 1}


def test_a_long_summary_is_cut_at_the_last_sentence_end_within_sixty_words():
    sentence = " ".join(["từ"] * 19) + "."
    (r,) = parse_finder_reply(_reply(summary=" ".join([sentence] * 4)))   # 80 words
    assert r["summary"] == " ".join([sentence] * 3)                       # 60 words, ends on a stop


def test_a_long_summary_without_a_sentence_end_is_cut_at_the_word_limit():
    (r,) = parse_finder_reply(_reply(summary=" ".join(["từ"] * 90)))
    assert len(r["summary"].split()) == 60 and r["summary"].endswith("từ…")


def test_a_summary_copied_from_a_prompt_example_or_empty_is_null():
    for copied in svc_mod._FINDER_EXAMPLE_SUMMARIES:
        assert parse_finder_reply(_reply(summary=copied.upper()))[0]["summary"] is None
    assert parse_finder_reply(_reply(summary="   "))[0]["summary"] is None
    for word in ("null", "None", "NULL"):
        assert parse_finder_reply(_reply(summary=word))[0]["summary"] is None


# --- the prompt -------------------------------------------------------------------

def _prompt(**settings):
    return ConversationExportService(FinderLLM("[]"), **settings)._finder_system_prompt()


def test_the_prompt_version_is_bumped():
    assert CONVERSATION_EXPORT_PROMPT_VERSION == "conversation-export-v14-concise"


def test_the_prompt_asks_for_the_new_fields_and_fills_the_new_placeholder():
    prompt = _prompt(context_turns_around_interruption=3)
    for key in ("assistant", "role_conf", "role_why", "interruptions", "summary",
                "topic_shift", "clarify", "disagree", "complete"):
        assert key in prompt
    assert "ít nhất 3 lượt trước và 3 lượt sau" in prompt
    assert "{context_turns}" not in prompt


def test_every_example_answer_in_the_prompt_carries_the_role_fields_for_kept_regions():
    prompt = _prompt()
    answers = [l[3:] for l in prompt.splitlines() if l.startswith("-> ")]
    assert len(answers) == 5
    for answer in answers:
        for region in parse_finder_reply(answer):
            if region["keep"]:
                assert region["assistant"] in ("A", "B") and region["role_conf"]
                assert region["summary"] is None, "the example summaries are refused on parse"
    assert re.search(r'"interruptions": \[\{"line": 3, "intent": "topic_shift"\}\]', prompt)


# --- validator gates ---------------------------------------------------------------

def _finder(segs, **cfg):
    return ConversationSelectionFinder(segs, TimelineMap(), _quiet(), None,
                                       ConversationSelectionConfig(**cfg))


def _soft(segs, *numbers):
    for n in numbers:
        segs[n].review_flags = [{"kind": "speaker_phrase", "phrase": "x", "conf": 0.9, "why": "t"}]


def test_a_single_speaker_phrase_flag_rejects_the_range():
    segs = _talk()
    _soft(segs, 2)
    cand, why = _finder(segs).evaluate(0, 14)
    assert cand is None and why == "review_flag_speaker_phrase"


def test_the_score_is_a_gate_only():
    cand, why = _finder(_talk(), min_score=101).evaluate(0, 14)
    assert cand is None and why == "low_score"
    assert _finder(_talk(), min_score=0).evaluate(0, 14)[0] is not None


def test_one_speaker_or_too_few_turns_or_a_bad_length_is_rejected():
    segs = _talk()
    assert _finder(segs, min_turns_each=9).evaluate(0, 14)[1] == "too_few_turns"
    assert _finder(segs, min_seconds=10, max_seconds=30).evaluate(0, 14)[1] == "duration"
    assert _finder(segs).evaluate(0, 0)[1] in ("duration", "not_two_speakers")


def test_new_config_keys_have_the_designed_defaults():
    cfg = ConversationSelectionConfig()
    assert (cfg.role_min_conf, cfg.keep_symmetric, cfg.keep_quarantine, cfg.assistant_channel) == (
        0.7, True, False, "left")
    assert (cfg.ipu_gap, cfg.overlap_min, cfg.backchannel_max_seconds, cfg.backchannel_max_words,
            cfg.backchannel_context, cfg.yield_within_seconds, cfg.min_backchannels,
            cfg.half_duplex_max_overlap, cfg.max_soft_flags,
            cfg.context_turns_around_interruption) == (0.2, 0.2, 1.0, 2, 1.0, 1.5, 1, 0.2, 2, 2)


def test_an_unknown_key_still_raises_and_a_bad_channel_does_too():
    with pytest.raises(TypeError):
        ConversationSelectionConfig.from_settings({"no_such_key": 1})
    with pytest.raises(ValueError):
        ConversationSelectionConfig(assistant_channel="middle")


def test_every_shipped_profile_still_builds_and_a100_has_the_designed_values():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        profiles = json.load(fh)["environments"]
    built = 0
    for name, profile in profiles.items():
        settings = (profile.get("models") or {}).get("conversation_selection")
        if settings is not None:
            ConversationSelectionConfig.from_settings(settings)
            built += 1
    assert built >= 2
    a100 = ConversationSelectionConfig.from_settings(
        profiles["a100"]["models"]["conversation_selection"])
    assert (a100.max_seconds, a100.min_turns_each, a100.min_turns_total,
            a100.count_backchannels_as_turns, a100.min_score) == (120, 1, 3, True, 70)


# --- selection: letters back to ids, lines to segments -------------------------------------

def test_the_finders_letter_becomes_the_speaker_id_and_its_lines_become_segment_indexes(tmp_path):
    segs = _interrupted_talk()
    reply = [_region(segs, assistant="A", role_conf=0.8,
                     interruptions=[{"line": 12, "intent": "topic_shift"},
                                    {"line": 99, "intent": "clarify"}])]
    svc = ConversationExportService(FinderLLM(reply), min_score=0)
    selection = svc.select(segs, timeline=TimelineMap(), noise=_quiet(), music_map=None,
                           waveform=_wave(), sample_rate=SR, base_name="ep")
    (item,) = selection.kept.values()
    assert item.assistant == A and item.role_conf == 0.8
    assert item.interruptions == {"00011": "topic_shift"}
    assert item.summary == "Hai người trao đổi."
    assert selection.report["interruptions_dropped"] == {"outside_region": 1}


def test_a_letter_that_is_not_in_the_region_means_no_assistant(tmp_path):
    segs = _talk()
    svc = ConversationExportService(FinderLLM([_region(segs, assistant="C")]), min_score=0)
    selection = svc.select(segs, timeline=TimelineMap(), noise=_quiet(), music_map=None,
                           waveform=_wave(), sample_rate=SR, base_name="ep")
    (item,) = selection.kept.values()
    assert item.assistant is None and item.role_conf is None


# --- the output: interruption ---------------------------------------------------------------

def test_an_interruption_is_filed_with_every_file_and_the_assistant_on_the_left(tmp_path):
    segs = _interrupted_talk()
    reply = [_region(segs, interruptions=[{"line": 12, "intent": "clarify"}])]
    _, result = _run(tmp_path, segs, reply, tracks=_real_tracks())
    assert len(result.exports) == 1, result.report
    row = result.exports[0]
    assert row["category"] == "interrupt" and row["quarantined"] is False
    folder = tmp_path / row["folder"]
    assert folder.parent == tmp_path / "interrupt"
    assert re.fullmatch(r"ep_01_\d+s", folder.name)
    assert sorted(p.name for p in folder.iterdir()) == [
        "assistant.wav", "conversation.json", "mixture.wav", "stereo_2ch.json", "stereo_2ch.wav",
        "transcript.txt", "user.wav"]
    doc = json.loads((folder / "conversation.json").read_text(encoding="utf-8"))
    assert doc["category"] == "interrupt"
    assert doc["tags"] == ["interruption"] and doc["event_counts"]["interruption"] == 1
    assert doc["role"] == "locked" and doc["swappable"] is False and doc["role_conf"] == 0.9
    assert doc["speakers"] == {"assistant": B, "user": A}
    assert doc["channels"]["left"] == "assistant" and doc["channels"]["right"] == "user"
    assert doc["summary"] == "Hai người trao đổi." and doc["topic"] == "chủ đề thử"
    assert doc["quality"]["self_contained"] == 5 and "separation" in doc["quality"]["gates_passed"]
    assert doc["quality"]["quarantine_reason"] is None
    assert doc["verification"]["ok"] is True and doc["orig_spans"]
    (event,) = [e for e in doc["events"] if e["type"] == "interruption"]
    assert event["by"] == "assistant" and event["of"] == "user" and event["intent"] == "clarify"
    assert event["response"] in ("yield", "adapt") and event["overlap"] == pytest.approx(1.0, abs=0.01)
    assert event["segment"] == "00011"
    assert {r["role"] for r in doc["conversation"]} == {"assistant", "user"}


def test_without_separation_output_the_roles_are_gated_from_the_mixture_and_it_says_so(tmp_path):
    _run(tmp_path, _talk(), separate=False)
    _, doc = _only(tmp_path)
    assert doc["channels"]["method"] == "time_gated" and doc["category"] == "turn_by_turn"


def test_assistant_and_user_files_hold_their_own_separated_track(tmp_path):
    segs = _talk()
    speech = [SimpleNamespace(speaker=A, start=0.0, end=1.0, audio=None,
                              bss_spans=[], bss_failed_spans=[])]
    svc = ConversationExportService(FinderLLM([_region(segs)]), min_score=0)
    svc.run(segs, timeline=TimelineMap(), noise=_quiet(), music_map=None, waveform=_wave(),
            sample_rate=SR, out_dir=str(tmp_path), base_name="ep 01",
            speech_segments=speech, separation_service=FakeSeparation(0.25, -0.25))
    folder, doc = _only(tmp_path)
    assistant, _ = sf.read(str(folder / "assistant.wav"), dtype="float32")
    user, _ = sf.read(str(folder / "user.wav"), dtype="float32")
    stereo, _ = sf.read(str(folder / "stereo_2ch.wav"), dtype="float32")
    mid = len(stereo) // 2
    # Speaker A (first) got +0.25 and is the user; B (second) got -0.25 and is the assistant.
    assert assistant[mid] == pytest.approx(-0.25, abs=1e-3) and user[mid] == pytest.approx(0.25, abs=1e-3)
    assert stereo[mid, 0] == pytest.approx(-0.25, abs=1e-3)
    assert stereo[mid, 1] == pytest.approx(0.25, abs=1e-3)
    assert doc["channels"]["method"] == "strict_separation_tracks"


def test_the_assistant_channel_setting_swaps_the_ears(tmp_path):
    segs = _talk()
    speech = [SimpleNamespace(speaker=A, start=0.0, end=1.0, audio=None,
                              bss_spans=[], bss_failed_spans=[])]
    svc = ConversationExportService(FinderLLM([_region(segs)]), min_score=0,
                                    assistant_channel="right")
    svc.run(segs, timeline=TimelineMap(), noise=_quiet(), music_map=None, waveform=_wave(),
            sample_rate=SR, out_dir=str(tmp_path), base_name="ep 01",
            speech_segments=speech, separation_service=FakeSeparation(0.25, -0.25))
    folder, doc = _only(tmp_path)
    stereo, _ = sf.read(str(folder / "stereo_2ch.wav"), dtype="float32")
    mid = len(stereo) // 2
    assert stereo[mid, 1] == pytest.approx(-0.25, abs=1e-3)
    assert stereo[mid, 0] == pytest.approx(0.25, abs=1e-3)
    assert doc["channels"]["left"] == "user" and doc["channels"]["right"] == "assistant"
    assert doc["verification"]["ok"] is True


def test_stereo_json_holds_only_the_assistants_words_timed_from_the_excerpt_start(tmp_path):
    segs = _talk()
    _, result = _run(tmp_path, segs, separate=False)
    folder, doc = _only(tmp_path)
    data = json.loads((folder / "stereo_2ch.json").read_text(encoding="utf-8"))
    rows = data["alignments"]
    assistant_segs = [s for s in segs if s.speaker == B and
                      doc["source_start"] <= s.start and s.end <= doc["source_end"] + 0.01]
    expected = [w for s in assistant_segs for w in s.words]
    assert [r[0] for r in rows] == [w["word"] for w in expected]
    assert all(r[2] == "SPEAKER_MAIN" and len(r[1]) == 2 for r in rows)
    first = rows[0]
    assert first[1][0] == pytest.approx(expected[0]["start"] - doc["source_start"], abs=0.002)
    assert all(0.0 <= r[1][0] <= r[1][1] <= doc["duration"] + 0.01 for r in rows)
    assert len(rows) == len(expected) and rows, "no user word is in the assistant's file"


def test_the_manifest_has_one_line_per_excerpt(tmp_path):
    segs = _interrupted_talk()
    reply = [_region(segs)]
    _, result = _run(tmp_path, segs, reply, tracks=_real_tracks())
    lines = [json.loads(l) for l in (tmp_path / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(lines) == len(result.exports) == 1
    row = lines[0]
    assert row["path"] == f"{result.exports[0]['folder']}/stereo_2ch.wav"
    assert (tmp_path / row["path"]).is_file()
    assert row["category"] == "interrupt" and row["tags"] == ["interruption"]
    assert row["role"] == "locked" and row["swappable"] is False
    assert row["topic"] == "chủ đề thử" and row["summary"] == "Hai người trao đổi."
    assert row["event_counts"]["interruption"] == 1 and row["duration"] > 60


def test_the_manifest_is_rewritten_each_run_and_old_folders_go(tmp_path):
    segs = _talk()
    _run(tmp_path, segs, separate=False)
    assert (tmp_path / "turn_by_turn").is_dir()
    (tmp_path / "tier_2_S" / "conversation_1").mkdir(parents=True)
    again = _talk()
    _backchannel(again, 1, B, 8 * 10 + 3.0)          # one backchannel: no category this time
    _run(tmp_path, again, separate=False)
    assert not (tmp_path / "turn_by_turn").exists() and not (tmp_path / "tier_2_S").exists()
    assert (tmp_path / "manifest.jsonl").read_text(encoding="utf-8") == ""


def test_the_transcript_marks_roles_and_interruptions(tmp_path):
    segs = _interrupted_talk()
    reply = [_region(segs, interruptions=[{"line": 12, "intent": "clarify"}])]
    _run(tmp_path, segs, reply, tracks=_real_tracks())
    folder, doc = _only(tmp_path)
    lines = (folder / "transcript.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(doc["conversation"])
    assert re.match(r"\[00:\d\d\.\d\] \[(user|assistant)\] ", lines[0])
    marked = [l for l in lines if ">> (interruption/clarify)" in l]
    assert len(marked) == 1 and "] [assistant] " in marked[0]


# --- categories -------------------------------------------------------------------------

def test_plain_alternating_talk_is_half_duplex_and_needs_no_separation(tmp_path):
    segs = _talk()
    _, result = _run(tmp_path, segs, separate=False)
    (row,) = result.exports
    assert row["category"] == "turn_by_turn" and row["tags"] == []
    assert (tmp_path / "turn_by_turn").is_dir() and row["verified"]
    _, doc = _only(tmp_path)
    assert doc["event_counts"].get("turn_switch", 0) >= 10 and "interruption" not in doc["event_counts"]


def test_two_backchannels_without_an_interruption_are_the_backchannel_category(tmp_path):
    segs = _talk()
    _backchannel(segs, 1, B, 8 * 10 + 3.0)         # inside A's turn 10 (80-87)
    _backchannel(segs, 2, B, 8 * 12 + 3.0)
    _, result = _run(tmp_path, segs, tracks=_real_tracks())
    (row,) = result.exports
    assert row["category"] == "backchannel", result.report
    assert row["tags"] == ["backchannel"] and row["event_counts"]["backchannel"] == 2


def test_an_interruption_with_backchannels_is_filed_as_interruption_with_both_tags(tmp_path):
    segs = _interrupted_talk()
    _backchannel(segs, 1, B, 8 * 4 + 3.0)
    _backchannel(segs, 2, B, 8 * 6 + 3.0)
    reply = [_region(segs)]
    _, result = _run(tmp_path, segs, reply, tracks=_real_tracks())
    (row,) = result.exports
    assert row["category"] == "interrupt"
    assert row["tags"] == ["interruption", "backchannel"]
    assert row["event_counts"]["backchannel"] == 2


def test_a_single_backchannel_is_filed_as_backchannel(tmp_path):
    segs = _talk()
    _backchannel(segs, 1, B, 8 * 10 + 3.0)
    _, result = _run(tmp_path, segs, tracks=_real_tracks())
    (row,) = result.exports
    assert row["category"] == "backchannel" and not result.report["not_exported"]


# --- the separation gate --------------------------------------------------------------------

def _split_around(line):
    """Three kept parts of the 30-turn talk with `line` (1-based) and one more line left out."""
    segs = _interrupted_talk()
    ordered = sorted(segs, key=lambda x: (x.start, x.end))
    return segs, [
        _region(ordered[:line - 1]) | {"start_line": 1, "end_line": line - 1},
        {"start_line": line, "end_line": line, "keep": False, "self_contained": 1,
         "topic": "dòng lỗi", "reason": "lỗi tách giọng"},
        _region(ordered[line:line + 12]) | {"start_line": line + 1, "end_line": line + 12},
        {"start_line": line + 13, "end_line": len(ordered), "keep": False,
         "self_contained": 1, "topic": "phần còn lại", "reason": "bỏ"},
    ]


def test_low_similarity_segments_are_dropped_before_the_model_picks(tmp_path):
    segs = _interrupted_talk()
    _, result = _run(tmp_path, segs, similarity=0.3, tracks=_real_tracks())
    assert result.exports == []                  # every overlap segment is a seam now
    assert "bad_separation" in json.dumps(result.report)


def test_without_separated_tracks_a_full_duplex_excerpt_is_dropped_by_default(tmp_path):
    _, result = _run(tmp_path, _interrupted_talk(), separate=False)
    assert result.exports == [] and result.report["not_exported"] == {"separation_gate": 1}


def test_without_separated_tracks_it_can_still_be_kept_in_quarantine(tmp_path):
    _, result = _run(tmp_path, _interrupted_talk(), separate=False, keep_quarantine=True)
    (row,) = result.exports
    assert row["quarantined"] and row["overlap_reasons"] == ["not_separated_tracks"]


def test_a_partly_separated_overlap_is_dropped_before_the_model_picks(tmp_path):
    _, result = _run(tmp_path, _interrupted_talk(), covered=0.5, tracks=_real_tracks())
    assert result.exports == []


def test_the_gate_thresholds_are_configurable(tmp_path):
    _, result = _run(tmp_path, _interrupted_talk(), similarity=0.3, tracks=_real_tracks(),
                     overlap_good_min_similarity=0.2)
    assert result.exports[0]["category"] == "interrupt"


def test_half_duplex_is_never_quarantined_even_when_gated_from_the_mixture(tmp_path):
    _, result = _run(tmp_path, _talk(), separate=False)
    assert result.exports[0]["category"] == "turn_by_turn" and not result.exports[0]["quarantined"]


# --- roles ---------------------------------------------------------------------------------

def test_a_missing_assistant_is_symmetric_and_kept_by_default(tmp_path):
    segs = _talk()
    reply = [_region(segs, assistant=None, role_conf=None)]
    reply[0].pop("assistant"), reply[0].pop("role_conf")
    _, result = _run(tmp_path, segs, reply, separate=False)
    (row,) = result.exports
    assert row["role"] == "symmetric" and row["swappable"] is True
    _, doc = _only(tmp_path)
    assert doc["role_source"] == "behaviour" and doc["role_conf"] is None


def test_low_role_conf_is_symmetric_and_keep_symmetric_false_drops_it(tmp_path):
    segs = _talk()
    reply = [_region(segs, role_conf=0.5)]
    _, kept = _run(tmp_path / "kept", segs, reply, separate=False)
    assert kept.exports[0]["role"] == "symmetric" and kept.exports[0]["swappable"] is True
    _, dropped = _run(tmp_path / "dropped", segs, reply, separate=False, keep_symmetric=False)
    assert dropped.exports == [] and dropped.report["not_exported"] == {"symmetric_roles": 1}


def test_an_assistant_that_holds_on_while_the_user_yields_loses_confidence(tmp_path):
    segs = _talk()
    # The user (A) cuts into the assistant (B, 88-95 s) with a short remark; B talks on.
    segs[12] = _seg(200, A, 93.0, 93.9, "khoan đã nào")
    segs.append(_seg(201, A, 96.0, 103.0))
    _cut_in(segs, 15, 1.0)                           # the assistant cuts the user and the user yields
    _, result = _run(tmp_path, segs, [_region(segs, role_conf=0.9)], tracks=_real_tracks())
    (row,) = result.exports
    assert row["category"] == "interrupt"
    _, doc = _only(tmp_path)
    assert doc["role_violations"] == ["assistant_holds_on_user_yields"]
    assert doc["role_conf"] == pytest.approx(0.7) and doc["role_conf_finder"] == 0.9
    assert doc["role"] == "symmetric" and doc["swappable"] is True
    responses = {(e["by"], e["response"]) for e in doc["events"] if e["type"] == "interruption"}
    assert responses == {("user", "continue"), ("assistant", "yield")}


def test_the_role_check_runs_on_the_events_of_the_excerpt(tmp_path):
    segs = _interrupted_talk()
    _, result = _run(tmp_path, segs, [_region(segs, assistant="B", role_conf=0.9)],
                     tracks=_real_tracks())
    _, doc = _only(tmp_path)
    assert doc["role"] == "locked" and doc["role_violations"] == []
    assert doc["role_conf_finder"] == 0.9


# --- result rows and the report -------------------------------------------------------------

def test_result_rows_keep_the_fields_other_stages_read(tmp_path):
    _, result = _run(tmp_path, _talk(), separate=False)
    (row,) = result.exports
    for key in ("id", "folder", "audio", "audio_2ch", "files", "duration", "topic", "verified",
                "semantic_score", "source_start", "source_end", "speaker_ids", "noise_kind"):
        assert key in row, key
    assert row["audio_2ch"] == f"{row['folder']}/stereo_2ch.wav"
    assert row["files"]["alignments"] == f"{row['folder']}/stereo_2ch.json"
    assert row["id"].startswith("ep_01_conversation_000001")
    assert (tmp_path / row["audio"]).is_file()


def test_the_report_counts_categories_and_the_prompt_version(tmp_path):
    _, result = _run(tmp_path, _talk(), separate=False)
    assert result.report["categories"] == {"turn_by_turn": 1}
    assert result.report["prompt_version"] == CONVERSATION_EXPORT_PROMPT_VERSION


def test_stage_output_names_categories_not_tiers(tmp_path):
    from services.stage_output_service import StageOutputService
    seen = {}

    class Out(StageOutputService):
        def __init__(self):
            pass

        def _finish(self, stage, name, rows, stats, extra=None):
            seen.update(stats)

    Out().write_conversation_exports(
        {"categories": {"turn_by_turn": 2}, "not_exported": {"no_category": 1}}, [])
    assert seen["categories"] == {"turn_by_turn": 2} and seen["not_exported"] == {"no_category": 1}
    assert "tiers" not in seen


def test_dropped_interruptions_are_counted_for_every_kept_region_even_if_it_fails_validation(tmp_path):
    segs = _talk()
    bad = _region(segs, self_contained=2, interruptions=[{"line": 99, "intent": "clarify"}])
    svc = ConversationExportService(FinderLLM([bad]), min_score=0)
    selection = svc.select(segs, timeline=TimelineMap(), noise=_quiet(), music_map=None,
                           waveform=_wave(), sample_rate=SR, base_name="ep")
    assert selection.report["interruptions_dropped"] == {"outside_region": 1}


def test_kept_example_regions_obey_the_prompts_own_rules_for_the_a100_values():
    import re
    from utils.conversation_selection import BACKCHANNELS, _fold
    prompt = _prompt(min_turns_each=3, min_seconds=30, max_seconds=120)
    blocks = re.split(r"\nVí dụ \d+ - [^\n]*\n", "\n" + prompt)[1:]
    pat = re.compile(r"\[(\d+)\] (\d+):(\d\d)-(\d+):(\d\d) \(d=([\d.]+)s\) ([A-Z]) \{[^}]*\}: (.*)")
    checked = 0
    for block in blocks:
        rows = {int(m[1]): (int(m[2]) * 60 + int(m[3]), int(m[4]) * 60 + int(m[5]),
                            float(m[6]), m[7], m[8])
                for m in map(pat.match, block.splitlines()) if m}
        answer = next(l for l in block.splitlines() if l.startswith("-> "))
        for region in parse_finder_reply(answer[3:]):
            if not region["keep"]:
                continue
            lines = [rows[n] for n in range(region["start_line"], region["end_line"] + 1)]
            span = lines[-1][1] - lines[0][0]
            assert 30 <= span <= 120, span
            turns, last = {}, None
            for _s, _e, d, who, text in lines:
                if d <= 1.0 and _fold(text) in BACKCHANNELS:
                    continue
                assert not (d <= 1.0 and len(text.split()) <= 2 and who == last), text
                if who != last:
                    turns[who] = turns.get(who, 0) + 1
                last = who
            assert len(turns) == 2 and min(turns.values()) >= 3, turns
            checked += 1
    assert checked == 4


# --- review fixes: manifests, skipped reruns, the gate record -----------------------------

def _pause_backchannels():
    """A's turns 10 and 12 have a half-second pause with B's "ừ"/"dạ" inside it, no overlap."""
    segs = _talk()
    for k, word in ((10, "ừ"), (12, "dạ")):
        base = k * 8.0
        segs[k] = _seg(k, A, base, base + 3.5)
        segs.append(_seg(300 + k, A, base + 4.0, base + 7.0))
        segs.append(_seg(400 + k, B, base + 3.6, base + 3.9, word))
    return segs


def test_quarantined_excerpts_go_to_their_own_manifest_not_the_training_one(tmp_path):
    _, result = _run(tmp_path, _interrupted_talk(), separate=False, keep_quarantine=True)
    assert result.exports[0]["quarantined"]
    assert (tmp_path / "manifest.jsonl").read_text(encoding="utf-8") == ""
    (line,) = (tmp_path / "manifest_quarantine.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(line)
    assert row["category"] == "_quarantine" and row["path"].startswith("_quarantine/")


def test_trainable_and_quarantined_rows_are_split_between_the_two_manifests(tmp_path):
    _, result = _run(tmp_path, _talk(), separate=False)
    assert len((tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert (tmp_path / "manifest_quarantine.jsonl").read_text(encoding="utf-8") == ""


def test_a_rerun_that_finds_nothing_clears_the_old_folders_and_manifests(tmp_path):
    segs = _talk()
    _run(tmp_path, segs, separate=False)
    assert (tmp_path / "turn_by_turn").is_dir()
    (tmp_path / "tier_2_S" / "conversation_1").mkdir(parents=True)
    _, again = _run(tmp_path, segs, reply=[_region(segs, self_contained=2)], separate=False)
    assert again.report["skipped"] == "no_valid_model_proposals"
    assert not (tmp_path / "turn_by_turn").exists() and not (tmp_path / "tier_2_S").exists()
    assert (tmp_path / "manifest.jsonl").read_text(encoding="utf-8") == ""
    assert (tmp_path / "manifest_quarantine.jsonl").read_text(encoding="utf-8") == ""


def test_a_rerun_without_audio_clears_too(tmp_path):
    segs = _talk()
    _run(tmp_path, segs, separate=False)
    svc = ConversationExportService(FinderLLM([_region(segs)]), min_score=0)
    result = svc.run(segs, timeline=TimelineMap(), noise=_quiet(), music_map=None, waveform=None,
                     sample_rate=SR, out_dir=str(tmp_path), base_name="ep 01")
    assert result.report["skipped"] == "no_audio"
    assert not (tmp_path / "turn_by_turn").exists()


def test_an_unavailable_llm_keeps_the_old_output_and_says_so(tmp_path):
    segs = _talk()
    _run(tmp_path, segs, separate=False)

    class Down(FinderLLM):
        def ensure_loaded(self):
            return False

    messages = []
    svc = ConversationExportService(
        Down([]), logger=SimpleNamespace(info=messages.append, warning=messages.append),
        min_score=0)
    result = svc.run(segs, timeline=TimelineMap(), noise=_quiet(), music_map=None,
                     waveform=_wave(), sample_rate=SR, out_dir=str(tmp_path), base_name="ep 01")
    assert result.report["skipped"] == "llm_unavailable"
    assert (tmp_path / "turn_by_turn").is_dir()
    assert (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").strip()
    assert any("left as it is" in m for m in messages)


def test_in_pause_backchannels_are_a_category_but_record_that_no_overlap_gate_applied(tmp_path):
    segs = _pause_backchannels()
    _, result = _run(tmp_path, segs, separate=False)
    (row,) = result.exports
    assert row["category"] == "backchannel" and not row["quarantined"]
    _, doc = _only(tmp_path)
    assert doc["overlap_seconds"] == 0 and doc["quality"]["separation_gate"] == "no_overlap"


def test_the_gate_is_recorded_as_applied_when_voices_overlap(tmp_path):
    _, result = _run(tmp_path, _interrupted_talk(), tracks=_real_tracks())
    _, doc = _only(tmp_path)
    assert doc["quality"]["separation_gate"] == "applied"
    _, half = _run(tmp_path / "h", _talk(), separate=False)
    assert _only(tmp_path / "h")[1]["quality"]["separation_gate"] == "not_applicable"


# --- the gate looks at the overlap, not the whole excerpt ----------------------------------

def _with_failed_span(tmp_path, at):
    segs, reply = _split_around(12)
    speech = _speech_for(segs)
    speech[11].bss_failed_spans = [(at, at + 0.4, "separation_failed", "x")]
    svc = ConversationExportService(FinderLLM(reply), min_score=0)
    return svc.run(sorted(segs, key=lambda s: (s.start, s.end)), timeline=TimelineMap(),
                   noise=_quiet(), music_map=None, waveform=_wave(), sample_rate=SR,
                   out_dir=str(tmp_path), base_name="ep 01", speech_segments=speech,
                   separation_service=_real_tracks())


def test_the_segment_with_a_separator_failure_is_left_out_and_the_clean_parts_are_kept(tmp_path):
    result = _with_failed_span(tmp_path, 90.0)
    assert len(result.exports) == 2              # before and after it, never across it
    assert all(row["category"] == "turn_by_turn" for row in result.exports)


def test_a_proposal_that_spans_the_failed_segment_is_refused(tmp_path):
    segs = _interrupted_talk()
    speech = _speech_for(segs)
    speech[11].bss_failed_spans = [(90.0, 90.4, "separation_failed", "x")]
    svc = ConversationExportService(FinderLLM([_region(segs)]), min_score=0)
    result = svc.run(sorted(segs, key=lambda s: (s.start, s.end)), timeline=TimelineMap(),
                     noise=_quiet(), music_map=None, waveform=_wave(), sample_rate=SR,
                     out_dir=str(tmp_path), base_name="ep 01", speech_segments=speech,
                     separation_service=_real_tracks())
    assert result.exports == []


# --- a window the retry answered is not "unanswered" ----------------------------------------

class FlakyOnceLLM(FinderLLM):
    def __init__(self, reply):
        super().__init__(reply)
        self.calls = 0

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None, **extra):
        self.calls += 1
        if self.calls == 1:
            return False, []
        return True, [self.reply for _ in user_messages]


def test_a_window_answered_on_the_retry_does_not_fail_the_file(tmp_path):
    segs = _talk()
    llm = FlakyOnceLLM([_region(segs)])
    svc = ConversationExportService(llm, min_score=0)
    result = svc.run(sorted(segs, key=lambda s: (s.start, s.end)), timeline=TimelineMap(),
                     noise=_quiet(), music_map=None, waveform=_wave(), sample_rate=SR,
                     out_dir=str(tmp_path), base_name="ep 01")
    assert llm.calls >= 2
    assert result.report["unanswered_first_pass"] == 1 and result.report["unanswered"] == 0
    assert result.exports


# --- the model files the conversation; the measurements only veto an invented label ----------

def test_the_models_category_is_kept_even_when_the_code_would_have_said_otherwise(tmp_path):
    segs = _interrupted_talk()                   # the measurements see an interruption
    reply = [_region(segs, category="backchannel")]
    _, result = _run(tmp_path, segs, reply, tracks=_real_tracks())
    (row,) = result.exports
    assert row["category"] == "backchannel"


def test_a_label_the_numbers_contradict_is_filed_by_the_numbers(tmp_path):
    segs = _talk()                               # strict turn taking, no overlap at all
    reply = [_region(segs, category="interrupt")]
    _, result = _run(tmp_path, segs, reply, separate=False)
    (row,) = result.exports
    assert row["category"] == "turn_by_turn"


def test_interrupt_on_a_region_with_only_a_listener_word_is_a_backchannel(tmp_path):
    segs = _talk()
    _backchannel(segs, 1, B, 8 * 10 + 3.0)
    reply = [_region(segs, category="interrupt")]
    _, result = _run(tmp_path, segs, reply, tracks=_real_tracks())
    (row,) = result.exports
    assert row["category"] == "backchannel"


def test_an_unknown_or_missing_category_falls_back_to_the_measurements(tmp_path):
    segs = _talk()
    reply = [_region(segs, category="half_duplex")]
    _, result = _run(tmp_path, segs, reply, separate=False)
    assert result.exports[0]["category"] == "turn_by_turn"


def test_the_prompt_defines_the_three_folders_and_the_failed_separation_flag():
    text = _prompt()
    for needle in ("interrupt:", "backchannel:", "turn_by_turn:", "bad_separation", '"category"'):
        assert needle in text


# --- bad_separation_positions: markers and slivers are not failures --------------------------

def _two_turns():
    return [_seg(0, A, 0.0, 7.0), _seg(1, B, 8.0, 15.0), _seg(2, A, 16.0, 23.0)]


def _speech(spans_by_segment, failed_by_segment=None):
    out = []
    for k, seg in enumerate(_two_turns()):
        out.append(SimpleNamespace(
            index=seg.index, speaker=seg.speaker, start=seg.start, end=seg.end,
            bss_spans=spans_by_segment.get(k, []),
            bss_failed_spans=(failed_by_segment or {}).get(k, [])))
    return out


def _bad(speech):
    from utils.conversation_selection import bad_separation_positions
    return bad_separation_positions(_two_turns(), speech, 0.6, 0.9, 0.2)


def test_a_same_speaker_passthrough_marker_is_not_a_failure():
    assert _bad(_speech({0: [(5.0, 9.0, -2.0)]})) == set()


def test_a_span_scored_only_by_exclusion_is_not_a_low_similarity():
    assert _bad(_speech({0: [(2.0, 4.0, -1.0)]})) == set()


def test_a_real_low_similarity_or_a_hard_failure_marks_the_segment():
    assert _bad(_speech({0: [(2.0, 4.0, 0.3)]})) == {0}
    assert _bad(_speech({}, {1: [(9.0, 10.0, "qc_sim", "x")]})) == {1}
    assert _bad(_speech({}, {1: [(9.0, 10.0, "same_speaker", "x")]})) == set()


def test_a_sliver_of_uncovered_overlap_is_not_a_miss():
    segs = [_seg(0, A, 0.0, 10.0), _seg(1, B, 9.7, 15.0)]       # 0.3 s of overlap
    from utils.conversation_selection import bad_separation_positions
    speech = [SimpleNamespace(index="0", speaker=A, start=0.0, end=10.0, bss_spans=[], bss_failed_spans=[])]
    assert bad_separation_positions(segs, speech, 0.6, 0.9, 0.2) == set()
    segs[1] = _seg(1, B, 8.0, 15.0)                              # 2 s uncovered: a real miss
    assert bad_separation_positions(segs, speech, 0.6, 0.9, 0.2) == {0, 1}
