"""Persona profile stage: schema, measures, events, LLM checks.

Run:  python -m pytest tests/test_persona_profile.py -q     (from podcast-pipeline/)
"""
import json
import os
import re
import sys
import unicodedata

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import services.persona_profile_service as svc_mod
from schemas.persona_profile import (
    AddressTerm, BackchannelProfile, Dist, Filler, InterruptionProfile, LexiconItem,
    Particle, PersonaProfile, PersonaStats, Rate, Ratio, SpeakerProfile, Style,
    TurnProfile)
from schemas.transcript import TranscriptSegment

A, B, C = "SPEAKER_00", "SPEAKER_01", "SPEAKER_02"


def seg(i, speaker, start, end, text, **kw):
    """A real TranscriptSegment with no word times."""
    return TranscriptSegment(
        index=f"{i:05d}", start=start, end=end, speaker=speaker, text=text,
        text_whisper="", text_phowhisper="", text_qwen3="", language="vi",
        bs_roformer=False, bss=False, **kw)


def build(segs, gaps=None):
    """Fill gap_before the way provenance does (first None, then start - prev end).

    `gaps` maps a list position to an explicit value (None = a join lies there).
    """
    gaps = gaps or {}
    prev = None
    for pos, s in enumerate(segs):
        s.gap_before = gaps[pos] if pos in gaps else (
            None if prev is None else round(s.start - prev, 3))
        prev = s.end
    return segs


def words(n, word="từ"):
    return " ".join(f"{word}{k}" for k in range(n))


def _style(**over):
    base = dict(
        speech_rate_wps=Dist(None, None, None, 9), words_per_sentence=Dist(None, None, None, 4),
        fillers=None, particles=None, address_terms=None, llm_candidate_forms=None,
        description=None)
    base.update(over)
    return Style(**base)


def _speaker(style, rate):
    return SpeakerProfile(
        speech_s=41.7, style=style,
        backchannel=BackchannelProfile(per_min_of_other=rate, lexicon=None,
                                       overlap_ratio=Ratio(None, 1)),
        interruption=InterruptionProfile(per_hour_of_other=Rate(0.0, 0, 22.1),
                                         yield_rate=Ratio(None, 0)),
        turn=TurnProfile(duration_s=Dist(None, None, None, 3),
                         response_latency_s=Dist(-0.4, 0.3, 1.2, 88)))


def test_round_trip_keeps_null_zero_and_empty_distinct():
    style = _style(fillers=[Filler("ừm", 1.2), Filler("ờm", 0.0)],
                   particles=[Particle("nhé", 6.23)], address_terms=[AddressTerm("anh", 0)],
                   llm_candidate_forms=[], description=None)
    profile = PersonaProfile(
        schema_version=1, params_version="pp-2026.10.2", source_type="real",
        recording_id="ep", usable_ratio=None,
        speakers={A: _speaker(style, Rate(0.0, 0, 22.1)),
                  B: _speaker(_style(), Rate(None, 1, 1.1))})
    back = json.loads(json.dumps(profile.to_dict(), ensure_ascii=False, allow_nan=False))
    a, b = back["speakers"][A], back["speakers"][B]
    assert a["backchannel"]["per_min_of_other"] == {"value": 0.0, "count": 0, "exposure_min": 22.1}
    assert a["style"]["llm_candidate_forms"] == [] and a["style"]["description"] is None
    assert a["style"]["fillers"][1] == {"form": "ờm", "per_min": 0.0}
    assert a["turn"]["response_latency_s"]["p10"] == -0.4
    assert b["backchannel"]["per_min_of_other"]["value"] is None
    assert b["style"]["llm_candidate_forms"] is None and b["style"]["fillers"] is None
    assert b["turn"]["duration_s"] == {"p10": None, "p50": None, "p90": None, "n": 3}
    assert back["usable_ratio"] is None and back["source_type"] == "real"


def test_nan_is_rejected_by_strict_json():
    profile = PersonaProfile(1, "v", "real", "ep", None,
                             {A: _speaker(_style(), Rate(float("nan"), 1, 1.0))})
    with pytest.raises(ValueError):
        json.dumps(profile.to_dict(), allow_nan=False)


def test_stats_to_dict_keeps_all_keys():
    stats = PersonaStats(excluded_s={"review_flag": 0.0}, unknown_gaps_skipped=0,
                         llm_windows=0, llm_ok=0, warnings=[])
    assert stats.to_dict() == {"excluded_s": {"review_flag": 0.0}, "unknown_gaps_skipped": 0,
                               "llm_windows": 0, "llm_ok": 0, "warnings": []}


# --- constants and thresholds ----------------------------------------------------------

def test_params_version_guard():
    """Changing any constant, dictionary or EventParams default means bump PARAMS_VERSION.

    Update this snapshot AND PARAMS_VERSION together; a snapshot edit without a
    version bump is exactly the mistake this test exists to make visible.
    """
    assert svc_mod.PARAMS_VERSION == "pp-2026.10.2"
    assert svc_mod.params_snapshot() == {
        "MIN_DIST_N": 20, "MIN_RATIO_N": 8, "MIN_EXPOSURE_MIN": 5.0,
        "MIN_EXPOSURE_INTERRUPTION_MIN": 20.0, "MIN_LEXICON_N": 5,
        "RATE_MIN_WORDS": 3, "RATE_MIN_UNIT_S": 1.0, "RATE_MIN_WPS": 0.5, "RATE_MAX_WPS": 9.0,
        "SENTENCE_MIN_WORDS": 2, "SENTENCE_MAX_WORDS": 80,
        "LATENCY_FLOOR_S": -3.0, "LATENCY_CAP_S": 6.0, "EDGE_MARGIN_S": 1.5,
        "NOISE_FLOOR": 0.10, "LLM_WINDOW_CHARS": 6000, "MAX_CANDIDATE_FORMS": 20,
        "DESCRIPTION_MAX_CHARS": 400,
        "FILLERS": ["ừm", "ờm", "ờ", "ưm", "hừm", "đại loại"],
        "PARTICLES": ["nhé", "nhỉ", "nha", "nhá", "nè", "nhể", "hả", "hử", "ạ", "chứ"],
        "ADDRESS_TERMS": ["anh", "chị", "em", "mình", "bạn", "tớ", "cậu", "tôi"],
        "EVENT_PARAMS": {
            "ipu_gap": 0.2, "overlap_min": 0.2, "backchannel_max_seconds": 1.0,
            "backchannel_max_words": 2, "backchannel_context": 1.0,
            "yield_within_seconds": 1.5, "min_backchannels": 2,
            "half_duplex_max_overlap": 0.2, "role_min_conf": 0.7},
    }
    assert svc_mod.PROFILE_PROMPT_VERSION == "persona-v1"


def test_dictionaries_are_nfc():
    for form in svc_mod.FILLERS + svc_mod.PARTICLES + svc_mod.ADDRESS_TERMS:
        assert unicodedata.normalize("NFC", form) == form


def test_percentile_is_linear_and_distribution_needs_enough_samples(monkeypatch):
    assert svc_mod.percentile([5, 1, 3, 2, 4], 0.1) == pytest.approx(1.4)
    assert svc_mod.percentile([5, 1, 3, 2, 4], 0.5) == 3
    assert svc_mod.percentile([5, 1, 3, 2, 4], 0.9) == pytest.approx(4.6)
    monkeypatch.setattr(svc_mod, "MIN_DIST_N", 5)
    d = svc_mod.make_dist([1, 2, 3, 4, 5])
    assert (d.p10, d.p50, d.p90, d.n) == (1.4, 3, 4.6, 5)
    few = svc_mod.make_dist([1, 2, 3, 4])
    assert (few.p10, few.p50, few.p90, few.n) == (None, None, None, 4)
    empty = svc_mod.make_dist([])
    assert (empty.p50, empty.n) == (None, 0)


def test_low_exposure_gives_null_value_but_enough_exposure_with_no_events_gives_zero():
    low = svc_mod.make_rate(2, 1.1, 5.0)
    assert (low.value, low.count, low.exposure_min) == (None, 2, 1.1)
    zero = svc_mod.make_rate(0, 31.0, 5.0)
    assert zero.value == 0.0 and zero.value is not None
    per_hour = svc_mod.make_rate(2, 20.0, 20.0, per=60.0)
    assert per_hour.value == pytest.approx(6.0)
    assert svc_mod.make_rate(1, 0.0, 0.0, per=60.0).value is None      # never divide by zero


def test_ratio_thresholds(monkeypatch):
    assert svc_mod.make_ratio(3, 7).value is None and svc_mod.make_ratio(3, 7).n == 7
    assert svc_mod.make_ratio(0, 8).value == 0.0
    assert svc_mod.make_ratio(6, 8).value == 0.75
    monkeypatch.setattr(svc_mod, "MIN_RATIO_N", 1)
    assert svc_mod.make_ratio(1, 2).value == 0.5


# --- P0 and exclusions -----------------------------------------------------------------

def test_prepare_normalizes_nfc_sorts_and_never_mutates_the_input():
    nfd = unicodedata.normalize("NFD", "ừm đúng rồi")
    late, early = seg(1, A, 5.0, 6.0, nfd), seg(0, B, 0.0, 1.0, "xin chào")
    late.words = [{"word": "ừm", "start": 5.0, "end": 5.5}]
    original = [late, early]
    prepared = svc_mod.prepare_segments(original, timeline=None)
    assert [s.index for s in prepared] == ["00000", "00001"]
    assert all(s.words is None for s in prepared)
    assert prepared[1].text == unicodedata.normalize("NFC", nfd)
    assert prepared[0].gap_before is None and prepared[1].gap_before == 4.0
    assert original == [late, early] and late.words is not None
    assert late.text == nfd and late.gap_before is None      # real segments untouched


def test_exclusions_one_reason_per_segment_in_priority_order(monkeypatch):
    flag = lambda kind: [{"kind": kind, "phrase": None, "conf": 0.9, "why": "x"}]
    rows = [
        seg(0, A, 0, 1, "a b c", review_flags=flag("speaker_turn")),
        seg(1, A, 1, 3, "a b c", review_flags=flag("speaker_phrase")),
        seg(2, A, 3, 6, "a b c", review_flags=flag("not_vietnamese")),
        seg(3, A, 6, 10, "a b c", has_music=True, review_flags=flag("speaker_turn")),
        seg(4, A, 10, 15, "a b c", noise_breakdown={"noise_speech": 0.2, "noise_env": 0.0,
                                                     "noise_room": 0.0}),
        seg(5, A, 15, 21, "a b c", unseparated=[{"start": 15, "end": 16, "reason": "x"}]),
        seg(6, A, 21, 28, "   "),
        seg(7, A, 28, 36, "a b c", noise_breakdown={"noise_speech": 0.05, "noise_env": 0.0,
                                                     "noise_room": 0.0}),   # clean
        seg(8, A, 36, 45, "a b c"),                                          # noise unmeasured
    ]
    reasons = [svc_mod.exclusion_reason(s) for s in rows]
    assert reasons == ["review_flag", "review_flag", "review_flag", "review_flag", "noise",
                       "unseparated_overlap", "empty_text", None, None]
    assert svc_mod.is_empty(rows[6]) and not svc_mod.is_empty(rows[0])
    assert svc_mod.excluded_seconds(rows) == {
        "review_flag": 10.0, "music": 0.0, "noise": 5.0,
        "unseparated_overlap": 6.0, "empty_text": 7.0}


def test_noise_cut_ignores_the_environment_variable_constant(monkeypatch):
    from utils import noise_map
    noisy = seg(0, A, 0, 1, "a b c", noise_breakdown={"noise_speech": 0.05, "noise_env": 0.0,
                                                       "noise_room": 0.0})
    before = svc_mod.exclusion_reason(noisy)
    monkeypatch.setattr(noise_map, "NOTICEABLE", 0.01)
    assert svc_mod.exclusion_reason(noisy) == before == None


def test_source_digest_changes_with_every_field_it_reads_and_is_stable():
    base = lambda **kw: build([seg(0, A, 0, 1, "xin chào", **kw), seg(1, B, 1.2, 2, "chào")])
    digest = lambda segs: svc_mod.source_digest_of(segs)
    assert digest(base()) == digest(base()) and len(digest(base())) == 40
    assert digest(base()) != digest(base(has_music=True))
    assert digest(base()) != digest(base(review_flags=[]))      # None vs [] differ
    assert digest(base()) != digest(base(crosses_cut=True))
    other = base()
    other[0].text = "xin chao"
    assert digest(base()) != digest(other)


# --- style ----------------------------------------------------------------------------

def test_sentence_split_keeps_decimals_and_drops_the_unfinished_tail():
    out = svc_mod.split_sentences("Tôi đi chợ. Bạn có đi không? Giá 3.5 triệu nhé! và còn nữa")
    assert out == ["Tôi đi chợ.", " Bạn có đi không?", " Giá 3.5 triệu nhé!"]
    assert svc_mod.sentence_word_count(" Giá 3.5 triệu nhé!") == 4
    assert svc_mod.split_sentences("TP.HCM rất đông") == []


def _talk(rows):
    """Alternating long turns so nothing is a backchannel; returns a prepared list."""
    return build([seg(i, sp, st, en, tx) for i, (sp, st, en, tx) in enumerate(rows)])


def test_style_counts_every_dictionary_form_and_particles_only_at_sentence_end(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    monkeypatch.setattr(svc_mod, "MIN_DIST_N", 1)
    prepared = _talk([
        (A, 0.0, 6.0, "Tôi đi chợ. Bạn có đi không? Giá 3.5 triệu nhé!"),
        (B, 7.0, 12.0, "ừm ờm em thấy ổn lắm ạ."),
    ])
    measures = svc_mod.measure_style(prepared)
    a = measures[A].style
    assert [f.form for f in a.fillers] == list(svc_mod.FILLERS)
    assert [p.form for p in a.particles] == list(svc_mod.PARTICLES)
    assert [t.form for t in a.address_terms] == list(svc_mod.ADDRESS_TERMS)
    assert {p.form: p.per_100_sent for p in a.particles}["nhé"] == pytest.approx(33.333, abs=1e-3)
    assert {p.form: p.per_100_sent for p in a.particles}["ạ"] == 0.0     # "bạn" is not "ạ"
    assert {t.form: t.count for t in a.address_terms}["bạn"] == 1
    assert measures[A].n_sentences == 3 and a.words_per_sentence.n == 3
    b = measures[B].style
    assert {f.form: f.per_min for f in b.fillers}["ừm"] > 0
    assert {t.form: t.count for t in b.address_terms}["em"] == 1
    assert {p.form: p.per_100_sent for p in b.particles}["ạ"] == 100.0
    assert a.llm_candidate_forms is None and a.description is None      # filled later


def test_low_exposure_nulls_dictionary_lists_but_keeps_counts_of_samples():
    prepared = _talk([(A, 0.0, 4.0, "ừm tôi đi chợ nhé."), (B, 5.0, 9.0, "dạ")])
    m = svc_mod.measure_style(prepared)[A]
    assert m.speech_min < svc_mod.MIN_EXPOSURE_MIN
    assert m.style.fillers is None and m.style.address_terms is None
    assert m.style.particles is None                          # n_sentences < MIN_DIST_N
    assert m.style.words_per_sentence.n == 1 and m.style.words_per_sentence.p50 is None


def test_nfd_text_matches_the_dictionary(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    nfd = unicodedata.normalize("NFD", "ừm tôi nghĩ vậy anh nhé.")
    segs = svc_mod.prepare_segments([seg(0, A, 0.0, 5.0, nfd), seg(1, B, 6.0, 9.0, "ok bạn")], None)
    a = svc_mod.measure_style(segs)[A].style
    assert {f.form: f.per_min for f in a.fillers}["ừm"] > 0
    assert {t.form: t.count for t in a.address_terms}["anh"] == 1


def test_excluded_segments_do_not_enter_style_but_stay_as_context(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    prepared = _talk([(A, 0.0, 5.0, "tôi đi chợ nhé."), (A, 6.0, 11.0, "anh ơi nhé.")])
    prepared[1].has_music = True
    a = svc_mod.measure_style(prepared)[A].style
    assert {t.form: t.count for t in a.address_terms}["anh"] == 0


def test_speech_rate_uses_long_enough_units_and_clips_outliers(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_DIST_N", 1)
    prepared = _talk([
        (A, 0.0, 4.0, words(8)),          # 2.0 w/s -> kept
        (B, 5.0, 5.9, words(4)),          # < 1.0 s -> skipped
        (A, 7.0, 8.0, words(10)),         # 10 w/s -> outlier, dropped
        (B, 9.0, 12.0, "một hai"),        # < 3 words -> skipped
    ])
    m = svc_mod.measure_style(prepared)
    assert m[A].style.speech_rate_wps.n == 1 and m[A].style.speech_rate_wps.p50 == 2.0
    assert m[B].style.speech_rate_wps.n == 0


# --- interaction ----------------------------------------------------------------------

def _alt(starts, speakers=(A, B), length=4.5, text=None):
    """Alternating turns at the given starts, each `length` s long."""
    return [seg(i, speakers[i % 2], st, st + length, text or words(5)) for i, st in enumerate(starts)]


def test_gap_before_none_splits_regions_and_is_never_a_latency():
    rows = build(_alt([0, 5.4, 9.8, 14.2, 18.6]))          # gaps 0.9 apart
    acc, unknown = svc_mod.measure_interaction(rows)
    assert unknown == 0
    assert len(acc[B].latencies) == 2 and len(acc[A].latencies) == 1
    rows = build(_alt([0, 5.4, 9.8, 14.2, 18.6]), gaps={1: None})
    acc, unknown = svc_mod.measure_interaction(rows)
    assert unknown == 1
    assert len(acc[B].latencies) == 1                       # the seam into B1 gave nothing
    assert all(v >= svc_mod.LATENCY_FLOOR_S for a in acc.values() for v in a.latencies.values())


def test_seam_next_to_an_empty_segment_still_separates_the_sides():
    rows = build([seg(0, A, 0, 5, words(5)), seg(1, "E", 7, 7.5, ""),
                  seg(2, B, 8, 12, words(5)), seg(3, A, 12.5, 17, words(5)),
                  seg(4, B, 17.5, 22, words(5)), seg(5, A, 22.5, 27, words(5))],
                 gaps={1: None})
    acc, unknown = svc_mod.measure_interaction(rows)
    assert unknown == 1
    assert 3.0 not in {round(v, 3) for a in acc.values() for v in a.latencies.values()}


def test_crosses_cut_segment_stands_alone_and_masks_its_overlap():
    # A[0,10] crosses a cut; B[5,6] talks inside it; A2 follows right after.
    rows = build([seg(0, A, 0, 10, words(10), crosses_cut=True), seg(1, B, 5, 6, words(3)),
                  seg(2, A, 10.1, 14, words(8)), seg(3, B, 14.5, 18, words(6)),
                  seg(4, A, 18.5, 22, words(6)), seg(5, B, 22.5, 26, words(6)),
                  seg(6, A, 26.5, 30, words(6))], gaps={1: None})
    acc, _ = svc_mod.measure_interaction(rows)
    assert not any(abs(v - 4.1) < 1e-6 for a in acc.values() for v in a.latencies.values())
    assert acc[B].other_talk_s <= 12.0 and acc[A].other_talk_s >= 0
    # B[5,6] lies in the crossing segment's span, so it is not A's listening time.
    assert acc[A].other_talk_s < 5.0 + 3.5 + 3.5 + 3.5


def test_latency_rules():
    # A talks to 5.0, B comes in at 4.8: -0.2 (turn_switch would not see it).
    rows = build([seg(0, A, 0, 5.0, words(8)), seg(1, B, 4.8, 9.0, words(8)),
                  seg(2, A, 9.5, 14.0, words(8)), seg(3, B, 14.5, 19.0, words(8)),
                  seg(4, A, 19.5, 24.0, words(8))])
    acc, _ = svc_mod.measure_interaction(rows)
    assert sorted(round(v, 3) for v in acc[B].latencies.values())[0] == -0.2


def test_latency_drops_a_unit_inside_the_other_voice_and_out_of_range_values():
    rows = build([seg(0, A, 0, 20.0, words(30)), seg(1, B, 2.0, 4.0, words(5)),     # B inside A
                  seg(2, A, 21.0, 26.0, words(8)), seg(3, B, 40.0, 44.0, words(8)),  # +14 s: too slow
                  seg(4, A, 44.5, 49.0, words(8)), seg(5, B, 49.5, 54.0, words(8)),
                  seg(6, A, 54.5, 59.0, words(8))])
    acc, _ = svc_mod.measure_interaction(rows)
    for a in acc.values():
        assert all(svc_mod.LATENCY_FLOOR_S <= v <= svc_mod.LATENCY_CAP_S
                   for v in a.latencies.values())
    assert 14.0 not in {round(v, 3) for a in acc.values() for v in a.latencies.values()}


def test_a_cut_in_that_returns_to_the_holder_is_not_a_latency():
    # X (A) stops at 5.0; B cuts in [4.5, 5.6] with 3 non-backchannel words; A goes on.
    rows = build([seg(0, A, 0, 5.0, words(8)), seg(1, B, 4.5, 5.6, words(3)),
                  seg(2, A, 6.0, 10.0, words(8)), seg(3, B, 10.5, 15.0, words(8)),
                  seg(4, A, 15.5, 20.0, words(8)), seg(5, B, 20.5, 25.0, words(8)),
                  seg(6, A, 25.5, 30.0, words(8))])
    acc, _ = svc_mod.measure_interaction(rows)
    # A's answer after B's short cut-in is not "taking the turn": its key (6.0, 10.0) is absent.
    assert (A, 6.0, 10.0) not in acc[A].latencies


def test_three_speakers_pairs_are_masked_and_deduplicated():
    rows = build([seg(0, A, 0, 6, words(8)), seg(1, C, 6.5, 7.2, "ừ"),
                  seg(2, A, 7.4, 13, words(8)), seg(3, B, 13.5, 18, words(8)),
                  seg(4, A, 18.5, 24, words(8)), seg(5, B, 24.5, 30, words(8)),
                  seg(6, A, 30.5, 36, words(8))])
    acc, _ = svc_mod.measure_interaction(rows)
    keys = [k for a in acc.values() for k in a.latencies]
    assert len(keys) == len(set(keys))
    # Exposure of the two backchannel-type rates uses the same other_talk_s.
    assert acc[A].other_talk_s > 0 and acc[C].other_talk_s >= 0


def test_backchannel_events_in_turn_lexicon_and_overlap(monkeypatch):
    # A talks long; B says "ừ" x3 and "dạ" x1 while A goes on, plus one while A is quiet.
    rows = [seg(0, A, 0, 30, words(40))]
    for i, (st, tx) in enumerate([(5.0, "ừ"), (10.0, "ừ"), (15.0, "dạ"), (20.0, "ừ")], start=1):
        rows.append(seg(i, B, st, st + 0.4, tx))
    rows.append(seg(5, A, 32, 40, words(10)))
    rows.append(seg(6, B, 45, 45.4, "ừ"))                      # after A stopped: not in a turn
    rows.append(seg(7, A, 47, 52, words(8)))
    acc, _ = svc_mod.measure_interaction(build(rows))
    kinds = [e["text"] for e in acc[B].backchannels.values()]
    assert kinds.count("ừ") == 3 and kinds.count("dạ") == 1
    assert all(e["in_turn"] for e in acc[B].backchannels.values())


def test_interruption_and_yield_counts_use_cut_ins_on_the_speaker():
    # B cuts into A twice; A yields once (stops) and once keeps talking (competitive).
    rows = build([seg(0, A, 0, 6.0, words(10)), seg(1, B, 5.0, 12.0, words(12)),
                  seg(2, A, 20, 30.0, words(14)), seg(3, B, 24.0, 34.0, words(14)),
                  seg(4, A, 34.2, 44.0, words(14)), seg(5, B, 44.5, 52.0, words(10)),
                  seg(6, A, 52.5, 60.0, words(10))])
    acc, _ = svc_mod.measure_interaction(rows)
    types = sorted(e["type"] for e in acc[A].cut_ins.values())
    assert "interruption" in types
    assert all(e["by"] == B for e in acc[A].cut_ins.values())
    assert all(e["by"] == B for e in acc[B].interruptions.values())


# --- assembly -------------------------------------------------------------------------

def _no_thresholds(monkeypatch):
    for name in ("MIN_EXPOSURE_MIN", "MIN_EXPOSURE_INTERRUPTION_MIN"):
        monkeypatch.setattr(svc_mod, name, 0.0)
    for name in ("MIN_DIST_N", "MIN_RATIO_N", "MIN_LEXICON_N"):
        monkeypatch.setattr(svc_mod, name, 1)


def _talk_file():
    rows = []
    t = 0.0
    for i in range(10):
        rows.append(seg(2 * i, A, t, t + 4.0, "Tôi đi chợ nhé. ừm anh thấy sao?"))
        rows.append(seg(2 * i + 1, B, t + 4.5, t + 8.5, "Em thấy ổn ạ. Dạ vâng."))
        t += 9.0
    return build(rows)


def test_compute_core_has_every_key_and_exact_stats(monkeypatch):
    _no_thresholds(monkeypatch)
    core = svc_mod.compute_core(_talk_file(), "ep01")
    d = json.loads(json.dumps(core.profile.to_dict(), allow_nan=False))
    assert d["schema_version"] == 1 and d["source_type"] == "real"
    assert d["params_version"] == svc_mod.PARAMS_VERSION and d["recording_id"] == "ep01"
    assert d["usable_ratio"] == 1.0 and set(d["speakers"]) == {A, B}
    for speaker in d["speakers"].values():
        assert set(speaker) == {"speech_s", "style", "backchannel", "interruption", "turn"}
        assert set(speaker["style"]) == {
            "speech_rate_wps", "words_per_sentence", "fillers", "particles", "address_terms",
            "llm_candidate_forms", "description"}
        assert speaker["style"]["llm_candidate_forms"] is None
    assert d["speakers"][A]["speech_s"] == 40.0
    s = core.stats.to_dict()
    assert set(s["excluded_s"]) == set(svc_mod.EXCLUSION_KEYS) and s["llm_windows"] == 0
    assert s["warnings"][0].startswith("backchannel_and_interruption_are_lower_bounds")


def test_single_speaker_has_null_interaction(monkeypatch):
    _no_thresholds(monkeypatch)
    rows = build([seg(0, A, 0, 5, "tôi đi chợ."), seg(1, A, 6, 11, "anh nhé.")])
    d = svc_mod.compute_core(rows, "solo").profile.to_dict()["speakers"][A]
    assert d["backchannel"]["per_min_of_other"] == {"value": None, "count": 0, "exposure_min": 0.0}
    assert d["interruption"]["yield_rate"]["value"] is None
    assert d["turn"]["response_latency_s"]["n"] == 0
    assert d["style"]["fillers"] is not None            # style still measured


def test_low_data_speaker_keeps_keys_with_nulls_and_real_counts():
    rows = _talk_file()
    d = svc_mod.compute_core(rows, "ep").profile.to_dict()["speakers"][B]
    assert d["backchannel"]["per_min_of_other"]["value"] is None
    assert d["backchannel"]["per_min_of_other"]["exposure_min"] > 0
    assert d["backchannel"]["lexicon"] is None
    assert d["style"]["fillers"] is None                 # 40 s of speech < 5 min


def test_warnings_and_usable_ratio():
    rows = build([seg(0, A, 0, 10, "tôi đi chợ nhé", review_flags=[{"kind": "asr_garbage"}]),
                  seg(1, B, 10.5, 12, "dạ vâng ạ")])
    stats = svc_mod.compute_core(rows, "ep").stats
    assert any(w.startswith("not_reviewed: 1 segment(s)") for w in stats.warnings)
    assert any(w.startswith("noise_unmeasured: 2 segment(s)") for w in stats.warnings)
    assert any(w.startswith("low_usable_ratio: 0.13") for w in stats.warnings)
    assert stats.excluded_s["review_flag"] == 10.0


# --- LLM ------------------------------------------------------------------------------

class FakeLLM:
    model_name = "fake/model"

    def __init__(self, reply='{"new_forms": [], "description": "Nói chậm."}', loaded=True):
        self.reply, self.loaded = reply, loaded
        self.batch_size, self.max_batch_tokens = 100, 0
        self.calls, self.system_prompts = [], []

    def ensure_loaded(self):
        return self.loaded

    def count_tokens(self, text):
        return len(text.split())

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None, **extra):
        self.calls.append(list(user_messages))
        self.system_prompts.append(system_prompt)
        out = [self.reply(m) if callable(self.reply) else self.reply for m in user_messages]
        return True, out


def _service(llm, **kw):
    return svc_mod.PersonaProfileService(llm, **kw)


def _prepared_talk():
    rows, t = [], 0.0
    for i in range(10):
        rows.append(seg(2 * i, A, t, t + 4.0, "Tôi thấy tức là bạn đúng. ừm anh nhé."))
        rows.append(seg(2 * i + 1, B, t + 4.5, t + 8.5, "Em thấy ổn ạ. Dạ vâng."))
        t += 9.0
    return svc_mod.prepare_segments(rows, None)


def test_constructor_rejects_unknown_keys_and_exposes_the_contract():
    with pytest.raises(TypeError):
        _service(FakeLLM(), window_tokens=5)
    s = _service(FakeLLM())
    assert (s.llm_enabled, s.thinking, s.max_new_tokens) == (True, False, 1024)
    assert _service(FakeLLM(), llm_enabled=False).llm_enabled is False


def test_namespace_changes_with_every_part_and_is_path_safe():
    base = _service(FakeLLM()).checkpoint_namespace_for("d1")
    assert base != _service(FakeLLM()).checkpoint_namespace_for("d2")
    assert base != _service(FakeLLM(), llm_enabled=False).checkpoint_namespace_for("d1")
    assert base != _service(FakeLLM(), thinking=True).checkpoint_namespace_for("d1")
    assert base != _service(FakeLLM(), max_new_tokens=2048).checkpoint_namespace_for("d1")
    assert svc_mod.PROFILE_PROMPT_VERSION in base and svc_mod.PARAMS_VERSION in base
    assert re.fullmatch(r"[A-Za-z0-9._-]+", base)


def test_llm_candidate_forms_validation():
    texts = ["tức là bạn đúng", "ừm tôi nghĩ vậy"]
    kept = svc_mod.validate_forms(["tức là", "xyz không có", "ạ", "ừm", "Tức Là", "bạn đúng"],
                                  texts, svc_mod.FILLERS + svc_mod.PARTICLES + svc_mod.ADDRESS_TERMS)
    assert kept == ["bạn đúng", "tức là"]              # sorted, deduplicated; "ạ" is a dictionary form
    assert svc_mod.validate_forms(["a b c d"], ["a b c d"], ()) == []      # more than 3 tokens
    assert svc_mod.validate_forms(["ạn"], ["bạn đúng"], ()) == []          # substring of a token
    many = [f"w{k}" for k in range(40)]
    assert len(svc_mod.validate_forms(many, [" ".join(many)], ())) == svc_mod.MAX_CANDIDATE_FORMS


def test_description_rules():
    assert svc_mod.validate_description("  Nói nhanh. ", 50) == "Nói nhanh."
    assert svc_mod.validate_description("", 50) is None
    assert svc_mod.validate_description("x" * 401, 50) is None
    assert svc_mod.validate_description("ok", svc_mod.MIN_DIST_N - 1) is None
    assert svc_mod.validate_description(None, 50) is None


def test_pick_window_is_even_and_bounded():
    texts = [f"đoạn {k:03d} " + "x" * 20 for k in range(100)]
    window = svc_mod.pick_window(texts, 500)
    assert len(window) <= 500 and window.startswith(texts[0])
    assert svc_mod.pick_window(["ngắn"], 500) == "ngắn"
    assert len(svc_mod.pick_window(["y" * 900], 500)) == 500


def test_llm_forms_description_and_counts_are_separate(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    monkeypatch.setattr(svc_mod, "MIN_DIST_N", 1)       # description needs rate n >= MIN_DIST_N
    reply = '{"new_forms": ["tức là", "không có trong lời", "ừm"], "description": "Nói chậm."}'
    llm = FakeLLM(reply)
    with_llm = _service(llm).profile(_prepared_talk(), "ep")
    off_llm = FakeLLM()
    without = _service(off_llm, llm_enabled=False).profile(_prepared_talk(), "ep")
    a = with_llm.profile["speakers"][A]["style"]
    assert a["llm_candidate_forms"] == ["tức là"] and a["description"] == "Nói chậm."
    assert with_llm.stats["llm_windows"] == 2 and with_llm.stats["llm_ok"] == 2
    assert with_llm.complete is True
    off = without.profile["speakers"][A]["style"]
    assert off["llm_candidate_forms"] is None and off["description"] is None
    assert without.complete is True and llm.calls and not off_llm.calls
    for sp in (A, B):                                  # numbers identical with or without the LLM
        x, y = with_llm.profile["speakers"][sp], without.profile["speakers"][sp]
        for part in ("fillers", "particles", "address_terms", "speech_rate_wps",
                     "words_per_sentence"):
            assert x["style"][part] == y["style"][part]
        assert x["backchannel"] == y["backchannel"] and x["turn"] == y["turn"]


def test_empty_forms_list_is_not_null(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    res = _service(FakeLLM('{"new_forms": [], "description": "ok"}')).profile(_prepared_talk(), "ep")
    assert res.profile["speakers"][A]["style"]["llm_candidate_forms"] == []


def test_unreadable_reply_retries_once_then_gives_null_and_is_not_complete(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    llm = FakeLLM("blah blah không phải json")
    res = _service(llm).profile(_prepared_talk(), "ep")
    st = res.profile["speakers"][A]["style"]
    assert st["llm_candidate_forms"] is None and st["description"] is None
    assert res.stats["llm_windows"] == 2 and res.stats["llm_ok"] == 0
    assert res.complete is False
    assert len(llm.calls) == 2                          # first ask + one strict retry
    assert any(w.startswith("llm_unavailable") for w in res.stats["warnings"])
    assert "JSON" in llm.system_prompts[1]


def test_llm_not_loaded_is_not_complete_but_numbers_are_written(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    res = _service(FakeLLM(loaded=False)).profile(_prepared_talk(), "ep")
    assert res.complete is False and res.profile["speakers"][A]["style"]["fillers"] is not None
    assert any(w.startswith("llm_unavailable") for w in res.stats["warnings"])


def test_speakers_without_exposure_are_not_asked():
    llm = FakeLLM()
    res = _service(llm).profile(_prepared_talk(), "ep")          # 40 s < 5 min each
    assert not llm.calls and res.stats["llm_windows"] == 0
    assert res.profile["speakers"][A]["style"]["description"] is None
    assert res.complete is True


# --- end to end -----------------------------------------------------------------------

def test_empty_input_is_skipped():
    res = _service(FakeLLM()).profile([], "ep")
    assert res.skipped == "no_segments" and res.complete is False
    assert res.profile == {} and res.stats == {}


def test_profile_is_deterministic_and_does_not_touch_the_input(monkeypatch):
    monkeypatch.setattr(svc_mod, "MIN_EXPOSURE_MIN", 0.0)
    rows = [seg(i, A if i % 2 == 0 else B, i * 5.0, i * 5.0 + 4.0,
                "Tôi đi chợ nhé. ừm anh thấy sao?") for i in range(12)]
    snapshot = [(s.words, s.gap_before, s.noise_breakdown, s.text, s.speaker) for s in rows]
    svc = _service(FakeLLM())
    first = svc.profile(svc.prepare(rows, None), "ep")
    second = svc.profile(svc.prepare(rows, None), "ep")
    dump = lambda r: json.dumps(r.profile, ensure_ascii=False, indent=2, allow_nan=False)
    assert dump(first) == dump(second)
    assert json.dumps(first.stats, sort_keys=True) == json.dumps(second.stats, sort_keys=True)
    assert [(s.words, s.gap_before, s.noise_breakdown, s.text, s.speaker) for s in rows] == snapshot
    digest = svc.source_digest(svc.prepare(rows, None))
    assert digest == svc.source_digest(svc.prepare(rows, None))
    assert svc.checkpoint_namespace_for(digest) == svc.checkpoint_namespace_for(digest)
