"""Picking two-person conversation excerpts out of a transcript.

The rules come from conversation_dataset_plan.md; what is pinned here is the
part the plan cannot say on its own: that an excerpt is exactly two people, that a
break point (a failed separation or a review flag) is never inside an excerpt, and
that noise and music, no longer measured, neither gate nor score a clip.

Run:  python -m pytest tests/test_conversation_selection.py -q     (from podcast-pipeline/)
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_selection import (
    BACKCHANNEL, INVALID, NORMAL, Candidate, ConversationSelectionConfig, ConversationSelectionFinder,
    classify, overlap_fraction, pick_non_overlapping, shortlist, tier_of)
from utils.excise import TimelineMap
from utils.music_map import MUSIC, MusicMap
from utils.noise_map import KINDS, NoiseTrack

A, B, C = "SPEAKER_00", "SPEAKER_01", "SPEAKER_02"


# --- builders ----------------------------------------------------------------

def _seg(i, speaker, start, end, text=None, **kw):
    return SimpleNamespace(
        index=f"{i:05d}", speaker=speaker, start=start, end=end,
        text=text if text is not None else f"đoạn {i} bàn về chuyện số {i} rất dài dòng",
        bss=False, unseparated=None, **kw)


def _talk(n, start=0.0, seg=7.0, gap=1.0, speakers=(A, B), first=0):
    """n alternating turns of `seg` seconds, `gap` apart."""
    out, t = [], start
    for k in range(n):
        out.append(_seg(first + k, speakers[k % 2], t, t + seg))
        t += seg + gap
    return out


def _track(total=600, base=0.001, **bursts):
    """A measured noise track in ORIGINAL time; bursts are (start, end, level)."""
    curves = {k: np.full(int(total * 100), base, dtype=np.float32) for k in KINDS}
    for kind, spans in bursts.items():
        for a, b, level in spans:
            curves[kind][int(a * 100):int(b * 100)] = level
    return NoiseTrack(curves, fps=100.0)


def _finder(segs, noise="quiet", timeline=None, music=None, **cfg):
    if isinstance(noise, str):
        noise = _track() if noise == "quiet" else NoiseTrack()
    return ConversationSelectionFinder(segs, timeline, noise, music, ConversationSelectionConfig(**cfg))


def _covers(cand, pos):
    return cand.first <= pos <= cand.last


# --- one segment ---------------------------------------------------------------

def test_a_short_listener_noise_is_a_backchannel_not_a_turn():
    assert classify(_seg(0, A, 0, 0.5, "ừ"))[0] == BACKCHANNEL
    assert classify(_seg(0, A, 0, 0.8, "Dạ vâng."))[0] == BACKCHANNEL


def test_the_same_word_held_for_a_second_or_more_is_a_real_turn():
    assert classify(_seg(0, A, 0, 1.5, "ừ"))[0] == NORMAL


def test_empty_or_punctuation_only_text_is_no_speech():
    assert classify(_seg(0, A, 0, 3, "")) == (INVALID, "no_speech")
    assert classify(_seg(0, A, 0, 3, "...")) == (INVALID, "no_speech")


def test_a_canned_outro_is_a_hallucination():
    assert classify(_seg(0, A, 0, 3, "Cảm ơn các bạn đã theo dõi")) == (INVALID, "hallucination")


def test_a_tiny_clip_that_is_not_a_backchannel_is_too_short():
    assert classify(_seg(0, A, 0, 0.2, "xin chào")) == (INVALID, "too_short")


def test_an_ordinary_sentence_is_normal():
    assert classify(_seg(0, A, 0, 6.0))[0] == NORMAL


# --- what a clip is ---------------------------------------------------------------

def test_a_two_person_conversation_yields_clips_of_the_right_length():
    cands = _finder(_talk(16)).candidates()
    assert cands
    for c in cands:
        assert 60.0 <= c.duration <= 240.0
        assert set(c.speakers) == {A, B}
        assert c.metrics["speaker_count"] == 2


def test_a_third_speaker_ends_the_block_and_is_never_inside_a_clip():
    segs = _talk(12) + [_seg(12, C, 96.0, 103.0)] + _talk(12, start=104.0, first=13)
    finder = _finder(segs)
    cands = finder.candidates()
    assert cands
    assert finder.stats["breaks"]["third_speaker"] >= 1
    assert not any(_covers(c, 12) for c in cands)
    assert all(C not in c.speakers for c in cands)


def test_a_long_pause_ends_the_block():
    segs = _talk(12) + _talk(12, start=96.0 + 5.0, first=12)
    finder = _finder(segs)
    cands = finder.candidates()
    assert finder.stats["breaks"]["silence"] == 1
    assert cands and not any(c.first < 12 <= c.last for c in cands)


def test_one_person_talking_too_long_ends_the_block():
    segs = (_talk(10) + [_seg(10 + k, A, 80.0 + k * 7.5, 80.0 + k * 7.5 + 7.0)
                         for k in range(8)])
    finder = _finder(segs)
    cands = finder.candidates()
    assert finder.stats["breaks"]["monologue"] >= 1
    assert all(c.metrics["max_monologue"] <= 45.0 for c in cands)


def test_a_monologue_limit_of_zero_means_no_limit():
    segs = (_talk(10) + [_seg(10 + k, A, 80.0 + k * 7.5, 80.0 + k * 7.5 + 7.0)
                         for k in range(8)])
    limited, free = _finder(segs), _finder(segs, max_monologue=0)
    assert limited.stats["breaks"].get("monologue", 0) >= 1
    assert free.stats["breaks"].get("monologue", 0) == 0
    assert len(free.blocks) < len(limited.blocks)


def test_a_block_that_is_too_short_gives_nothing_and_is_counted():
    finder = _finder(_talk(6))
    assert finder.candidates() == []
    assert finder.stats["blocks_short"] == 1


def test_clips_start_and_stop_on_a_turn_or_a_pause_not_mid_speech():
    """A speaker continuing after a beat is one utterance, not a place to cut."""
    segs, t = [], 0.0
    for k in range(36):
        speaker = A if (k // 3) % 2 == 0 else B
        segs.append(_seg(k, speaker, t, t + 5.0))
        t += 5.0 + 0.3
    cands = _finder(segs).candidates()
    assert cands
    for c in cands:
        first, last = segs[c.first], segs[c.last]
        before = segs[c.first - 1] if c.first else None
        after = segs[c.last + 1] if c.last + 1 < len(segs) else None
        assert before is None or before.speaker != first.speaker
        assert after is None or after.speaker != last.speaker


def test_a_long_recording_gives_only_clips_inside_the_length_window():
    cands = _finder(_talk(120)).candidates()
    assert len(cands) > 1
    assert all(60.0 <= c.duration <= 240.0 for c in cands)
    picked = pick_non_overlapping(cands)
    for a, b in zip(picked, picked[1:]):
        assert a.end <= b.start


def test_a_listener_who_only_says_uh_huh_is_not_a_second_speaker():
    segs = []
    t = 0.0
    for k in range(20):
        segs.append(_seg(2 * k, A, t, t + 8.0))
        segs.append(_seg(2 * k + 1, B, t + 8.2, t + 8.7, "ừ"))
        t += 9.0
    finder = _finder(segs)
    assert finder.candidates() == []
    # The listener's "ừ" never breaks the speaker's monologue (plan section 3),
    # so the block ends on that rule before a second speaker is ever counted.
    assert finder.stats["breaks"]["monologue"] >= 1


def test_a_second_speaker_with_a_single_real_turn_is_not_a_conversation():
    run = [_seg(k, A, k * 7.5, k * 7.5 + 7.0) for k in range(6)]          # 44.5s
    reply = [_seg(6, B, 45.5, 52.5)]
    more = [_seg(7 + k, A, 53.0 + k * 7.5, 53.0 + k * 7.5 + 7.0) for k in range(6)]
    finder = _finder(run + reply + more)
    assert finder.candidates() == []
    assert finder.stats["candidates_rejected"]["too_few_turns"] >= 1


def test_one_turn_each_is_enough_when_the_setting_says_so():
    """An interview with one long answer is a question and its answer."""
    run = [_seg(k, A, k * 7.5, k * 7.5 + 7.0) for k in range(6)]
    reply = [_seg(6, B, 45.5, 52.5)]
    more = [_seg(7 + k, A, 53.0 + k * 7.5, 53.0 + k * 7.5 + 7.0) for k in range(6)]
    finder = _finder(run + reply + more, min_turns_each=1)
    assert finder.candidates()
    assert finder.stats["candidates_rejected"].get("too_few_turns", 0) == 0


def test_a_b_a_with_a_backchannel_is_a_minimum_conversation():
    segs = [
        _seg(0, A, 0.0, 8.0),
        _seg(1, B, 8.2, 8.7, "ừ"),
        _seg(2, A, 8.9, 16.9),
    ]
    finder = _finder(
        segs,
        min_seconds=10.0,
        max_seconds=30.0,
        min_turns_each=1,
        min_turns_total=3,
        count_backchannels_as_turns=True,
    )
    assert finder.evaluate(0, 2)[0] is not None


def test_a_b_without_a_return_fails_the_three_turn_policy():
    segs = [_seg(0, A, 0.0, 8.0), _seg(1, B, 8.2, 8.7, "ừ")]
    finder = _finder(
        segs,
        min_seconds=5.0,
        max_seconds=30.0,
        min_turns_each=1,
        min_turns_total=3,
        count_backchannels_as_turns=True,
    )
    assert finder.evaluate(0, 1)[1] == "too_few_turns"


def test_a_hallucinated_segment_ends_the_block():
    segs = _talk(14)
    segs[7] = _seg(7, B, 56.0, 63.0, "Hãy subscribe cho kênh")
    finder = _finder(segs)
    finder.candidates()
    assert finder.stats["breaks"]["invalid_hallucination"] == 1
    assert finder.stats["invalid"]["hallucination"] == 1


def test_overlapping_speech_is_allowed_and_counted():
    segs = _talk(16)
    for k in range(1, 16, 2):
        segs[k].start -= 1.5            # the second speaker cuts in early
    finder = _finder(segs)
    cands = finder.candidates()
    assert cands
    assert max(c.metrics["overlap_count"] for c in cands) > 0


def test_speech_that_could_not_be_separated_costs_points_but_is_not_a_reason_to_drop():
    segs = _talk(16)
    for s in segs:
        s.unseparated = [{"start": s.start, "end": s.start + 1.5}]
    cands = _finder(segs).candidates()
    assert cands
    assert all(c.components["speaker_correctness"] < 1.0 for c in cands)


# --- joins -----------------------------------------------------------------------

def test_a_cut_in_the_timeline_no_longer_breaks_a_block():
    """Music is no longer cut out, so a join is not a break point: the block runs across it."""
    timeline = TimelineMap(kept=[(0, 95.5, 0), (200, 500, 95.5)])
    finder = _finder(_talk(30), timeline=timeline)
    cands = finder.candidates()
    assert "seam" not in finder.stats["breaks"]
    assert "seam_inside_segment" not in finder.stats["breaks"]
    assert any(c.pad_start < 95.5 < c.pad_end for c in cands)


# --- padding -----------------------------------------------------------------------

def test_padding_adds_a_little_air_where_there_is_room():
    finder = _finder(_talk(16))
    cand, why = finder.evaluate(2, 13)
    assert cand is not None, why
    assert cand.pad_start == pytest.approx(cand.start - 0.2)
    assert cand.pad_end == pytest.approx(cand.end + 0.2)


def test_padding_stops_at_the_neighbouring_turn():
    segs = _talk(16, gap=0.1)
    finder = _finder(segs)
    cand, why = finder.evaluate(2, 13)
    assert cand is not None, why
    assert cand.pad_start == pytest.approx(cand.start - 0.1)


def test_padding_is_not_added_when_the_neighbour_overlaps():
    segs = _talk(16, gap=-0.5)
    cand, why = _finder(segs).evaluate(2, 13)
    assert cand is not None, why
    assert cand.pad_start == cand.start




# --- noise and music are no longer inputs -------------------------------------------------

def test_a_noisy_track_no_longer_changes_which_clips_are_found():
    quiet = _finder(_talk(16), noise=_track()).candidates()
    loud = _finder(_talk(16), noise=_track(noise_speech=[(0, 600, 0.9)],
                                           noise_env=[(0, 600, 0.9)])).candidates()
    assert quiet and [(c.first, c.last) for c in loud] == [(c.first, c.last) for c in quiet]
    assert all(c.noise == {} for c in loud)


def test_an_unmeasured_track_is_not_a_reason_to_refuse():
    finder = _finder(_talk(16), noise="none")
    assert finder.candidates()
    assert "noise_measured" not in finder.report()
    assert "lasting_noise_spans" not in finder.report()


def test_the_music_map_is_ignored_and_no_share_is_reported():
    music = MusicMap([(0.0, 40.0, MUSIC)])
    finder = _finder(_talk(30), music=music)
    cands = finder.candidates()
    assert cands
    assert all(c.music_patched_share is None for c in cands)
    assert "music_patched" not in finder.report()["candidates_rejected"]

# --- scoring -----------------------------------------------------------------------------

def _components(**metrics):
    base = {"duration": 100.0, "turn_latency_avg": 0.5, "speaker_balance": 1.0,
            "switch_count": 20, "backchannel_count": 0, "silence_ratio": 0.0,
            "unseparated_share": 0.0}
    base.update(metrics)
    return _finder(_talk(2))._components(base)


def test_a_natural_turn_gap_scores_full_and_a_slow_or_interrupting_one_less():
    assert _components(turn_latency_avg=0.5)["turn_taking"] == 1.0
    assert _components(turn_latency_avg=2.0)["turn_taking"] == pytest.approx(0.5)
    assert _components(turn_latency_avg=3.5)["turn_taking"] == 0.0
    assert _components(turn_latency_avg=-0.2)["turn_taking"] == pytest.approx(0.5)
    assert _components(turn_latency_avg=None)["turn_taking"] == 0.0


def test_length_scores_full_on_the_plateau_and_less_toward_the_edges():
    assert _components(duration=120.0)["duration"] == 1.0
    assert _components(duration=60.0)["duration"] == pytest.approx(0.7)
    assert _components(duration=240.0)["duration"] == pytest.approx(0.7)
    assert _components(duration=75.0)["duration"] == pytest.approx(0.85)


def test_tiers():
    assert [tier_of(s) for s in (90, 85, 84.9, 70, 55, 40, 39.9)] == [
        "S", "S", "A", "A", "B", "C", "Reject"]


def test_weights_are_normalised_so_a_score_never_passes_100():
    finder = _finder(_talk(2))
    comps = {k: 1.0 for k in ("turn_taking", "duration", "balance", "interaction",
                              "continuity", "speaker_correctness")}
    assert finder._score(comps) == 100.0


def test_an_old_cleanliness_weight_is_accepted_and_ignored():
    """Old profiles still load; the cleanliness term is gone, so its weight changes nothing."""
    comps = {k: 1.0 for k in ("turn_taking", "duration", "balance", "interaction",
                              "continuity", "speaker_correctness")}
    plain = _finder(_talk(2))._score(comps)
    old = _finder(_talk(2), weights={"cleanliness": 1000.0})
    assert old._score(comps) == plain == 100.0
    assert "cleanliness" not in old.cfg.resolved_weights


def test_low_interaction_is_soft_penalized_when_its_weight_is_increased():
    comps = {k: 1.0 for k in ("turn_taking", "duration", "balance", "interaction",
                              "continuity", "speaker_correctness")}
    comps["interaction"] = 0.0
    baseline = _finder(_talk(2))._score(comps)
    tuned = _finder(_talk(2), weights={"interaction": 25.0})._score(comps)
    assert tuned < baseline and tuned > 55.0


def test_the_configuration_refuses_what_it_does_not_know():
    with pytest.raises(TypeError):
        ConversationSelectionConfig(min_secs=30)
    with pytest.raises(ValueError):
        ConversationSelectionConfig(weights={"vibes": 1})
    with pytest.raises(ValueError):
        ConversationSelectionConfig(min_seconds=300, max_seconds=100)


# --- evaluating a window the model trimmed -------------------------------------------------

def test_a_trimmed_window_gets_the_same_checks_as_a_scanned_one():
    segs = _talk(30)
    finder = _finder(segs)
    ok, why = finder.evaluate(2, 13)
    assert ok is not None, why
    assert finder.evaluate(2, 5)[1] == "duration"
    assert finder.evaluate(5, 200)[1] == "out_of_range"


def test_a_window_across_two_blocks_is_refused():
    segs = _talk(14) + _talk(14, start=120.0, first=14)    # a pause splits them
    finder = _finder(segs)
    assert finder.evaluate(2, 20)[1] == "not_one_block"


def test_a_window_with_only_one_speaker_in_it_is_refused():
    segs = [_seg(0, A, 0, 7), _seg(1, A, 7.5, 14.5), _seg(2, A, 15, 22)]
    segs += _talk(12, start=23.0, first=3, speakers=(B, A))
    finder = _finder(segs, min_seconds=10.0)
    assert finder.evaluate(0, 2)[1] == "not_two_speakers"


# --- choosing among candidates -------------------------------------------------------------

def _cand(start, end, score):
    return Candidate(first=0, last=0, start=start, end=end, pad_start=start,
                     pad_end=end, speakers=(A, B), metrics={}, noise={},
                     music_patched_share=None, components={}, score=score,
                     tier=tier_of(score))


def test_two_good_clips_beat_one_slightly_better_one_that_overlaps_both():
    big, left, right = _cand(0, 100, 90), _cand(0, 50, 60), _cand(50, 100, 60)
    assert pick_non_overlapping([big, left, right]) == [left, right]


def test_one_much_better_clip_beats_two_poor_ones():
    big, left, right = _cand(0, 100, 95), _cand(0, 50, 30), _cand(50, 100, 30)
    assert pick_non_overlapping([left, big, right]) == [big]


def test_clips_may_touch_but_not_overlap():
    a, b, c = _cand(0, 60, 80), _cand(60, 120, 80), _cand(59, 119, 90)
    assert pick_non_overlapping([a, b]) == [a, b]
    assert len(pick_non_overlapping([a, c])) == 1


def test_nothing_to_pick_from_picks_nothing():
    assert pick_non_overlapping([]) == []


def test_the_shortlist_spreads_over_the_recording_instead_of_repeating_one_stretch():
    cluster = [_cand(s, s + 100, 90 - s) for s in (0, 2, 4, 6, 8)]
    far = _cand(500, 600, 50)
    chosen = shortlist(cluster + [far], k=2)
    assert len(chosen) == 2
    assert far in chosen and chosen[0].start == 0
    assert overlap_fraction(cluster[0], far) == 0.0


def test_the_report_says_why_a_recording_gave_few_clips():
    finder = _finder(_talk(6))
    finder.candidates()
    report = finder.report()
    assert report["segments"] == 6 and report["blocks"] == 1
    assert report["blocks_short"] == 1 and "noise_measured" not in report


# --- rescue: clean spans inside a region the validator rejected whole ----------

def test_rescue_returns_the_clean_stretches_around_a_third_speaker():
    segs = _talk(30, seg=7.0, gap=1.0)
    segs[15] = _seg(15, C, segs[15].start, segs[15].end)
    finder = _finder(segs, min_seconds=20, max_seconds=120, min_score=0)
    cand, why = finder.evaluate(0, 29)
    assert cand is None
    spans = finder.rescue(0, 29)
    assert spans and all(not _covers(c, 15) for c in spans)
    assert all(finder.evaluate(c.first, c.last)[0] is not None for c in spans)
    assert all(a.last < b.first for a, b in zip(spans, spans[1:]))


def test_rescue_finds_nothing_when_no_clean_span_is_long_enough():
    segs = _talk(6, seg=3.0, gap=0.5)
    finder = _finder(segs, min_seconds=60, max_seconds=120, min_score=0)
    assert finder.rescue(0, 5) == []


def test_rescue_respects_its_evaluation_budget():
    segs = _talk(60, seg=7.0, gap=1.0)
    finder = _finder(segs, min_seconds=20, max_seconds=60, min_score=0, rescue_max_evaluations=2)
    before = finder.stats["candidates_generated"]
    finder.rescue(0, 59)
    assert finder.stats["candidates_generated"] - before <= 2
