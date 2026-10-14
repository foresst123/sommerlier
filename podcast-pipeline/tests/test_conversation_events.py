"""Full-duplex events and the role check, on synthetic word-aligned segments.

Run:  python -m pytest tests/test_conversation_events.py -q     (from podcast-pipeline/)
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversation_events import (
    EventParams, analyze, categorize, decide_roles, event_counts, label_events, tags_of)

A, B = "SPEAKER_00", "SPEAKER_01"
P = EventParams()


def seg(i, speaker, start, text, step=0.3, flags=None, words=True):
    """A segment whose words follow each other every `step` seconds from `start`."""
    tokens = text.split()
    end = start + step * len(tokens)
    out = SimpleNamespace(index=f"{i:05d}", speaker=speaker, start=start, end=end,
                          text=text, review_flags=flags)
    if words:
        out.words = [{"word": w, "start": round(start + k * step, 3),
                      "end": round(start + (k + 1) * step, 3)} for k, w in enumerate(tokens)]
    return out


def long_text(n):
    return " ".join(f"từ{k}" for k in range(n))


def events_of(segments, marked=None, params=P):
    return analyze(segments, params, marked).events


def of_type(events, kind):
    return [e for e in events if e["type"] == kind]


# --- IPUs -------------------------------------------------------------------------

def test_a_pause_longer_than_the_ipu_gap_splits_one_speakers_speech():
    s = seg(1, A, 0.0, long_text(6))
    s.words[3]["start"] += 0.5          # 0.5 s of silence inside the turn
    s.words[3]["end"] += 0.5
    for w in s.words[4:]:
        w["start"] += 0.5
        w["end"] += 0.5
    s.end += 0.5
    units = analyze([s], P).units
    assert len(units) == 2 and units[0].end < units[1].start


def test_a_shorter_pause_or_touching_segments_are_one_ipu():
    s1 = seg(1, A, 0.0, long_text(4))
    s2 = seg(2, A, s1.end + 0.1, long_text(4))
    assert len(analyze([s1, s2], P).units) == 1


def test_a_segment_without_words_is_one_ipu_spanning_the_segment():
    s = seg(1, A, 2.0, long_text(5), words=False)
    (unit,) = analyze([s], P).units
    assert (unit.start, unit.end) == (s.start, s.end)


# --- backchannel ------------------------------------------------------------------

def talk_with(back_text="ừ", back_start=3.0, other_end=None):
    main = seg(1, A, 0.0, long_text(20))                  # speaks 0.0 - 6.0
    back = seg(2, B, back_start, back_text)
    follow = seg(3, A, 6.1, long_text(6)) if other_end is None else None
    return [main, back] + ([follow] if follow else [])


def test_a_short_listener_word_inside_the_others_speech_is_a_backchannel():
    ev = of_type(events_of(talk_with()), "backchannel")
    assert len(ev) == 1
    e = ev[0]
    assert e["by"] == B and e["during"] == A and e["text"] == "ừ"
    assert e["overlap"] == pytest.approx(0.3, abs=0.01)


def test_two_words_in_the_list_are_still_a_backchannel_three_are_not():
    assert of_type(events_of(talk_with("đúng rồi")), "backchannel")
    assert not of_type(events_of(talk_with("ừ đúng rồi")), "backchannel")


def test_a_word_outside_the_list_is_not_a_backchannel():
    assert not of_type(events_of(talk_with("tuyệt")), "backchannel")


def test_a_backchannel_longer_than_a_second_is_not_one():
    segs = talk_with("đúng rồi")
    segs[1] = seg(2, B, 3.0, "đúng rồi", step=0.7)
    assert not of_type(events_of(segs), "backchannel")


def test_the_other_speaker_must_keep_speaking_after_it():
    main = seg(1, A, 0.0, long_text(10))                  # ends at 3.0
    back = seg(2, B, 2.8, "ừ")                            # ends 3.1, A already done
    assert not of_type(events_of([main, back]), "backchannel")


def test_the_other_speaker_must_have_been_speaking_just_before():
    back = seg(1, B, 0.0, "ừ")
    main = seg(2, A, 0.4, long_text(10))
    assert not of_type(events_of([back, main]), "backchannel")


def test_a_backchannel_may_sit_in_a_short_pause_between_the_others_ipus():
    a1 = seg(1, A, 0.0, long_text(10))                    # ends 3.0
    back = seg(2, B, 3.3, "dạ")
    a2 = seg(3, A, 3.9, long_text(6))
    assert of_type(events_of([a1, back, a2]), "backchannel")


# --- interruption / competitive overlap -------------------------------------------

def cut_in(holder_words=20, cut_at=3.0, cutter_words=8, holder_resumes=None):
    holder = seg(1, A, 0.0, long_text(holder_words))
    cutter = seg(2, B, cut_at, long_text(cutter_words))
    return holder, cutter


def test_listener_cutting_in_while_the_holder_stops_is_an_interruption_with_a_yield():
    holder, cutter = cut_in(holder_words=12, cut_at=3.0)            # holder ends 3.6
    ev = of_type(events_of([holder, cutter]), "interruption")
    assert len(ev) == 1
    e = ev[0]
    assert e["by"] == B and e["of"] == A
    assert e["response"] == "yield" and e["yield_after"] == pytest.approx(0.6, abs=0.01)
    assert e["overlap"] == pytest.approx(0.6, abs=0.01)
    assert e["intent"] is None


def test_a_holder_who_goes_on_while_the_cutter_gives_up_is_a_continue():
    holder, cutter = cut_in(holder_words=30, cut_at=3.0, cutter_words=2)   # holder to 9.0
    ev = of_type(events_of([holder, cutter]), "interruption")
    assert len(ev) == 1 and ev[0]["response"] == "continue"
    assert ev[0]["yield_after"] is None


def test_both_talking_on_for_more_than_the_yield_window_is_a_competitive_overlap():
    holder, cutter = cut_in(holder_words=30, cut_at=3.0, cutter_words=12)  # both past 1.5 s
    events = events_of([holder, cutter])
    assert of_type(events, "competitive_overlap") and not of_type(events, "interruption")
    assert of_type(events, "competitive_overlap")[0]["overlap"] > P.yield_within_seconds


def test_an_overlap_shorter_than_the_minimum_is_no_event():
    holder = seg(1, A, 0.0, long_text(10))                          # ends 3.0
    cutter = seg(2, B, 2.9, long_text(8))
    events = events_of([holder, cutter])
    assert not of_type(events, "interruption") and not of_type(events, "competitive_overlap")


def test_the_llm_marked_line_gives_the_intent_and_adapt_when_the_reply_follows():
    holder, cutter = cut_in(holder_words=12, cut_at=3.0, cutter_words=8)    # cutter ends 5.4
    reply = seg(3, A, 5.8, long_text(8))
    ev = of_type(events_of([holder, cutter, reply], marked={cutter.index: "clarify"}),
                 "interruption")[0]
    assert ev["intent"] == "clarify" and ev["response"] == "adapt"


def test_without_the_mark_a_yield_stays_a_yield_even_if_the_holder_speaks_again():
    holder, cutter = cut_in(holder_words=12, cut_at=3.0, cutter_words=8)
    reply = seg(3, A, 6.2, long_text(8))
    ev = of_type(events_of([holder, cutter, reply]), "interruption")[0]
    assert ev["response"] == "yield"


def test_a_marked_line_with_no_overlap_is_not_an_interruption():
    s1, s2 = seg(1, A, 0.0, long_text(10)), seg(2, B, 3.5, long_text(8))
    assert not of_type(events_of([s1, s2], marked={s2.index: "clarify"}), "interruption")


# --- turn switch / merged backchannel ----------------------------------------------

def test_a_change_of_speaker_without_overlap_is_a_turn_switch_with_its_gap():
    s1, s2 = seg(1, A, 0.0, long_text(10)), seg(2, B, 3.5, long_text(8))
    (e,) = of_type(events_of([s1, s2]), "turn_switch")
    assert e["by"] == B and e["of"] == A and e["gap"] == pytest.approx(0.5, abs=0.01)


def test_a_backchannel_is_not_a_turn_switch():
    assert not of_type(events_of(talk_with()), "turn_switch")


def flagged(phrase):
    return [{"kind": "speaker_phrase", "phrase": phrase, "conf": 0.9, "why": "x"}]


def test_a_soft_speaker_phrase_with_word_times_is_a_merged_backchannel_by_the_other():
    s = seg(1, A, 0.0, "tôi nghĩ vậy đúng rồi và cẩn thận", flags=flagged("đúng rồi"))
    other = seg(2, B, 5.0, long_text(5))
    (e,) = of_type(events_of([s, other]), "merged_backchannel")
    assert e["by"] == B and e["during"] == A and e["text"] == "đúng rồi"
    assert (e["start"], e["end"]) == (pytest.approx(0.9), pytest.approx(1.5))


def test_a_merged_backchannel_needs_alignment_and_a_found_phrase():
    s = seg(1, A, 0.0, "tôi nghĩ vậy đúng rồi", flags=flagged("đúng rồi"), words=False)
    other = seg(2, B, 5.0, long_text(5))
    assert not of_type(events_of([s, other]), "merged_backchannel")
    s2 = seg(3, A, 0.0, "tôi nghĩ vậy", flags=flagged("không có"))
    assert not of_type(events_of([s2, other]), "merged_backchannel")


# --- category ---------------------------------------------------------------------

def an(segments, marked=None):
    return analyze(segments, P, marked)


def test_no_overlap_and_no_backchannel_is_half_duplex():
    a = an([seg(1, A, 0.0, long_text(10)), seg(2, B, 3.5, long_text(8))])
    assert categorize(a.events, a.max_overlap, P) == ("turn_by_turn", None)


def test_an_interruption_wins_over_backchannels_and_keeps_both_tags():
    holder = seg(1, A, 0.0, long_text(26))                # to 7.8: stops soon after the cut-in
    b1 = seg(2, B, 2.0, "ừ")
    b2 = seg(3, B, 5.0, "dạ")
    cutter = seg(4, B, 7.0, long_text(8))
    follow = seg(5, A, 10.0, long_text(5))
    a = an([holder, b1, b2, cutter, follow])
    counts = event_counts(a.events)
    assert counts["backchannel"] == 2 and counts["interruption"] == 1
    assert categorize(a.events, a.max_overlap, P) == ("interrupt", None)
    assert tags_of(a.events) == ["interruption", "backchannel"]


def test_two_backchannels_without_interruption_is_the_backchannel_category():
    main = seg(1, A, 0.0, long_text(30))
    segs = [main, seg(2, B, 2.0, "ừ"), seg(3, B, 5.0, "dạ"), seg(4, A, 9.1, long_text(5))]
    a = an(segs)
    assert categorize(a.events, a.max_overlap, P) == ("backchannel", None)


def test_one_backchannel_alone_is_the_backchannel_category():
    a = an(talk_with())
    assert categorize(a.events, a.max_overlap, P) == ("backchannel", None)


def test_a_competitive_overlap_is_a_backchannel_not_turn_by_turn():
    holder, cutter = cut_in(holder_words=30, cut_at=3.0, cutter_words=12)
    a = an([holder, cutter])
    assert categorize(a.events, a.max_overlap, P) == ("backchannel", None)


def test_a_cut_in_the_holder_talks_through_is_a_backchannel_not_an_interrupt():
    holder, cutter = cut_in(holder_words=30, cut_at=3.0, cutter_words=3)
    a = an([holder, cutter])
    (e,) = of_type(a.events, "interruption")
    assert e["response"] == "continue"
    assert categorize(a.events, a.max_overlap, P) == ("backchannel", None)


def test_a_question_like_reply_is_not_a_backchannel_on_its_text_alone():
    holder = seg(1, A, 0.0, long_text(30))
    reply = seg(2, B, 4.0, "thế à")
    a = an([holder, reply, seg(3, A, 9.1, long_text(5))])
    assert not of_type(a.events, "backchannel")


def test_any_overlap_that_is_not_an_interrupt_is_never_turn_by_turn():
    holder, cutter = cut_in(holder_words=30, cut_at=3.0, cutter_words=3)
    a = an([holder, cutter])
    strict = EventParams(min_backchannels=2)
    assert categorize(a.events, a.max_overlap, strict) == ("backchannel", None)


def test_gate_spans_cover_overlaps_and_overlapping_backchannels():
    holder, cutter = cut_in(holder_words=12, cut_at=3.0)
    a = an([holder, cutter])
    assert len(a.gate_spans) == 1 and a.gate_spans[0][0] == pytest.approx(3.0)
    assert an([seg(1, A, 0.0, long_text(4)), seg(2, B, 3.0, long_text(4))]).gate_spans == []


# --- roles -----------------------------------------------------------------------

def ev(kind, by, of, response=None, intent=None, **kw):
    return {"type": kind, "by": by, "of": of, "response": response, "intent": intent,
            "start": 0.0, "end": 1.0, "overlap": 0.5, **kw}


def test_a_confident_assistant_without_violations_is_locked():
    r = decide_roles([ev("interruption", B, A, "yield")], (A, B), A, 0.9, P)
    assert (r.assistant, r.user, r.role, r.swappable) == (A, B, "locked", False)
    assert r.role_conf == pytest.approx(0.9) and r.violations == []


def test_low_confidence_is_symmetric_and_swappable():
    r = decide_roles([], (A, B), A, 0.6, P)
    assert r.assistant == A and r.role == "symmetric" and r.swappable is True


def test_an_assistant_that_holds_on_when_interrupted_while_the_user_yields_is_a_violation():
    events = [ev("interruption", B, A, "continue"),            # user cuts the assistant: assistant goes on
              ev("interruption", A, B, "yield")]               # assistant cuts the user: user yields
    r = decide_roles(events, (A, B), A, 0.9, P)
    assert len(r.violations) == 1
    assert r.role_conf == pytest.approx(0.7) and r.role == "symmetric"


def test_an_assistant_that_yields_is_not_a_violation():
    events = [ev("interruption", B, A, "yield"), ev("interruption", A, B, "yield")]
    r = decide_roles(events, (A, B), A, 0.9, P)
    assert r.violations == [] and r.role == "locked"


def test_an_assistant_starting_two_topic_shifts_is_a_violation_each_violation_costs_point_two():
    events = [ev("interruption", A, B, "yield", "topic_shift"),
              ev("interruption", A, B, "yield", "topic_shift")]
    r = decide_roles(events, (A, B), A, 0.95, P)
    assert len(r.violations) == 1 and r.role_conf == pytest.approx(0.75) and r.role == "symmetric"


def test_two_violations_cost_point_four():
    events = [ev("interruption", B, A, "continue"), ev("interruption", A, B, "yield", "topic_shift"),
              ev("interruption", A, B, "yield", "topic_shift")]
    r = decide_roles(events, (A, B), A, 1.0, P)
    assert len(r.violations) == 2 and r.role_conf == pytest.approx(0.6)


def test_the_role_conf_threshold_is_configurable():
    r = decide_roles([], (A, B), A, 0.6, EventParams(role_min_conf=0.5))
    assert r.role == "locked"


def test_a_missing_or_foreign_assistant_is_chosen_by_behaviour_and_symmetric():
    events = [ev("interruption", B, A, "yield"), ev("interruption", B, A, "yield"),
              ev("interruption", A, B, "continue")]
    for given in (None, "SPEAKER_09"):
        r = decide_roles(events, (A, B), given, 0.95, P)
        assert r.assistant == A and r.user == B                 # A is interrupted and yields more
        assert r.role == "symmetric" and r.swappable is True


def test_without_interruptions_the_speaker_answering_questions_is_the_assistant():
    segs = [seg(1, A, 0.0, "bạn chạy bộ buổi nào ?"), seg(2, B, 3.0, long_text(6)),
            seg(3, A, 6.0, "còn đau gối thì sao ?"), seg(4, B, 9.0, long_text(6))]
    a = an(segs)
    r = decide_roles(a.events, (A, B), None, None, P, units=a.units)
    assert r.assistant == B and r.role == "symmetric" and r.role_conf is None


def test_a_missing_assistant_with_no_evidence_still_gets_a_deterministic_answer():
    r = decide_roles([], (A, B), None, None, P)
    assert {r.assistant, r.user} == {A, B} and r.role == "symmetric"


# --- labelling -----------------------------------------------------------------

def test_events_are_labelled_with_roles_and_shifted_to_the_excerpt_clock():
    holder, cutter = cut_in(holder_words=12, cut_at=13.0)
    holder.words = [dict(w, start=w["start"] + 10, end=w["end"] + 10) for w in holder.words]
    holder.start += 10
    holder.end += 10
    out = label_events(analyze([holder, cutter], P).events, {A: "assistant", B: "user"}, 10.0)
    (e,) = of_type(out, "interruption")
    assert e["by"] == "user" and e["of"] == "assistant"
    assert e["start"] == pytest.approx(3.0)


# --- review fixes: pauses, episodes, in-turn backchannels, clamped words -------------

def timed(i, speaker, pieces):
    """A segment with explicit (start, end, word) pieces."""
    s = SimpleNamespace(index=f"{i:05d}", speaker=speaker, start=pieces[0][0], end=pieces[-1][1],
                        text=" ".join(p[2] for p in pieces), review_flags=None)
    s.words = [{"word": w, "start": a, "end": b} for a, b, w in pieces]
    return s


def run(n, start, end):
    return [(start + k * (end - start) / n, start + (k + 1) * (end - start) / n, f"w{k}")
            for k in range(n)]


def test_a_holder_who_pauses_and_talks_on_does_not_yield_and_no_reversed_event_appears():
    a1 = timed(1, A, run(10, 0.0, 3.0))
    a2 = timed(2, A, run(10, 3.25, 6.0))            # 0.25 s pause, then A goes on
    b = timed(3, B, run(10, 2.0, 5.0))
    events = events_of([a1, a2, b])
    assert not of_type(events, "interruption")
    (e,) = of_type(events, "competitive_overlap")
    assert e["by"] == B and e["of"] == A
    assert not [x for x in events if x["type"] in ("interruption", "competitive_overlap")
                and x["by"] == A]


def test_a_holder_who_stops_for_good_is_a_yield_even_with_a_later_turn():
    a1 = timed(1, A, run(10, 0.0, 3.0))
    b = timed(2, B, run(8, 2.0, 4.4))
    a2 = timed(3, A, run(6, 6.0, 7.8))               # speaks again only after the cutter is done
    (e,) = of_type(events_of([a1, b, a2]), "interruption")
    assert e["response"] == "yield" and e["yield_after"] == pytest.approx(1.0)


def test_a_cut_in_with_a_pause_is_one_event_with_the_whole_overlap():
    holder = timed(1, A, run(40, 0.0, 12.0))
    b1 = timed(2, B, run(3, 4.0, 4.9))
    b2 = timed(3, B, run(20, 5.5, 11.5))             # 0.6 s pause, then B goes on
    events = events_of([holder, b1, b2])
    cut = [e for e in events if e["type"] in ("interruption", "competitive_overlap")]
    assert len(cut) == 1 and cut[0]["by"] == B and cut[0]["of"] == A
    assert cut[0]["overlap"] == pytest.approx(0.9 + 6.0, abs=0.01)
    assert (cut[0]["start"], cut[0]["end"]) == (pytest.approx(4.0), pytest.approx(11.5))


def test_a_listener_pause_longer_than_the_merge_window_starts_a_new_episode():
    holder = timed(1, A, run(60, 0.0, 18.0))
    b1 = timed(2, B, run(4, 2.0, 3.2))
    b2 = timed(3, B, run(4, 8.0, 9.2))
    cut = [e for e in events_of([holder, b1, b2]) if e["type"] == "interruption"]
    assert len(cut) == 2


def test_a_backchannel_with_no_overlap_far_from_a_turn_does_not_count_for_the_category():
    a1 = seg(1, A, 0.0, long_text(10))                   # 0-3.0
    bc1 = seg(2, B, 3.5, "ừ")                            # in a 1 s gap: not in the turn
    a2 = seg(3, A, 4.5, long_text(10))
    bc2 = seg(4, B, 8.0, "dạ")
    a3 = seg(5, A, 9.0, long_text(5))
    a = an([a1, bc1, a2, bc2, a3])
    assert event_counts(a.events)["backchannel"] == 2
    assert not any(e["in_turn"] for e in a.events if e["type"] == "backchannel")
    assert categorize(a.events, a.max_overlap, P) == ("backchannel", None)
    assert a.gate_spans == []


def test_backchannels_that_overlap_count_and_enter_the_gate_spans():
    main = seg(1, A, 0.0, long_text(30))
    a = an([main, seg(2, B, 2.0, "ừ"), seg(3, B, 5.0, "dạ"), seg(4, A, 9.1, long_text(5))])
    assert all(e["in_turn"] for e in a.events if e["type"] == "backchannel")
    assert len(a.gate_spans) == 2


def test_words_outside_the_segment_are_ignored_or_clamped_when_building_ipus():
    s = seg(1, A, 2.0, long_text(5))                     # 2.0 - 3.5
    s.words.append({"word": "xa", "start": 40.0, "end": 40.3})
    s.words.append({"word": "sớm", "start": 0.0, "end": 0.5})
    s.words[0]["start"] = 1.5                            # starts before the segment
    (unit,) = analyze([s], P).units
    assert (unit.start, unit.end) == (pytest.approx(2.0), pytest.approx(3.5))


# --- what counts as the other speaker really stopping -----------------------------------

def test_an_ordinary_hand_over_with_a_sliver_of_overlap_is_not_an_interrupt():
    holder = seg(1, A, 0.0, long_text(32))                  # to 9.6
    nxt = seg(2, B, 9.3, long_text(14))                     # B starts 0.3 s before A ends
    a = an([holder, nxt])
    assert categorize(a.events, a.max_overlap, P)[0] == "backchannel"
    assert not [e for e in a.events if e["type"] == "interruption" and e["response"] != "continue"]


def test_a_short_comment_is_not_an_interrupt_even_if_the_holder_pauses():
    holder = seg(1, A, 0.0, long_text(12))                  # to 3.6
    comment = seg(2, B, 3.0, long_text(4))                  # 1.2 s, under cutter_min_seconds
    a = an([holder, comment])
    assert categorize(a.events, a.max_overlap, P)[0] == "backchannel"


def test_the_holder_taking_the_floor_straight_back_is_not_a_yield():
    holder = seg(1, A, 0.0, long_text(12))                  # to 3.6
    cutter = seg(2, B, 3.0, long_text(8))                   # to 5.4
    back = seg(3, A, 5.6, long_text(8))                     # 0.2 s after the cutter ends
    a = an([holder, cutter, back])
    assert categorize(a.events, a.max_overlap, P)[0] == "backchannel"


def test_a_long_cut_in_that_stops_the_holder_is_an_interrupt():
    holder, cutter = cut_in(holder_words=12, cut_at=3.0, cutter_words=8)   # overlap 0.6, cutter 2.4 s
    a = an([holder, cutter])
    assert categorize(a.events, a.max_overlap, P)[0] == "interrupt"


def test_the_models_mark_makes_a_short_cut_in_count():
    holder = seg(1, A, 0.0, long_text(12))
    comment = seg(2, B, 3.0, long_text(4))
    a = an([holder, comment], marked={comment.index: "clarify"})
    assert categorize(a.events, a.max_overlap, P)[0] == "interrupt"
