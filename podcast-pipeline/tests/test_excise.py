"""Cutting sung passages out, and finding the way back to the original clock.

Marking a sung stretch and letting later stages skip it does not work: the
diarizer still clusters on it and ASR still receives it, so lyrics reach the
transcript scored as dialogue. Cutting is the only thing that keeps them out --
and it moves every timestamp after the cut, which is what TimelineMap exists to
undo.

Run:  python -m pytest tests/test_excise.py -q     (from podcast-pipeline/)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.excise import TimelineMap, excise

SR = 1000


def _ramp(seconds=10):
    """One distinct value per second, so a cut is visible in the output."""
    return np.repeat(np.arange(seconds, dtype=np.float32), SR)


# --- what leaves and what stays ---------------------------------------------

def test_a_cut_stretch_is_gone_and_the_rest_survives():
    out, _ = excise(_ramp(), SR, [(3.0, 5.0)], fade=0.0)
    present = set(np.unique(out).tolist())
    assert 3.0 not in present and 4.0 not in present
    assert {0.0, 2.0, 5.0, 9.0} <= present


def test_the_recording_shortens_by_exactly_what_was_cut():
    out, timeline = excise(_ramp(), SR, [(3.0, 5.0)], fade=0.0)
    assert abs(len(out) / SR - 8.0) < 0.01
    assert abs(timeline.removed - 2.0) < 0.01


def test_overlapping_cuts_are_merged_not_double_counted():
    out, _ = excise(_ramp(), SR, [(3.0, 5.0), (4.0, 6.0)], fade=0.0)
    assert abs(len(out) / SR - 7.0) < 0.01


def test_nothing_to_cut_returns_the_audio_unchanged():
    audio = _ramp()
    out, timeline = excise(audio, SR, [], fade=0.0)
    assert np.array_equal(out, audio)
    assert timeline.removed == 0.0


def test_an_island_too_short_to_use_goes_with_its_neighbours():
    """20ms between two cuts is not speech anyone can use, and keeping it would
    put two crossfades back to back over almost nothing."""
    out, _ = excise(_ramp(), SR, [(3.0, 4.0), (4.02, 6.0)], fade=0.0)
    assert 4.0 not in set(np.unique(out).tolist())


def test_empty_audio_does_not_raise():
    out, timeline = excise(np.zeros(0, dtype=np.float32), SR, [(1.0, 2.0)])
    assert len(out) == 0
    assert not timeline


# --- the joins ---------------------------------------------------------------

def test_the_seam_does_not_dip_to_silence():
    """Fading each side to zero and butting the results together dips at the
    join -- the artifact the fade exists to remove. The sides have to overlap."""
    audio = np.ones(10 * SR, dtype=np.float32)
    out, _ = excise(audio, SR, [(3.0, 5.0)], fade=0.03)
    assert out.min() > 0.95


def test_uncorrelated_speech_keeps_its_level_across_the_seam():
    """Equal-power ramps: two unrelated stretches sum in power, not amplitude."""
    sr = 16000
    audio = (np.random.default_rng(0).standard_normal(10 * sr) * 0.1).astype(np.float32)
    out, _ = excise(audio, sr, [(3.0, 5.0)], fade=0.03)

    fade_n = int(0.03 * sr)
    seam = int(3.0 * sr) - fade_n
    at_seam = np.sqrt((out[seam:seam + fade_n] ** 2).mean())
    elsewhere = np.sqrt((out[:seam - 1000] ** 2).mean())
    assert abs(20 * np.log10(at_seam / elsewhere)) < 1.0


def test_the_crossfade_costs_the_overlap_length():
    audio = np.ones(10 * SR, dtype=np.float32)
    plain, _ = excise(audio, SR, [(3.0, 5.0)], fade=0.0)
    faded, _ = excise(audio, SR, [(3.0, 5.0)], fade=0.03)
    assert len(plain) - len(faded) == int(0.03 * SR)


# --- finding the way back ----------------------------------------------------

def test_an_instant_before_the_cut_is_unmoved():
    _, timeline = excise(_ramp(), SR, [(3.0, 5.0)], fade=0.0)
    assert abs(timeline.to_original(2.5) - 2.5) < 0.01


def test_an_instant_after_the_cut_maps_back_over_it():
    _, timeline = excise(_ramp(), SR, [(3.0, 5.0)], fade=0.0)
    assert abs(timeline.to_original(3.5) - 5.5) < 0.01


def test_a_removed_instant_has_no_place_in_the_cut_timeline():
    """None, not a nearby guess: a caller must be able to tell that the audio
    for this moment is gone."""
    _, timeline = excise(_ramp(), SR, [(3.0, 5.0)], fade=0.0)
    assert timeline.to_cut(4.0) is None
    assert abs(timeline.to_cut(6.0) - 4.0) < 0.01


def test_the_two_directions_agree():
    _, timeline = excise(_ramp(20), SR, [(3.0, 5.0), (12.0, 14.0)], fade=0.0)
    for original in (0.5, 2.9, 5.1, 11.5, 14.5, 19.0):
        cut = timeline.to_cut(original)
        assert cut is not None
        assert abs(timeline.to_original(cut) - original) < 0.01


def test_an_empty_map_is_the_identity():
    empty = TimelineMap()
    assert empty.to_original(7.0) == 7.0
    assert empty.to_cut(7.0) == 7.0


def test_the_map_round_trips_through_a_checkpoint():
    _, timeline = excise(_ramp(), SR, [(3.0, 5.0)], fade=0.0)
    revived = TimelineMap.from_json(timeline.to_json())
    assert abs(revived.to_original(3.5) - 5.5) < 0.01


# --- the map that has to move with the audio --------------------------------

def test_the_music_map_follows_the_cut():
    """Spans stay in original time while everything downstream moves; comparing
    the two would silently mis-locate every bed."""
    from utils.music_map import MUSIC, SINGING, MusicMap

    _, timeline = excise(np.zeros(20 * SR, dtype=np.float32), SR,
                         [(2.0, 4.0)], fade=0.0)
    moved = MusicMap([(2.0, 4.0, SINGING), (10.0, 12.0, MUSIC)]).remap(timeline)

    kinds = [k for _, _, k in moved.spans]
    assert SINGING not in kinds, "a cut stretch has no place in the new timeline"
    assert abs(moved.spans[0][0] - 8.0) < 0.01


# --- the flow as a whole -----------------------------------------------------

def test_stripping_then_cutting_leaves_each_where_it_belongs():
    """The two passes run in this order because both address spans in original
    time: cutting first would move every bed before the separator saw it."""
    from schemas.audio import AudioData
    from services.music_service import MusicService
    from utils.music_map import MUSIC, SINGING, MusicMap

    class HalvingSeparator:
        def separate_segment(self, audio, sr):
            return (np.asarray(audio) * 0.5).astype(np.float32)

    music_map = MusicMap([(2.0, 4.0, MUSIC), (10.0, 12.0, SINGING)])
    audio = AudioData(name="t", waveform=np.ones(20 * SR, dtype=np.float32),
                      sample_rate=SR, duration=20.0, audio_segment=None)
    service = MusicService(panns_model=None, bs_roformer_model=HalvingSeparator(), logger=None)

    service.strip_music_spans(audio, music_map)
    assert audio.waveform[3 * SR] == 0.5, "the bed should have been stripped"
    assert audio.waveform[11 * SR] == 1.0, "singing is cut, not stripped"

    trimmed, timeline = excise(audio.waveform, SR,
                               [(a, b) for a, b, _ in music_map.excised_spans()],
                               fade=0.0)
    assert abs(len(trimmed) / SR - 18.0) < 0.01
    assert trimmed[3 * SR] == 0.5, "the stripped stretch must survive the cut"

    moved = music_map.remap(timeline)
    assert SINGING not in [k for _, _, k in moved.spans]


def test_applying_cached_patches_twice_changes_nothing():
    """run() is re-entered per stage and reloads the audio, so the patches are
    re-applied every time; doing so must not separate vocals out of vocals."""
    from schemas.audio import AudioData
    from services.music_service import MusicService
    from utils.music_map import MUSIC, MusicMap

    class HalvingSeparator:
        def separate_segment(self, audio, sr):
            return (np.asarray(audio) * 0.5).astype(np.float32)

    def fresh():
        return AudioData(name="t", waveform=np.ones(20 * SR, dtype=np.float32),
                         sample_rate=SR, duration=20.0, audio_segment=None)

    service = MusicService(panns_model=None, bs_roformer_model=HalvingSeparator(), logger=None)
    once = fresh()
    patches = service.strip_music_spans(once, MusicMap([(5.0, 8.0, MUSIC)]))

    twice = fresh()
    service.apply_music_patches(twice, patches)
    service.apply_music_patches(twice, patches)
    assert np.array_equal(once.waveform, twice.waveform)


def test_stripping_music_accepts_read_only_cached_waveform():
    """Cached audio may be backed by a read-only mmap; stripping still edits it."""
    from schemas.audio import AudioData
    from services.music_service import MusicService
    from utils.music_map import MUSIC, MusicMap

    class HalvingSeparator:
        def separate_segment(self, audio, sr):
            return (np.asarray(audio) * 0.5).astype(np.float32)

    waveform = np.ones(20 * SR, dtype=np.float32)
    waveform.setflags(write=False)
    audio = AudioData(name="t", waveform=waveform, sample_rate=SR,
                      duration=20.0, audio_segment=None)

    patches = MusicService(
        panns_model=None, bs_roformer_model=HalvingSeparator(), logger=None
    ).strip_music_spans(audio, MusicMap([(5.0, 8.0, MUSIC)]))

    assert patches
    assert audio.waveform.flags.writeable
    assert audio.waveform[6 * SR] == 0.5


def test_applying_cached_patches_accepts_read_only_cached_waveform():
    """Re-entered stage runs can reload the waveform from a read-only cache."""
    from schemas.audio import AudioData
    from services.music_service import MusicService

    waveform = np.ones(20 * SR, dtype=np.float32)
    waveform.setflags(write=False)
    audio = AudioData(name="t", waveform=waveform, sample_rate=SR,
                      duration=20.0, audio_segment=None)

    MusicService.apply_music_patches(
        audio, [(5 * SR, np.full(3 * SR, 0.25, dtype=np.float32))]
    )

    assert audio.waveform.flags.writeable
    assert audio.waveform[6 * SR] == 0.25


def test_a_span_past_the_end_is_clamped():
    out, _ = excise(np.zeros(5 * SR, dtype=np.float32), SR, [(3.0, 999.0)], fade=0.0)
    assert abs(len(out) / SR - 3.0) < 0.01


def test_the_service_refuses_to_cut_away_the_recording():
    """A tagger that calls most of a podcast singing has gone wrong; an empty
    waveform downstream is the worst possible answer to that."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "services", "pipeline_service.py"),
                  encoding="utf-8").read()
    assert "CUT_SHARE_LIMIT" in source
    assert "keeping the audio " in source


def test_the_per_segment_pass_is_skipped_when_the_waveform_was_cleaned():
    """Otherwise a vocal separator is handed its own output and asked to find a
    voice in it a second time -- and reloaded to do it."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "services", "pipeline_service.py"),
                  encoding="utf-8").read()
    assert '_has_map = self.step_enabled(args, "music_analysis")' in source
    assert "the waveform was already cleaned" in source


def test_a_source_file_routes_stripping_through_the_hi_res_path():
    """MusicService must prefer separate_span when it knows the original file:
    that is the whole 44.1kHz gain, and it was invisible from the waveform."""
    from schemas.audio import AudioData
    from services.music_service import MusicService
    from utils.music_map import MUSIC, MusicMap

    calls = []

    class Separator:
        def separate_span(self, path, start, end, sr, reference):
            calls.append((path, start, end, sr, len(reference)))
            return np.full(len(reference), 0.25, dtype=np.float32)

        def separate_segment(self, audio, sr):
            raise AssertionError("the 16kHz fallback must not run here")

    audio = AudioData(name="t", waveform=np.ones(20 * SR, dtype=np.float32),
                      sample_rate=SR, duration=20.0, audio_segment=None)
    service = MusicService(panns_model=None, bs_roformer_model=Separator(), logger=None)
    service.strip_music_spans(audio, MusicMap([(5.0, 8.0, MUSIC)]),
                              source_path="/tmp/original.mp3")

    assert calls == [("/tmp/original.mp3", 5.0, 8.0, SR, 3 * SR)]
    assert audio.waveform[6 * SR] == 0.25


def test_a_separator_without_the_hi_res_path_still_works():
    """The 16kHz path is the contract; hi-res is an optimisation on top, and a
    stub or an older separator that lacks it must not break the stage."""
    from schemas.audio import AudioData
    from services.music_service import MusicService
    from utils.music_map import MUSIC, MusicMap

    class OldSeparator:
        def separate_segment(self, audio, sr):
            return (np.asarray(audio) * 0.5).astype(np.float32)

    audio = AudioData(name="t", waveform=np.ones(20 * SR, dtype=np.float32),
                      sample_rate=SR, duration=20.0, audio_segment=None)
    MusicService(panns_model=None, bs_roformer_model=OldSeparator(), logger=None
                 ).strip_music_spans(audio, MusicMap([(5.0, 8.0, MUSIC)]),
                                     source_path="/tmp/original.mp3")
    assert audio.waveform[6 * SR] == 0.5


def test_a_failed_hi_res_decode_falls_back_to_the_16khz_slice():
    """A source that cannot be decoded at the span offset must not lose the
    strip: the pipeline already holds a usable slice."""
    from schemas.audio import AudioData
    from services.music_service import MusicService
    from utils.music_map import MUSIC, MusicMap

    class Separator:
        def separate_span(self, path, start, end, sr, reference):
            return None                      # decode failed, caller falls back
        def separate_segment(self, audio, sr):
            return (np.asarray(audio) * 0.5).astype(np.float32)

    audio = AudioData(name="t", waveform=np.ones(20 * SR, dtype=np.float32),
                      sample_rate=SR, duration=20.0, audio_segment=None)
    MusicService(panns_model=None, bs_roformer_model=Separator(), logger=None
                 ).strip_music_spans(audio, MusicMap([(5.0, 8.0, MUSIC)]),
                                     source_path="/tmp/original.mp3")
    assert audio.waveform[6 * SR] == 0.5


# --- provenance: where a cut-timeline interval came from ----------------------

def _timeline():
    """20s of audio with 5.0-8.0 and 12.0-13.0 removed.

    kept (original)      -> cut
      0.0 -  5.0         ->  0.0 -  5.0
      8.0 - 12.0         ->  5.0 -  9.0
     13.0 - 20.0         ->  9.0 - 16.0
    """
    from utils.excise import TimelineMap
    return TimelineMap([(0.0, 5.0, 0.0), (8.0, 12.0, 5.0), (13.0, 20.0, 9.0)])


def test_an_interval_inside_one_piece_is_one_span():
    spans = _timeline().spans_to_original(6.0, 7.0)
    assert len(spans) == 1
    assert spans[0] == pytest.approx((9.0, 10.0))


def test_an_interval_across_a_join_is_both_pieces_not_one_range():
    """[to_original(start), to_original(end)] would claim 4.0-9.0, which
    includes the 5.0-8.0 that was removed -- audio this segment never held."""
    spans = _timeline().spans_to_original(4.0, 6.0)
    assert len(spans) == 2
    assert spans[0] == pytest.approx((4.0, 5.0))
    assert spans[1] == pytest.approx((8.0, 9.0))
    covered = sum(b - a for a, b in spans)
    assert covered == pytest.approx(2.0), "the pieces must add up to the segment"


def test_an_interval_across_two_joins_is_three_pieces():
    spans = _timeline().spans_to_original(4.5, 9.5)
    assert len(spans) == 3
    assert [round(a, 2) for a, _ in spans] == [4.5, 8.0, 13.0]


def test_crosses_cut_says_whether_the_audio_is_glued():
    tl = _timeline()
    assert tl.crosses_cut(4.0, 6.0) is True
    assert tl.crosses_cut(6.0, 7.0) is False


def test_an_empty_or_reversed_interval_has_no_provenance():
    tl = _timeline()
    assert tl.spans_to_original(3.0, 3.0) == []
    assert tl.spans_to_original(5.0, 4.0) == []


def test_an_uncut_recording_maps_every_interval_to_itself():
    """No excision means the two timelines are the same one, and provenance
    must not become a special case for the caller."""
    from utils.excise import TimelineMap
    empty = TimelineMap()
    assert empty.spans_to_original(3.0, 7.0) == [(3.0, 7.0)]
    assert empty.crosses_cut(3.0, 7.0) is False
    assert empty.cut_between(3.0, 7.0) is False


def test_a_pause_spanning_a_join_is_not_a_pause():
    """The gap between two turns is the thing a full-duplex corpus learns. A
    gap measured across a join is however much was removed there, not silence."""
    tl = _timeline()
    assert tl.cut_between(4.9, 5.1) is True     # the 5.0 join sits inside
    assert tl.cut_between(6.0, 7.0) is False    # wholly inside one piece
    assert tl.cut_between(8.9, 9.1) is True     # the 9.0 join


def test_a_turn_ending_exactly_on_a_join_still_counts():
    """Touching endpoints count: what follows is removed audio whatever the
    arithmetic says."""
    tl = _timeline()
    assert tl.cut_between(5.0, 5.0) is True
    assert tl.cut_between(3.0, 5.0) is True


def test_the_end_of_the_recording_is_not_a_join():
    """kept[-1] ends the audio; there is nothing removed after it to mistake
    for a gap."""
    tl = _timeline()
    assert tl.cut_between(15.0, 16.0) is False


def test_provenance_survives_a_json_round_trip():
    from utils.excise import TimelineMap
    revived = TimelineMap.from_json(_timeline().to_json())
    assert revived.crosses_cut(4.0, 6.0) is True
    assert revived.cut_between(4.9, 5.1) is True


# --- replaying a checkpointed cut --------------------------------------------
#
# run() is re-entered once per stage and reloads the whole recording each time,
# so the cut has to be reproduced on every entry. Reproducing it from the music
# map is not the same thing: the map is rebuilt from thresholds that may have
# moved and a cut_singing flag that may have been switched off since. When that
# rebuild comes back empty, the audio stays whole while the timeline still says
# it was shortened -- and diarization then runs on a different timeline from
# every timestamp that follows it, with nothing to notice.

def test_the_removed_spans_are_the_complement_of_the_kept_ones():
    tl = _timeline()      # 0-5, 8-12, 13-20 kept out of 20s
    assert tl.removed_spans(20.0) == [(5.0, 8.0), (12.0, 13.0)]


def test_a_recording_that_opens_on_a_cut_reports_the_lead_in():
    from utils.excise import TimelineMap
    tl = TimelineMap([(4.0, 10.0, 0.0)])
    assert tl.removed_spans(10.0) == [(0.0, 4.0)]


def test_a_recording_that_ends_inside_a_cut_needs_its_length():
    """Without the original duration the tail cannot be told from the end."""
    from utils.excise import TimelineMap
    tl = TimelineMap([(0.0, 6.0, 0.0)])
    assert tl.removed_spans() == []
    assert tl.removed_spans(10.0) == [(6.0, 10.0)]


def test_an_uncut_timeline_removed_nothing():
    from utils.excise import TimelineMap
    assert TimelineMap().removed_spans(20.0) == []
    assert TimelineMap([(0.0, 20.0, 0.0)]).removed_spans(20.0) == []


def test_replaying_a_cut_reproduces_it_sample_for_sample():
    """This is the property the re-entry path depends on: cutting, then
    cutting again from the timeline alone, must give the same audio."""
    sr = 1000
    waveform = np.arange(20 * sr, dtype=np.float32) / sr
    once, timeline = excise(waveform, sr, [(5.0, 8.0), (12.0, 13.0)])

    again, _ = excise(waveform, sr, timeline.removed_spans(20.0))
    assert len(again) == len(once)
    assert np.allclose(again, once)


def test_replaying_survives_a_json_round_trip():
    """The timeline reaches the next entry through a checkpoint, not memory."""
    from utils.excise import TimelineMap
    sr = 1000
    waveform = np.arange(20 * sr, dtype=np.float32) / sr
    once, timeline = excise(waveform, sr, [(2.0, 4.0), (15.0, 19.0)])
    revived = TimelineMap.from_json(timeline.to_json())
    again, _ = excise(waveform, sr, revived.removed_spans(20.0))
    assert np.allclose(again, once)


def test_replaying_a_dropped_island_keeps_it_dropped():
    """excise drops a kept stretch under min_keep together with the span beside
    it. The timeline records that, so the replay must not resurrect it."""
    sr = 1000
    waveform = np.arange(20 * sr, dtype=np.float32) / sr
    once, timeline = excise(waveform, sr, [(5.0, 8.0), (8.02, 11.0)])
    again, _ = excise(waveform, sr, timeline.removed_spans(20.0))
    assert np.allclose(again, once)


def test_the_pipeline_replays_from_the_timeline_not_from_the_map():
    """The bug this guards: `cuts` is recomputed per entry, the timeline is
    not. Reaching for `cuts` here is what puts the two out of step."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "services", "pipeline_service.py"),
                  encoding="utf-8").read()
    start = source.index("elif timeline:")
    block = source[start:source.index("self.timeline = timeline", start)]
    assert "replay = timeline.removed_spans(" in block
    # What matters is what reaches excise, not whether `cuts` is mentioned --
    # the warning beside it legitimately reads the map to compare the two.
    assert "excise(audio_data.waveform, audio_data.sample_rate, replay)" in block, (
        "the replay handed to excise must come from the timeline")


def test_overlapping_cuts_do_not_look_like_a_settings_change():
    """The music map pads `music`, `song` and `singing` separately, so two
    spans that were adjacent overlap after padding and a raw sum counts the
    overlap twice. Comparing that sum against the timeline made the re-entry
    path warn "music settings have changed" on every ordinary run -- naming a
    difference that was only the double count."""
    from utils.excise import _merge
    sr = 1000
    waveform = np.arange(20 * sr, dtype=np.float32) / sr
    cuts = [(5.0, 8.3), (8.0, 11.0)]          # overlap 0.3s, as padding produces

    _, timeline = excise(waveform, sr, cuts)
    raw    = sum(b - a for a, b in cuts)
    merged = sum(b - a for a, b in _merge(cuts))
    have   = sum(b - a for a, b in timeline.removed_spans(20.0))

    assert raw - merged == pytest.approx(0.3), "the raw sum double counts"
    assert merged == pytest.approx(have), "merged, the two agree exactly"


def test_the_reentry_warning_compares_merged_spans():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(root, "services", "pipeline_service.py"),
                  encoding="utf-8").read()
    start = source.index("elif timeline:")
    block = source[start:source.index("self.timeline = timeline", start)]
    assert "_merge([(a, b) for a, b, _ in cuts])" in block
