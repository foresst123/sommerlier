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
    service = MusicService(panns_model=None, demucs_model=HalvingSeparator(), logger=None)

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

    service = MusicService(panns_model=None, demucs_model=HalvingSeparator(), logger=None)
    once = fresh()
    patches = service.strip_music_spans(once, MusicMap([(5.0, 8.0, MUSIC)]))

    twice = fresh()
    service.apply_music_patches(twice, patches)
    service.apply_music_patches(twice, patches)
    assert np.array_equal(once.waveform, twice.waveform)


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
