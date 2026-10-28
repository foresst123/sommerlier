import numpy as np
import pytest

from utils.ambience_bed import (
    AmbienceParams,
    _has_hum,
    build_bed,
    compose_user_channel,
    find_quiet_gaps,
    noise_floor_track,
    postprocess_assistant,
    quality_report,
    safe_mask,
)

SR = 16000


def _noise(seconds, sigma, seed=0):
    return np.random.default_rng(seed).normal(0.0, sigma, int(round(seconds * SR)))


def _tone(t0, t1, freq, amp, total_s):
    """A sine voice on [t0, t1) of a total_s-long clock."""
    out = np.zeros(int(round(total_s * SR)))
    i0, i1 = int(round(t0 * SR)), int(round(t1 * SR))
    t = np.arange(i1 - i0) / SR
    out[i0:i1] = amp * np.sin(2 * np.pi * freq * t)
    return out


def _recording(total_s, speech, sigma=1e-3, voice_amp=0.2, seed=0):
    rec = _noise(total_s, sigma, seed)
    for a, b in speech:
        rec += _tone(a, b, 220.0, voice_amp, total_s)
    return rec


# -- find_quiet_gaps ---------------------------------------------------------

def test_find_quiet_gaps_margins_and_min_length():
    p = AmbienceParams()  # after 0.25, before 0.10, min 0.40
    gaps = find_quiet_gaps([(1.0, 2.0), (2.5, 3.0)], 0.0, 5.0, p)
    # (0 .. 0.9) before the first word; the 0.15 s between the two is too short; (3.25 .. 5.0) after the second
    assert len(gaps) == 2
    assert gaps[0] == pytest.approx((0.0, 0.9))
    assert gaps[1] == pytest.approx((3.25, 5.0))


def test_find_quiet_gaps_unions_overlapping_speakers_and_clips_window():
    p = AmbienceParams()
    # two speakers overlapping: one union (1.0 .. 3.0); a span before the window is clipped away
    gaps = find_quiet_gaps([(1.0, 2.0), (1.5, 3.0), (-1.0, 0.5)], 0.0, 6.0, p)
    # union (0 .. 0.5) and (1.0 .. 3.0): the only gap left is after 3.0 + 0.25
    assert gaps == [pytest.approx((3.25, 6.0))]


def test_energy_filter_drops_a_cough_in_the_gap():
    p = AmbienceParams()
    total = 6.0
    rec = _noise(total, 1e-3, seed=3)
    # a cough: 0.7 s of loud noise inside the first gap (1.25 .. 2.9), not marked as speech
    i0, i1 = int(1.5 * SR), int(2.2 * SR)
    rec[i0:i1] += np.random.default_rng(9).normal(0.0, 0.05, i1 - i0)
    spans = [(0.0, 1.0), (3.0, 4.0), (5.0, 6.0)]
    without = find_quiet_gaps(spans, 0.0, total, p)
    with_wave = find_quiet_gaps(spans, 0.0, total, p, waveform=rec, sr=SR)
    assert len(without) == 2
    assert len(with_wave) == 1
    assert with_wave[0] == pytest.approx((4.25, 4.9))


# -- noise_floor_track -------------------------------------------------------

def test_noise_floor_track_length_and_never_zero():
    p = AmbienceParams()
    track = noise_floor_track(np.zeros(10 * SR), SR, [], 1.0, 4.3, p)
    assert track.size == round(3.3 * SR)
    assert np.all(track >= 1e-6)
    empty = noise_floor_track(np.zeros(0), SR, [], 0.0, 1.0, p)
    assert empty.size == SR and np.all(empty >= 1e-6)


def test_noise_floor_track_follows_a_level_change():
    p = AmbienceParams()
    rec = np.concatenate([_noise(5.0, 1e-3, 1), _noise(5.0, 1e-2, 2)])
    track = noise_floor_track(rec, SR, [], 0.0, 10.0, p)
    first = np.median(track[int(1.5 * SR):int(3.5 * SR)])
    second = np.median(track[int(6.5 * SR):int(8.5 * SR)])
    assert 0.5e-3 < first < 2e-3
    assert 5 < second / first < 20


def test_noise_floor_track_ignores_speech():
    p = AmbienceParams()
    rec = _recording(6.0, [(2.0, 4.0)], sigma=1e-3, voice_amp=0.5)
    track = noise_floor_track(rec, SR, [(2.0, 4.0)], 0.0, 6.0, p)
    inside = np.median(track[int(2.2 * SR):int(3.8 * SR)])
    assert 0.5e-3 < inside < 2e-3


# -- build_bed ---------------------------------------------------------------

SPEECH = [(2.0, 8.0), (10.0, 12.0), (20.0, 25.0)]
TOTAL = 30.0


def _setup(sigma=1e-3):
    p = AmbienceParams(seed=7)
    rec = _recording(TOTAL, SPEECH, sigma=sigma)
    gaps = find_quiet_gaps(SPEECH, 0.0, TOTAL, p)
    return p, rec, gaps


def test_build_bed_length_local_and_no_repeats():
    p, rec, gaps = _setup()
    start, end = 5.0, 10.0
    res = build_bed(rec, SR, gaps, SPEECH, start, end, p, focus_s=7.5)
    assert res.audio.size == round((end - start) * SR)
    assert res.audio.dtype == np.float32
    assert res.tier == "local"
    assert res.pieces >= 1
    assert res.repeats == 0


def test_build_bed_tier_file_when_gaps_are_far():
    p, rec, gaps = _setup()
    res = build_bed(rec, SR, gaps, SPEECH, 5.0, 10.0, p, focus_s=500.0)
    assert res.tier == "file"


def test_build_bed_tier_synthetic_without_gaps():
    p, rec, _ = _setup()
    res = build_bed(rec, SR, [], SPEECH, 5.0, 10.0, p)
    assert res.tier == "synthetic"
    assert res.audio.size == round(5.0 * SR)


def test_build_bed_tier_pink_when_nothing_measurable():
    p = AmbienceParams(seed=1)
    res = build_bed(np.zeros(0), SR, [], [], 0.0, 2.0, p)
    assert res.tier == "pink"
    assert res.audio.size == 2 * SR
    assert np.all(res.audio != 0.0)


def test_build_bed_never_silent_never_clips():
    p, rec, gaps = _setup()
    for kwargs in (
        dict(gaps=gaps, focus_s=7.5),
        dict(gaps=gaps, focus_s=500.0),
        dict(gaps=[], focus_s=None),
    ):
        res = build_bed(rec, SR, kwargs["gaps"], SPEECH, 5.0, 10.0, p, focus_s=kwargs["focus_s"])
        assert np.count_nonzero(res.audio == 0.0) == 0
        assert np.max(np.abs(res.audio)) <= 1.0
    assert build_bed(np.zeros(0), SR, [], [], 0.0, 1.0, p).audio.min() != 0.0


def test_build_bed_level_follows_the_floor():
    p, rec, gaps = _setup(sigma=1e-3)
    res = build_bed(rec, SR, gaps, SPEECH, 5.0, 10.0, p, focus_s=7.5)
    rms = float(np.sqrt(np.mean(res.audio.astype(np.float64) ** 2)))
    assert 0.5e-3 < rms < 2e-3
    synth = build_bed(rec, SR, [], SPEECH, 5.0, 10.0, p)
    srms = float(np.sqrt(np.mean(synth.audio.astype(np.float64) ** 2)))
    assert 0.3e-3 < srms < 3e-3
    pink = build_bed(np.zeros(0), SR, [], [], 0.0, 2.0, p)
    prms = float(np.sqrt(np.mean(pink.audio.astype(np.float64) ** 2)))
    assert 0.3 * 10 ** (-65 / 20) < prms < 3 * 10 ** (-65 / 20)


def test_build_bed_is_deterministic_per_seed():
    p, rec, gaps = _setup()
    a = build_bed(rec, SR, gaps, SPEECH, 5.0, 10.0, p, focus_s=7.5).audio
    b = build_bed(rec, SR, gaps, SPEECH, 5.0, 10.0, p, focus_s=7.5).audio
    assert np.array_equal(a, b)
    q = AmbienceParams(seed=8)
    c = build_bed(rec, SR, gaps, SPEECH, 5.0, 10.0, q, focus_s=7.5).audio
    assert not np.array_equal(a, c)


def test_build_bed_repeats_when_gap_is_too_short():
    p = AmbienceParams(seed=2)
    rec = _recording(4.0, [(0.0, 1.0), (2.2, 4.0)], sigma=1e-3)
    gaps = find_quiet_gaps([(0.0, 1.0), (2.2, 4.0)], 0.0, 4.0, p)  # one gap of ~0.95 s
    res = build_bed(rec, SR, gaps, [(0.0, 1.0), (2.2, 4.0)], 0.0, 6.0, p, focus_s=1.0)
    assert res.audio.size == 6 * SR
    assert res.repeats > 0


def _scattered_case():
    """200 s: a short gap next to focus (100 s), long gaps 60+ s away. Excerpt 95..105 needs ~12.4 s of gap."""
    p = AmbienceParams(seed=5)
    spans = [(10.0, 20.0), (30.0, 97.0), (99.0, 200.0)]
    rec = _recording(200.0, spans, sigma=1e-3, seed=5).astype(np.float32)
    gaps = find_quiet_gaps(spans, 0.0, 200.0, p)
    return p, rec, gaps, spans


def test_build_bed_local_tier_when_local_gaps_cover_the_excerpt():
    p, rec, gaps, spans = _scattered_case()
    # 0.5 s excerpt, the 1.65 s gap at 97.25..98.9 is local and covers it
    res = build_bed(rec, SR, gaps, spans, 97.5, 98.0, p, focus_s=97.75)
    assert res.tier == "local"
    assert res.repeats == 0


def test_build_bed_file_tier_when_local_gaps_are_too_short():
    p, rec, gaps, spans = _scattered_case()
    res = build_bed(rec, SR, gaps, spans, 95.0, 105.0, p, focus_s=100.0)
    assert res.tier == "file"
    assert res.repeats == 0
    assert res.audio.size == 10 * SR
    assert np.count_nonzero(res.audio == 0.0) == 0


def test_build_bed_does_not_copy_the_whole_recording():
    import tracemalloc

    p, rec, gaps, spans = _scattered_case()
    rec = np.concatenate([rec] * 3)  # 600 s; a float64 copy of it would be 77 MB
    gaps = [(a + k * 200.0, b + k * 200.0) for k in range(3) for a, b in gaps]
    spans = [(a + k * 200.0, b + k * 200.0) for k in range(3) for a, b in spans]
    tracemalloc.start()
    try:
        res = build_bed(rec, SR, gaps, spans, 300.0, 310.0, p, focus_s=305.0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert res.audio.size == 10 * SR
    assert peak < 20e6


def test_has_hum_detects_a_tone_not_noise():
    t = np.arange(SR * 2) / SR
    assert _has_hum(1e-2 * np.sin(2 * np.pi * 50.0 * t) + _noise(2.0, 1e-4, 4))
    assert not _has_hum(_noise(2.0, 1e-3, 5))


# -- safe_mask ---------------------------------------------------------------

def _idx(t):
    return int(round(t * SR))


def test_safe_mask_dilates_other_speech_and_caps_the_tail():
    p = AmbienceParams()  # dilate 150 ms, tail cap 400 ms
    n = 10 * SR
    m = safe_mask(n, SR, 0.0, [(2.0, 3.0)], [], [], [], p)
    assert m.shape == (n,) and m.dtype == bool
    assert m[_idx(1.8)] and not m[_idx(1.9)]
    assert not m[_idx(3.1)] and m[_idx(3.3)]
    # the measured tail is applied after the speech, capped at reverb_tail_max_ms
    capped = safe_mask(n, SR, 0.0, [(2.0, 3.0)], [], [], [], p, tail_ms=5000)
    assert not capped[_idx(3.5)] and capped[_idx(3.6)]
    short = safe_mask(n, SR, 0.0, [(2.0, 3.0)], [], [], [], p, tail_ms=100)
    assert not short[_idx(3.2)] and short[_idx(3.3)]


def test_safe_mask_third_bad_and_long_overlap():
    p = AmbienceParams()
    n = 10 * SR
    m = safe_mask(n, SR, 0.0, [], [(6.0, 7.0)], [(8.0, 8.5)], [(4.0, 4.5)], p)
    assert not m[_idx(5.95)] and m[_idx(5.8)]  # third speaker, 100 ms margin
    assert not m[_idx(8.2)] and m[_idx(8.6)]   # failed separation, exact
    assert m[_idx(7.95)]                       # no dilation for bad spans
    assert not m[_idx(4.25)] and m[_idx(4.6)]  # long overlap, exact


def test_safe_mask_short_overlap_is_cut_at_its_middle():
    p = AmbienceParams()
    n = 10 * SR
    m = safe_mask(n, SR, 0.0, [], [], [], [(9.0, 9.05)], p)  # 50 ms < min_overlap_s
    assert not m[_idx(9.025)] and not m[_idx(9.0)]
    assert m[_idx(8.9)] and m[_idx(9.2)]


def test_safe_mask_respects_start_offset():
    p = AmbienceParams()
    m = safe_mask(2 * SR, SR, 10.0, [(10.5, 10.7)], [], [], [], p)
    assert not m[_idx(0.6)] and m[_idx(0.2)] and m[_idx(1.9)]


# -- compose_user_channel ----------------------------------------------------

def _compose_case():
    p = AmbienceParams(mode="raw_mask")
    total = 5.0
    n = int(total * SR)
    raw = _recording(total, [(2.0, 3.0)], sigma=1e-3, voice_amp=0.2, seed=11)
    raw[: int(0.5 * SR)] = 0.0  # a digital-silent stretch in the recording
    user = _tone(1.0, 2.0, 150.0, 0.1, total) + _tone(3.5, 4.5, 150.0, 0.1, total)
    bed = _noise(total, 1e-3, seed=12)
    safe = safe_mask(n, SR, 0.0, [(2.0, 3.0)], [], [], [], p)
    return p, raw, user, bed, safe


def test_compose_m1_raw_is_kept_in_the_safe_region():
    p, raw, user, bed, safe = _compose_case()
    out = compose_user_channel(raw, user, bed, safe, SR, p)
    assert out.size == raw.size
    keep = slice(_idx(0.6), _idx(1.75))
    assert np.allclose(out[keep], raw[keep], atol=1e-6)
    keep2 = slice(_idx(3.25), _idx(5.0))
    assert np.allclose(out[keep2], raw[keep2], atol=1e-6)


def test_compose_m1_unsafe_is_user_plus_bed_not_a_double_room():
    p, raw, user, bed, safe = _compose_case()
    out = compose_user_channel(raw, user, bed, safe, SR, p)
    # the other speaker is talking here: no raw, no ghost. Probe where the other voice is loudest.
    seg = slice(_idx(2.0), _idx(3.0))
    t = _idx(2.0) + int(np.argmax(np.abs(raw[seg])))
    assert abs(out[t] - (user[t] + bed[t])) < 1e-6
    assert abs(out[t] - raw[t]) > 0.1


def test_compose_m1_no_digital_silence_longer_than_10ms():
    p, raw, user, bed, safe = _compose_case()
    out = compose_user_channel(raw, user, bed, safe, SR, p)
    zero = out == 0.0
    run = best = 0
    for z in zero:
        run = run + 1 if z else 0
        best = max(best, run)
    assert best <= 0.010 * SR


def test_compose_m2_bed_only_adds_user_to_bed():
    p, raw, user, bed, safe = _compose_case()
    p.mode = "bed_only"
    out = compose_user_channel(raw, user, bed, safe, SR, p)
    assert np.allclose(out, user + bed, atol=1e-6)


def test_compose_never_clips():
    p, raw, user, bed, safe = _compose_case()
    loud_user = user * 9.0
    p.mode = "bed_only"
    out = compose_user_channel(raw, loud_user, bed + 0.5, safe, SR, p)
    assert np.max(np.abs(out)) <= 0.98 + 1e-6
    p.mode = "raw_mask"
    out = compose_user_channel(raw * 3.0, loud_user, bed, safe, SR, p)
    assert np.max(np.abs(out)) <= 1.0


def test_compose_splice_does_not_add_up_a_correlated_voice():
    # the user talks through the safe/unsafe boundary (other speaker from 2.5 s, unsafe from 2.35 s):
    # raw and user_track are the same voice, so the crossfade must not peak above it
    p = AmbienceParams(mode="raw_mask")
    total = 4.0
    n = int(total * SR)
    raw = _tone(1.0, 2.5, 220.0, 0.2, total)
    safe = safe_mask(n, SR, 0.0, [(2.5, 3.0)], [], [], [], p)
    bed = np.zeros(n)
    out = compose_user_channel(raw, raw.copy(), bed, safe, SR, p)
    w = slice(_idx(2.33), _idx(2.37))
    assert np.max(np.abs(out[w])) <= 1.05 * np.max(np.abs(raw[w]))
    assert np.allclose(out[w], raw[w], atol=1e-4)


# -- postprocess_assistant ---------------------------------------------------

def _assistant_case():
    total = 7.0
    n = int(total * SR)
    before = np.zeros(n)
    before[_idx(1.0):_idx(1.5)] = _tone(1.0, 1.5, 200.0, 0.2, total)[_idx(1.0):_idx(1.5)]
    before[_idx(3.0):_idx(3.2)] = _noise(0.2, 0.05, seed=21)  # a backchannel
    before[_idx(5.0):_idx(6.0)] = _tone(5.0, 6.0, 200.0, 0.2, total)[_idx(5.0):_idx(6.0)]
    before[_idx(5.5):_idx(6.0)] = 1e-4 * np.sign(before[_idx(5.5):_idx(6.0)] + 1e-12)
    clean = np.zeros(n)
    clean[_idx(1.0):_idx(1.5)] = 0.3 * before[_idx(1.0):_idx(1.5)]  # chain quieter than before (-10 dB)
    clean[_idx(5.0):_idx(6.0)] = before[_idx(5.0):_idx(6.0)]
    clean[_idx(3.0):_idx(3.2)] = 1e-4 * _noise(0.2, 1.0, seed=22)  # the filter erased the backchannel
    clean += 1e-4 * _noise(total, 1.0, seed=23)  # tails of the chain between sentences
    return clean, before


def test_postprocess_exact_zero_outside_islands():
    clean, before = _assistant_case()
    out, rep = postprocess_assistant(clean, before, SR, AmbienceParams())
    assert out.size == clean.size
    assert out[_idx(0.5)] == 0.0 and out[_idx(2.0)] == 0.0
    assert out[_idx(1.5 + 0.05)] == 0.0
    assert rep["islands"] == 3


def test_postprocess_restores_an_erased_backchannel():
    clean, before = _assistant_case()
    out, rep = postprocess_assistant(clean, before, SR, AmbienceParams())
    assert rep["restored"] == 1
    seg = out[_idx(3.02):_idx(3.18)]
    ref = before[_idx(3.02):_idx(3.18)]
    assert np.sqrt(np.mean(seg ** 2)) > 0.5 * np.sqrt(np.mean(ref ** 2))


def test_postprocess_expander_zeros_under_the_floor():
    clean, before = _assistant_case()
    out, rep = postprocess_assistant(clean, before, SR, AmbienceParams())
    quiet = out[_idx(5.6):_idx(5.9)]
    assert np.all(quiet == 0.0)
    assert np.any(out[_idx(5.1):_idx(5.4)] != 0.0)
    assert rep["zeroed_fraction"] > 0.0


def test_postprocess_level_match_pulls_back_to_the_band():
    clean, before = _assistant_case()
    out, rep = postprocess_assistant(clean, before, SR, AmbienceParams())
    assert rep["level_adjusted"] >= 1
    a = out[_idx(1.0):_idx(1.5)]
    b = before[_idx(1.0):_idx(1.5)]
    ratio_db = 20 * np.log10(np.sqrt(np.mean(a ** 2)) / np.sqrt(np.mean(b ** 2)))
    assert -6.0 - 0.5 <= ratio_db <= 6.0 + 0.5


def test_postprocess_explicit_islands_are_used():
    clean, before = _assistant_case()
    out, rep = postprocess_assistant(clean, before, SR, AmbienceParams(), islands=[(5.0, 6.0)])
    assert rep["islands"] == 1
    assert out[_idx(1.2)] == 0.0


# -- quality_report ----------------------------------------------------------

def test_quality_report_user_silence_outside_spans():
    user = _noise(6.0, 1e-3, seed=31)
    user[_idx(4.0):_idx(4.5)] = 0.0          # 0.5 s of digital silence outside the user's speech
    user[_idx(0.2):_idx(0.8)] = 0.0          # zeros inside the user's speech do not count
    assistant = _noise(6.0, 1e-3, seed=32)
    rep = quality_report(user, assistant, SR, [(0.0, 1.0)], [(2.0, 3.0)], bed_tier="local")
    assert rep["user_digital_silence_s"] == pytest.approx(0.5, abs=1e-3)
    assert rep["bed_tier"] == "local"


def test_quality_report_assistant_floor_and_rms_jump():
    user = np.concatenate([_noise(1.0, 1e-3, seed=41), _noise(1.0, 1e-1, seed=42)])
    assistant = _noise(3.0, 1e-3, seed=43)
    rep = quality_report(user, assistant, SR, [], [], bed_tier=None)
    assert -66.0 < rep["assistant_floor_dbfs"] < -54.0
    assert rep["user_rms_jump_db"] > 30.0
    assert rep["bed_tier"] is None
    smooth = quality_report(_noise(2.0, 1e-3, seed=44), assistant, SR, [], [])
    assert smooth["user_rms_jump_db"] == 0.0
