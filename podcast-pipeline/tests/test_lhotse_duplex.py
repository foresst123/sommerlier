import numpy as np
import soundfile as sf

from utils.lhotse_duplex import (
    BackchannelParams,
    CoverageParams,
    anchor_backchannel_ids,
    check_item_coverage,
    find_unannotated_speech,
    resolve_backchannels,
)

SR = 16000


def turn(tid, ch, start, dur, ss=None, se=None, words=None):
    words = words if words is not None else [
        [f"w{i}", start + 0.5 * i, 0.4, None] for i in range(6)
    ]
    return {
        "id": tid, "channel": ch, "start": start, "duration": dur,
        "speech_start": start if ss is None else ss,
        "speech_end": start + dur if se is None else se,
        "words": words,
    }


def cand(key, ch, start, end):
    return {
        "key": key, "channel": ch, "role": "user" if ch == 0 else "assistant",
        "speaker": "s", "start": start, "end": end, "text": "ừ",
        "words": [["ừ", start, end - start, None]], "source": "event",
    }


def test_attach_word_index_and_fields():
    anchor = turn("a", 1, 0.0, 4.0)  # words start at 0,.5,1,1.5,2,2.5
    r = resolve_backchannels([anchor], [cand("c", 0, 1.2, 1.5)])
    assert not r.demoted and not r.dropped
    bc = r.attached[0]
    assert bc["id"] == "a-bc0" and bc["k"] == 0 and bc["anchor_id"] == "a"
    assert bc["anchor_word_index"] == 3  # words at 0, .5, 1.0 start <= 1.2
    assert bc["channel"] == 1 - anchor["channel"]


def test_word_index_clamped():
    anchor = turn("a", 1, 0.0, 10.0)
    early = resolve_backchannels([anchor], [cand("c", 0, 0.0, 0.2)]).attached[0]
    assert early["anchor_word_index"] == 1
    late = resolve_backchannels([anchor], [cand("c", 0, 9.0, 9.3)]).attached[0]
    assert late["anchor_word_index"] == 6
    # starts before the first word: still >= 1
    a2 = turn("b", 1, 0.0, 5.0, words=[["x", 1.0, 0.3, None]])
    assert resolve_backchannels([a2], [cand("c", 0, 0.2, 0.4)]).attached[0]["anchor_word_index"] == 1


def test_beyond_anchor_span_is_demoted():
    anchor = turn("a", 1, 0.0, 2.0)
    r = resolve_backchannels([anchor], [cand("c", 0, 1.8, 2.5)])
    assert r.demoted and not r.attached and not r.dropped


def test_overlap_same_channel_turn_dropped():
    anchor = turn("a", 1, 0.0, 2.0)
    own = turn("u", 0, 3.0, 2.0)
    r = resolve_backchannels([anchor, own], [cand("c", 0, 3.5, 3.8)])
    assert len(r.dropped) == 1
    assert "no_anchor_same_channel_overlap" in r.dropped[0]["reason"]
    assert not r.demoted


def test_anchor_without_words_dropped():
    anchor = turn("a", 1, 0.0, 2.0, words=[])
    r = resolve_backchannels([anchor], [cand("c", 0, 0.5, 0.7)])
    assert r.dropped[0]["reason"] == "anchor_without_words"


def test_multiple_backchannels_k_and_ids():
    anchor = turn("a", 1, 0.0, 5.0)
    r = resolve_backchannels([anchor], [cand("c2", 0, 3.0, 3.2), cand("c1", 0, 1.0, 1.2)])
    assert [b["id"] for b in r.attached] == ["a-bc0", "a-bc1"]
    assert [b["key"] for b in r.attached] == ["c1", "c2"]
    assert anchor_backchannel_ids(r.attached) == {"a": ["a-bc0", "a-bc1"]}
    # idempotent
    assert resolve_backchannels([anchor], [cand("c2", 0, 3.0, 3.2), cand("c1", 0, 1.0, 1.2)]).attached == r.attached


def test_anchor_choice_prefers_speech_then_latest_start():
    t1 = turn("t1", 1, 0.0, 6.0, ss=0.0, se=2.0)
    t2 = turn("t2", 1, 1.0, 6.0, ss=1.0, se=6.0)
    r = resolve_backchannels([t1, t2], [cand("c", 0, 3.0, 3.2)])
    assert r.attached[0]["anchor_id"] == "t2"
    r = resolve_backchannels([t1, t2], [cand("c", 0, 1.5, 1.7)])  # both speaking -> later start
    assert r.attached[0]["anchor_id"] == "t2"
    t3 = turn("t3", 1, 1.0, 6.0, ss=4.0, se=6.0)
    r = resolve_backchannels([t1, t3], [cand("c", 0, 1.5, 1.7)])  # only t1 speaking
    assert r.attached[0]["anchor_id"] == "t1"


def tone(n_sec, amp=0.1):
    t = np.arange(int(n_sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def make_wav(path, left_regions, right_regions, total=6.0):
    data = np.zeros((int(total * SR), 2), dtype=np.float32)
    for ch, regions in ((0, left_regions), (1, right_regions)):
        for s, e in regions:
            data[int(s * SR):int(e * SR), ch] = tone(e - s)
    sf.write(str(path), data, SR)
    return str(path)


def test_coverage_flags_unannotated_per_channel(tmp_path):
    wav = make_wav(tmp_path / "a.wav", [(1.0, 2.0)], [(3.0, 4.0)])
    v = check_item_coverage(wav, {0: [], 1: [(3.0, 4.0)]})
    assert len(v) == 1 and v[0]["channel"] == 0
    assert abs(v[0]["start"] - 1.0) < 0.05 and abs(v[0]["end"] - 2.0) < 0.05
    v = check_item_coverage(wav, {0: [(1.0, 2.0)], 1: [(0.0, 1.0)]})
    assert [x["channel"] for x in v] == [1]


def test_coverage_partial_uncovered_tail(tmp_path):
    wav = make_wav(tmp_path / "a.wav", [(1.0, 2.0)], [])
    v = check_item_coverage(wav, {0: [(1.0, 1.5)], 1: []})
    assert len(v) == 1 and 1.4 < v[0]["start"] < 1.6


def test_coverage_ok_when_covered_silent_or_short(tmp_path):
    wav = make_wav(tmp_path / "a.wav", [(1.0, 2.0)], [(3.0, 3.2)])  # ch1 only 200 ms
    assert check_item_coverage(wav, {0: [(0.9, 2.1)], 1: []}) == []
    silent = make_wav(tmp_path / "s.wav", [], [])
    assert check_item_coverage(silent, {0: [], 1: []}) == []


def test_find_unannotated_ignores_clicks():
    x = np.zeros(SR * 2, dtype=np.float32)
    x[1000:1100] = 0.5  # ~6 ms click
    assert find_unannotated_speech(x, SR, []) == []


def noise(n_sec, rms, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(n_sec * SR)) * rms).astype(np.float32)


def write_stereo(path, ch0, ch1):
    sf.write(str(path), np.stack([ch0, ch1], axis=1), SR)
    return str(path)


def test_room_tone_on_user_channel_is_not_unannotated_speech(tmp_path):
    # (a) user channel = continuous room tone RMS 0.01 (above the fixed 0.005), speech 1-3 s annotated
    user = noise(10.0, 0.01, seed=1)
    user[SR:3 * SR] = noise(2.0, 0.2, seed=2)
    wav = write_stereo(tmp_path / "a.wav", user, np.zeros_like(user))
    thresholds = {}
    assert check_item_coverage(wav, {0: [(1.0, 3.0)], 1: []}, thresholds=thresholds) == []
    assert 0.03 < thresholds[0] < 0.045  # about 4 x the 0.0096 floor
    # with the floor switched off, the same room is rejected (the old behaviour)
    assert find_unannotated_speech(user, SR, [(1.0, 3.0)], CoverageParams(floor_factor=0))


def test_speech_over_room_tone_still_violates():
    # (b) room tone RMS 0.01 everywhere, plus an unannotated 3 s speech at RMS 0.2
    user = noise(10.0, 0.01, seed=3)
    user[5 * SR:8 * SR] = noise(3.0, 0.2, seed=4)
    regions = find_unannotated_speech(user, SR, [(1.0, 2.0)])
    assert len(regions) == 1
    s, e = regions[0]
    assert abs(s - 5.0) < 0.05 and abs(e - 8.0) < 0.05


def test_assistant_gaps_keep_fixed_threshold(tmp_path):
    # (c) assistant channel: exact zeros between sentences, one 4 s sentence unannotated
    asst = np.zeros(int(10 * SR), dtype=np.float32)
    asst[SR // 2:3 * SR // 2] = noise(1.0, 0.07, seed=7)[:SR]  # annotated sentence
    asst[3 * SR:7 * SR] = noise(4.0, 0.07, seed=8)            # unannotated sentence
    wav = write_stereo(tmp_path / "c.wav", np.zeros_like(asst), asst)
    thresholds = {}
    v = check_item_coverage(wav, {0: [], 1: [(0.5, 1.5)]}, thresholds=thresholds)
    assert [(x["channel"], round(x["start"]), round(x["end"])) for x in v] == [(1, 3, 7)]
    assert thresholds[1] == 0.005  # zero floor: the fixed threshold stays


def test_all_zero_channel_has_no_violation(tmp_path):
    # (d) a channel of only exact zeros
    zeros = np.zeros(int(6 * SR), dtype=np.float32)
    wav = write_stereo(tmp_path / "d.wav", zeros, zeros)
    thresholds = {}
    assert check_item_coverage(wav, {0: [], 1: []}, thresholds=thresholds) == []
    assert thresholds == {0: 0.005, 1: 0.005}
