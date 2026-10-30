import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import dz_pipeline as DZ  # noqa: E402
from utils import windowed_clustering as W  # noqa: E402

FS = 0.08
REF_VBX = ("/tmp/claude-0/-home-user-sommerlier/bcd0ee40-8f99-5834-b118-52f1f997448f/scratchpad/vbx_ref/VBx.py")


def _speaker_at(t, turn=6.0, n=3):
    return int(t // turn) % n


def _make_fakes(n_speakers=3, dim=32, n_slots=4, seed=0, overlap=None):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_speakers, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    def truth(t):
        out = [_speaker_at(t, n=n_speakers)]
        if overlap and overlap[0] <= t < overlap[1]:
            out.append((out[0] + 1) % n_speakers)
        return out

    def prob_fn(s, e):
        n = int(round((e - s) / FS))
        t = s + (np.arange(n) + 0.5) * FS
        perm = rng.permutation(n_slots)                  # local slot labels are arbitrary in every window
        out = np.full((n, n_slots), 0.05)
        for i, ti in enumerate(t):
            for spk in truth(ti):
                out[i, perm[spk]] = 0.95
        return out

    def embed_fn(span, mask):
        t = span[0] + (np.flatnonzero(mask > 0) + 0.5) * FS
        votes = np.bincount([sp for ti in t for sp in truth(ti)[:1]], minlength=n_speakers)
        return centres[int(np.argmax(votes))] * 4.0 + rng.normal(scale=0.2, size=dim)

    def vbx_fn(train):
        sims = train @ centres.T
        labels = np.argmax(sims, axis=1)
        q = np.eye(n_speakers)[labels]
        vbx_fn.last_info = {"train_embeddings": len(train)}
        return q, np.ones(n_speakers) / n_speakers

    return prob_fn, embed_fn, vbx_fn, truth


def _agreement(segments, truth, duration, margin=0.5):
    """Share of frames, away from speaker changes, where the clusters match the truth under the best mapping."""
    t = (np.arange(int(duration / FS)) + 0.5) * FS
    n_spk = 1 + max(sp for ti in t for sp in truth(ti))
    clusters = sorted({r["cluster"] for r in segments})
    pred = np.full((len(clusters), len(t)), False)
    for r in segments:
        pred[clusters.index(r["cluster"])] |= (t >= r["start"]) & (t < r["end"])
    ok = total = 0
    for spk in range(n_spk):
        want = np.array([spk in truth(ti) for ti in t])
        near_change = np.array([min((ti - 6.0 * round(ti / 6.0)) ** 2, 9) ** 0.5 < margin for ti in t])
        keep = ~near_change
        best = max((np.mean(pred[c][keep] == want[keep]) for c in range(len(clusters))), default=0.0)
        ok += best
        total += 1
    return ok / total


def test_binarise_window_thresholds_and_median_filters():
    raw = np.array([[0.9, 0.1]] * 5 + [[0.1, 0.1]] + [[0.9, 0.1]] * 5)      # one-frame hole in slot 0
    out = DZ.binarise_window(raw, thr=0.5, median_frames=3)
    assert out[:, 0].tolist() == [1.0] * 11 and out[:, 1].sum() == 0
    assert DZ.binarise_window(raw, thr=0.5, median_frames=1)[5, 0] == 0.0


def test_speaker_count_is_the_rounded_mean_over_covering_windows():
    a = np.ones((4, 2))                                   # two active slots on frames 0-3
    b = np.zeros((4, 2))
    b[:, 0] = 1.0                                         # one active slot on frames 2-5
    count = DZ.speaker_count([(0, a), (2, b)], 8, max_speakers=20)
    assert count.tolist() == [2, 2, 2, 2, 1, 1, 0, 0]     # frames 2-3: mean(2, 1) = 1.5 rounds to 2 (half to even)
    assert DZ.speaker_count([(0, np.ones((3, 5)))], 3, max_speakers=2).tolist() == [2, 2, 2]


def test_binarize_frames_cuts_at_frame_middles_like_pyannote():
    values = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1])
    segs = DZ.binarize_frames(values, 0.1)
    assert segs == [pytest.approx((0.25, 0.55)), pytest.approx((0.75, 0.85))]    # middles of frames 2 / 5, 7 / last
    assert DZ.binarize_frames(np.zeros(3), 0.1) == [] and DZ.binarize_frames(np.array([]), 0.1) == []


def test_constrained_argmax_is_a_hungarian_assignment_and_leaves_extra_slots_out():
    soft = np.array([[[0.70, 0.68], [0.69, 0.20]]])      # the greedy pick would take 0.70 first and strand slot 1
    assert DZ.constrained_argmax(soft).tolist() == [[1, 0]]
    soft = np.array([[[0.9], [0.8], [np.nan]]])           # one cluster, three slots: one wins, nan rows take the minimum
    hard = DZ.constrained_argmax(soft)[0]
    assert hard.tolist() == [0, -2, -2]


def test_three_speakers_with_shuffled_slots_come_out_as_three_clusters():
    prob_fn, embed_fn, vbx_fn, truth = _make_fakes()
    res = DZ.run_piece_dz(60.0, prob_fn, embed_fn, vbx_fn=vbx_fn, window_s=16.0, hop_s=0.8, frame_s=FS)
    assert res.n_clusters == 3
    assert res.stats["windows"] > 50 and res.stats["embeddings"] > 100
    assert _agreement(res.segments, truth, 60.0) > 0.95
    for fit in res.windows:                                # cannot-link: two slots of a window never share a cluster
        used = fit.cluster_of[fit.cluster_of >= 0]
        assert len(used) == len(set(used.tolist()))


def test_overlap_keeps_two_clusters_on_at_once_through_the_count():
    prob_fn, embed_fn, vbx_fn, truth = _make_fakes(overlap=(20.0, 23.0))
    res = DZ.run_piece_dz(60.0, prob_fn, embed_fn, vbx_fn=vbx_fn, window_s=16.0, hop_s=0.8, frame_s=FS)
    t = 21.5
    on = {r["cluster"] for r in res.segments if r["start"] <= t < r["end"]}
    assert len(on) == 2


def test_the_mask_drops_overlap_frames_unless_too_few_clean_frames_are_left():
    calls = []
    n_slot = 2

    def prob_fn(s, e):
        n = int(round((e - s) / FS))
        out = np.full((n, n_slot), 0.05)
        out[:, 0] = 0.95
        out[n // 2:, 1] = 0.95                            # slot 1 overlaps slot 0 on the second half
        return out

    def embed_fn(span, mask):
        calls.append(mask.copy())
        return np.array([1.0, 0.0])

    def vbx_fn(train):
        return np.ones((len(train), 1)), np.ones(1)

    DZ.run_piece_dz(16.0, prob_fn, embed_fn, vbx_fn=vbx_fn, window_s=16.0, hop_s=0.8, frame_s=FS, min_num_frames=1)
    n = len(calls[0])
    slot0, slot1 = calls[0], calls[1]
    assert slot0[: n // 2].all() and not slot0[n // 2:].any()     # clean frames only
    assert slot1.sum() == n // 2 and slot1[n // 2:].all()         # no clean frame left: the whole mask is used


def test_embeddings_below_the_clean_frame_ratio_are_not_clustered_on_but_are_still_assigned():
    n_slot = 2
    seen = {}

    def prob_fn(s, e):
        n = int(round((e - s) / FS))
        out = np.full((n, n_slot), 0.05)
        out[:100, 0] = 0.95                               # 100 clean frames of 200
        out[150:152, 1] = 0.95                            # 2 clean frames: below the 10% ratio (20 frames)
        return out

    def embed_fn(span, mask):
        return np.array([1.0, 0.0]) if mask.sum() > 5 else np.array([0.0, 1.0])

    def vbx_fn(train):
        seen["train"] = train.copy()
        return np.ones((len(train), 1)), np.ones(1)

    res = DZ.run_piece_dz(30.0, prob_fn, embed_fn, vbx_fn=vbx_fn, window_s=16.0, hop_s=4.0, frame_s=FS,
                          min_num_frames=1)
    assert (np.abs(seen["train"] - np.array([1.0, 0.0])).sum(axis=1) < 1e-9).all()    # the short slot is left out
    assert res.stats["train_embeddings"] < res.stats["embeddings"]
    assert all(fit.cluster_of[1] >= 0 for fit in res.windows)                          # but it gets a cluster


def test_fewer_than_two_embeddings_gives_one_cluster_without_calling_vbx():
    def prob_fn(s, e):
        out = np.full((int(round((e - s) / FS)), 2), 0.05)
        out[:, 0] = 0.95
        return out

    def vbx_fn(train):
        raise AssertionError("VBx must not run below two embeddings")

    res = DZ.run_piece_dz(10.0, prob_fn, lambda span, mask: np.array([1.0, 2.0]), vbx_fn=vbx_fn, window_s=10.0,
                          hop_s=1.0, frame_s=FS)
    assert res.n_clusters == 1 and res.segments and {r["cluster"] for r in res.segments} == {0}


def test_np_unique_shift_leaves_an_empty_cluster_zero_unless_it_is_dropped():
    def prob_fn(s, e):
        out = np.full((int(round((e - s) / FS)), 3), 0.05)
        out[:, 0] = 0.95                                  # one speaker, three slots: two stay inactive
        return out

    def vbx_fn(train):
        return np.ones((len(train), 1)), np.ones(1)

    kwargs = dict(vbx_fn=vbx_fn, window_s=10.0, hop_s=1.0, frame_s=FS)
    with_ids = DZ.run_piece_dz(20.0, prob_fn, lambda span, mask: np.array([1.0, 0.0]), drop_empty_clusters=False, **kwargs)
    dropped = DZ.run_piece_dz(20.0, prob_fn, lambda span, mask: np.array([1.0, 0.0]), drop_empty_clusters=True, **kwargs)
    assert with_ids.n_clusters == 2 and {r["cluster"] for r in with_ids.segments} == {1}   # DiariZen's shifted ids
    assert dropped.n_clusters == 1 and {r["cluster"] for r in dropped.segments} == {0}


@pytest.mark.skipif(not os.path.isfile(REF_VBX), reason="needs the reference VBx.py downloaded outside the repo")
def test_with_the_reference_vbx_two_speakers_stay_two_clusters():
    # Runs only where the scratchpad copy of diarizen/clustering/VBx.py exists (the repo does not carry it).
    spec = importlib.util.spec_from_file_location("vbx_ref", REF_VBX)
    vbx = importlib.util.module_from_spec(spec)
    sys.modules["vbx_ref"] = vbx
    spec.loader.exec_module(vbx)
    dim = 48
    prob_fn, embed_fn, _, truth = _make_fakes(n_speakers=2, dim=dim, seed=3)
    fn = DZ.make_dz_vbx_fn(vbx.cluster_vbx, lambda x: np.sqrt(dim) * x, lambda x, lda_dim: x[:, :lda_dim],
                           np.ones(dim), lda_dim=32)
    res = DZ.run_piece_dz(60.0, prob_fn, embed_fn, vbx_fn=fn, window_s=16.0, hop_s=0.8, frame_s=FS)
    assert fn.last_info["train_embeddings"] > 100
    assert res.n_clusters == 2 and _agreement(res.segments, truth, 60.0) > 0.9


def test_run_file_stitches_two_dz_pieces_into_the_same_three_speakers():
    prob_fn, embed_fn, vbx_fn, truth = _make_fakes(seed=5)

    pieces = [W.Piece(0.0, 50.0, 0.0, 60.0), W.Piece(50.0, 100.0, 40.0, 100.0)]

    def make_prob(piece):
        return lambda s, e: prob_fn(s + piece.audio_start, e + piece.audio_start)

    def make_embed(piece):
        return lambda span, mask: embed_fn((span[0] + piece.audio_start, span[1] + piece.audio_start), mask)

    out = W.run_file(100.0, None, make_prob_fn=make_prob, make_embed_fn=make_embed, pieces=pieces, piece_s=240.0,
                     window_s=16.0, hop_s=0.8, frame_s=FS, piece_fn=DZ.run_piece_dz, vbx_fn=vbx_fn)
    labels = {r["speaker"] for r in out["segments"]}
    assert len(labels) == 3
    assert [p["clusters"] for p in out["pieces"]] == [3, 3]


def test_active_slots_left_without_a_cluster_are_counted():
    # Two slots are active in every window but VBx finds one cluster: the Hungarian step can serve only one of them.
    def prob_fn(s, e):
        n = int(round((e - s) / FS))
        out = np.full((n, 2), 0.05)
        out[: n // 2, 0] = 0.95
        out[n // 2:, 1] = 0.95
        return out

    def vbx_fn(train):
        return np.ones((len(train), 1)), np.ones(1)

    res = DZ.run_piece_dz(30.0, prob_fn, lambda span, mask: np.array([1.0, 0.0]), vbx_fn=vbx_fn,
                          window_s=16.0, hop_s=4.0, frame_s=FS)
    assert res.stats["active_unassigned"] == res.stats["windows"]


def test_inspect_probs_summarises_a_probability_dump(tmp_path, capsys):
    from tools import inspect_probs as I
    piece = W.Piece(0.0, 20.0, 0.0, 20.0)
    spans = [(0.0, 16.0), (2.0, 18.0)]
    probs = []
    for s, e in spans:
        n = int(round((e - s) / FS))
        p = np.full((n, 3), 0.05)
        p[: n // 2, 0] = 0.9
        p[n // 4: 3 * n // 4, 1] = 0.9           # overlaps slot 0 for a quarter of the window
        probs.append(p)
    W.save_piece_probs(str(tmp_path / "piece_000.npz"), piece, spans, probs, FS)
    (tmp_path / "plan.json").write_text('{"frame_s": %s, "pieces": [{"file": "piece_000.npz"}]}' % FS, encoding="utf-8")
    wins, frame_s = I.load_windows(str(tmp_path))
    r = I.summarise(wins, frame_s, 0.5)
    assert r["windows"] == 2 and r["active_slots_per_window"] == 2.0
    assert 0.2 < r["overlap"] < 0.5 and r["overlap_3plus"] == 0.0
    assert I.main([str(tmp_path)]) == 0
    assert "clean>=10%" in capsys.readouterr().out


def test_median_filter_size_follows_seconds_not_the_frame_rate():
    # Nemotron-3's output is 10 ms per frame: 0.22 s is 23 frames there, 3 frames of 80 ms.
    assert DZ.frames_for_seconds(0.22, 0.01) == 23
    assert DZ.frames_for_seconds(0.22, 0.08) == 3
    assert DZ.frames_for_seconds(0.0, 0.01) == 1
    assert DZ.frames_for_seconds(0.2, 0.01) == 21            # 20 is even: the next odd size


def test_drop_short_on_runs_keeps_runs_of_min_on_s_and_longer_and_never_touches_off_frames():
    fs = 0.05                                             # 0.15 s = 3 frames, 0.2 s = 4 frames, 0.25 s = 5 frames
    track = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0], dtype=float)
    out = DZ.drop_short_on_runs(track, fs, 0.2)
    assert out.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]   # 3-frame and 1-frame runs gone
    assert (out[track == 0] == 0).all()                    # OFF stays OFF
    assert DZ.drop_short_on_runs(track, fs, 0.0).tolist() == track.tolist()
    assert DZ.drop_short_on_runs(np.array([]), fs, 0.2).size == 0


def test_min_on_s_removes_a_short_on_run_from_the_binarised_output():
    fs = 0.05
    track = np.zeros(40)
    track[2:5] = 1.0                                       # 0.15 s
    track[10:15] = 1.0                                     # 0.25 s
    track[20:24] = 1.0                                     # exactly 0.2 s
    kept = DZ.binarize_frames(DZ.drop_short_on_runs(track, fs, 0.2), fs)
    assert kept == [pytest.approx((0.525, 0.775)), pytest.approx((1.025, 1.225))]
    assert DZ.binarize_frames(track, fs) == [pytest.approx((0.125, 0.275)), pytest.approx((0.525, 0.775)),
                                             pytest.approx((1.025, 1.225))]


def test_run_piece_dz_with_min_on_s_zero_is_the_same_output_as_without_the_argument():
    prob_fn, embed_fn, vbx_fn, _ = _make_fakes(overlap=(20.0, 23.0))
    plain = DZ.run_piece_dz(60.0, prob_fn, embed_fn, vbx_fn=vbx_fn, window_s=16.0, hop_s=0.8, frame_s=FS)
    prob_fn, embed_fn, vbx_fn, _ = _make_fakes(overlap=(20.0, 23.0))
    zero = DZ.run_piece_dz(60.0, prob_fn, embed_fn, vbx_fn=vbx_fn, window_s=16.0, hop_s=0.8, frame_s=FS,
                           min_on_s=0.0)
    assert zero.segments == plain.segments and zero.n_clusters == plain.n_clusters
    assert zero.stats == plain.stats and zero.windows[0].cluster_of.tolist() == plain.windows[0].cluster_of.tolist()


def test_run_piece_dz_min_on_s_cleans_each_cluster_on_its_own():
    # The overlap piece: speaker 2 sits on speaker 1 for 3 s (20-23). Turns last 6 s. A 3.5 s minimum drops only the
    # 3 s overlap run; the turns of both clusters stay and the other speaker keeps talking through the overlap.
    prob_fn, embed_fn, vbx_fn, _ = _make_fakes(overlap=(20.0, 23.0))
    plain = DZ.run_piece_dz(60.0, prob_fn, embed_fn, vbx_fn=vbx_fn, window_s=16.0, hop_s=0.8, frame_s=FS)
    prob_fn, embed_fn, vbx_fn, _ = _make_fakes(overlap=(20.0, 23.0))
    cut = DZ.run_piece_dz(60.0, prob_fn, embed_fn, vbx_fn=vbx_fn, window_s=16.0, hop_s=0.8, frame_s=FS,
                          min_on_s=3.5)
    # each run is dropped or kept as a whole, so the cut output is the plain output without its short runs
    short = [r for r in plain.segments if r["end"] - r["start"] < 3.5 - 1e-9]
    assert short and len(short) < len(plain.segments)
    assert cut.segments == [r for r in plain.segments if r not in short]
    assert all(r["end"] - r["start"] >= 3.5 - 1e-6 for r in cut.segments)
    assert len({r["cluster"] for r in cut.segments if r["start"] <= 21.5 < r["end"]}) == 1   # overlap run gone
    assert cut.n_clusters == plain.n_clusters and cut.stats == plain.stats    # counts and VBx are untouched


def test_negative_or_nan_min_on_s_is_refused():
    prob_fn, embed_fn, vbx_fn, _ = _make_fakes()
    for bad in (-0.1, float("nan")):
        with pytest.raises(ValueError):
            DZ.run_piece_dz(10.0, prob_fn, embed_fn, vbx_fn=vbx_fn, window_s=10.0, hop_s=1.0, frame_s=FS,
                            min_on_s=bad)
