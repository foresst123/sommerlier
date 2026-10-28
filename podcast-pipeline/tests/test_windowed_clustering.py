import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import windowed_clustering as W  # noqa: E402

FS = 0.08


def _frame_probs(s, e, active_fns):
    """(frames, labels) probabilities for [s, e): 0.9 where a label's predicate holds, else 0.1."""
    n = max(1, int(round((e - s) / FS)))
    t = s + (np.arange(n) + 0.5) * FS
    if not active_fns:
        return np.zeros((n, 0))
    return np.stack([np.where(f(t), 0.9, 0.1) for f in active_fns], axis=1)


def test_pieces_respect_min_max_and_share_30s_of_audio():
    speech = [(i * 30.0, i * 30.0 + 25.0) for i in range(34)]    # 5 s pauses every 30 s
    pieces = W.plan_pieces(1000.0, speech, target_s=240, min_s=180, max_s=300, overlap_s=30)
    assert len(pieces) == 4
    assert pieces[0].core_start == 0.0 and pieces[-1].core_end == 1000.0
    assert pieces[0].audio_start == 0.0 and pieces[-1].audio_end == 1000.0
    for p, q in zip(pieces, pieces[1:]):
        assert p.core_end == q.core_start
        assert p.audio_end - q.audio_start == pytest.approx(30.0)
    for p in pieces:
        assert 180.0 <= p.core_end - p.core_start <= 300.0
    assert W.piece_bounds(240) == (240.0, 180.0, 300.0)


def test_window_spans_slide_and_end_at_duration():
    assert W.window_spans(60.0) == [(0.0, 30.0), (10.0, 40.0), (20.0, 50.0), (30.0, 60.0)]
    assert W.window_spans(35.0) == [(0.0, 30.0), (5.0, 35.0)]
    assert W.window_spans(20.0) == [(0.0, 20.0)]
    assert W.window_spans(0.0) == []


def test_cannot_link_keeps_same_window_labels_apart_even_with_same_voice():
    def prob_fn(s, e):
        return _frame_probs(s, e, [lambda t: t < 10, lambda t: (t >= 15) & (t < 25)])

    def embed_fn(spans):
        return np.array([[1.0, 0.01]] * len(spans))      # both labels sound identical

    res = W.run_piece(30.0, prob_fn, embed_fn)
    win = res.windows[0]
    assert win.cluster_of[0] >= 0 and win.cluster_of[1] >= 0
    assert win.cluster_of[0] != win.cluster_of[1]
    assert res.n_clusters == 2
    assert res.stats["cannot_link_violations"] == 0


def test_swapped_local_labels_still_give_two_consistent_clusters():
    vec_a = np.array([1.0, 0.1, 0.0])
    vec_b = np.array([0.1, 1.0, 0.0])
    rng = np.random.default_rng(0)

    def is_a(t):
        return t < 40.0

    def is_b(t):
        return t >= 40.0

    def prob_fn(s, e):
        idx = int(round(s / 10.0))
        fns = [is_a, is_b] if idx % 2 == 0 else [is_b, is_a]   # local order flips per window
        return _frame_probs(s, e, fns + [lambda t: np.zeros(len(t), dtype=bool)])

    def embed_fn(spans):
        out = []
        for s, e in spans:
            mid = (s + e) / 2.0
            out.append((vec_a if mid < 40.0 else vec_b) + rng.normal(0.0, 0.01, 3))
        return np.array(out)

    res = W.run_piece(80.0, prob_fn, embed_fn)
    assert len(res.windows) == 6
    a_ids = {res.windows[w].cluster_of[0 if w % 2 == 0 else 1] for w in range(4)}
    b_ids = {res.windows[w].cluster_of[1 if w % 2 == 0 else 0] for w in range(2, 6)}
    assert len(a_ids) == 1 and len(b_ids) == 1
    assert -1 not in a_ids | b_ids and a_ids != b_ids
    assert res.n_clusters == 2
    c_a, c_b = a_ids.pop(), b_ids.pop()
    at_20 = [s["cluster"] for s in res.segments if s["start"] <= 20 <= s["end"]]
    at_60 = [s["cluster"] for s in res.segments if s["start"] <= 60 <= s["end"]]
    assert at_20 == [c_a] and at_60 == [c_b]


def test_hungarian_beats_greedy_on_the_swap_matrix():
    sim = np.array([[0.70, 0.68], [0.69, 0.20]])
    # greedy takes (0, 0) and (1, 1) for 0.90; the assignment takes (0, 1) and (1, 0) for 1.37
    assert list(W.assign_labels(sim)) == [1, 0]


def test_stitched_pieces_keep_the_same_speaker_labels():
    duration = 500.0
    vecs = {True: np.array([1.0, 0.1, 0.0]), False: np.array([0.1, 1.0, 0.0])}

    def truth_a(t):
        return np.floor(t / 20.0) % 2 == 0

    def truth_b(t):
        return ~truth_a(t)

    def make_prob_fn(piece):
        def prob_fn(s, e):
            fns = [truth_a, truth_b]
            if int(round((piece.audio_start + s) / 10.0)) % 2:
                fns = [truth_b, truth_a]                 # local order differs per piece too
            return _frame_probs(s, e, [lambda t, f=f: f(t + piece.audio_start) for f in fns])
        return prob_fn

    def make_embed_fn(piece):
        rng = np.random.default_rng(1)

        def embed_fn(spans):
            out = []
            for s, e in spans:
                mid = piece.audio_start + (s + e) / 2.0
                out.append(vecs[bool(math.floor(mid / 20.0) % 2 == 0)] + rng.normal(0.0, 0.01, 3))
            return np.array(out)
        return embed_fn

    out = W.run_file(duration, [(0.0, duration)], make_prob_fn=make_prob_fn, make_embed_fn=make_embed_fn)
    assert len(out["pieces"]) == 2
    segs = out["segments"]
    assert len({s["speaker"] for s in segs}) == 2
    by_truth = {}
    for seg in segs:
        mid = (seg["start"] + seg["end"]) / 2.0
        by_truth.setdefault(bool(math.floor(mid / 20.0) % 2 == 0), set()).add(seg["speaker"])
    assert all(len(v) == 1 for v in by_truth.values())
    assert by_truth[True] != by_truth[False]
    assert sum(s["end"] - s["start"] for s in segs) > 0.9 * duration


def test_weak_label_joins_nearest_cluster_without_a_new_speaker():
    vec_a = np.array([1.0, 0.1, 0.0])
    vec_b = np.array([0.1, 1.0, 0.0])

    def prob_fn(s, e):
        if s == 0.0:      # only the weak label, 0.5 s of clean speech
            return _frame_probs(s, e, [lambda t: (t >= 10.0) & (t < 10.5)])
        if s in (10.0, 20.0):
            return _frame_probs(s, e, [lambda t: t < 30.0])          # speaker A
        return _frame_probs(s, e, [lambda t: t >= 30.0])             # speaker B

    def embed_fn(spans):
        return np.array([vec_a if (s + e) / 2.0 < 30.0 else vec_b for s, e in spans])

    res = W.run_piece(60.0, prob_fn, embed_fn)
    assert res.n_clusters == 2
    assert not res.windows[0].has_emb[0]
    assert res.windows[0].cluster_of[0] == res.windows[1].cluster_of[0]
    assert res.windows[3].cluster_of[0] != res.windows[1].cluster_of[0]
    assert res.stats["weak_labels"] == 1


def test_silence_gives_no_segments():
    def prob_fn(s, e):
        return np.zeros((int(round((e - s) / FS)), 3))

    def embed_fn(spans):
        raise AssertionError("no speech, so nothing to embed")

    res = W.run_piece(30.0, prob_fn, embed_fn)
    assert res.segments == [] and res.n_clusters == 0
    out = W.run_file(60.0, [], make_prob_fn=lambda p: prob_fn, make_embed_fn=lambda p: embed_fn)
    assert out["segments"] == []


def test_postprocessing_matches_the_nemotron_segments_values():
    from utils.nemotron_segments import POSTPROCESSING_BASE
    assert W.POSTPROCESSING == POSTPROCESSING_BASE


def test_run_script_loads_the_module_by_path():
    # The NeMo-side script loads modules by path; a dataclass in the module fails to build
    # unless the module is registered in sys.modules first.
    from tools import windowed_nemotron_run as script
    mod = script._load("windowed_clustering_by_path", "utils/windowed_clustering.py")
    try:
        assert callable(mod.run_piece)
        assert mod.Piece(0.0, 1.0, 0.0, 1.0).audio_duration == 1.0
    finally:
        sys.modules.pop("windowed_clustering_by_path", None)
