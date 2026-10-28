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


# ---- arm G: cluster_fn hook, VBx clustering, probability dump ----

REF_VBX = "/tmp/claude-0/-home-user-sommerlier/bcd0ee40-8f99-5834-b118-52f1f997448f/scratchpad/vbx_ref/VBx.py"


def _two_speaker_piece_fns():
    """Piece of 60 s: speaker A (label 0) before t=25, speaker B (label 1) after. Windows hold both."""
    def prob_fn(s, e):
        return _frame_probs(s, e, [lambda t: t < 25.0, lambda t: t >= 25.0])

    def embed_fn(spans):
        return np.array([[1.0, 0.0, 0.0] if (s + e) / 2.0 < 25.0 else [0.0, 1.0, 0.0] for s, e in spans])

    return prob_fn, embed_fn


def _oracle_vbx(calls=None):
    """Fake cluster_vbx: one-hot responsibilities by speaker, read off the first two PLDA dims."""
    def fake(ahc_init, fea, Phi, Fa, Fb, loopProb=0.0, maxIters=20, init_smoothing=7.0):
        if calls is not None:
            calls.append({"Fa": Fa, "Fb": Fb, "maxIters": maxIters, "init_smoothing": init_smoothing,
                          "phi_len": len(Phi), "n": len(ahc_init)})
        spk = (fea[:, 1] > fea[:, 0]).astype(int)
        return np.eye(2)[spk], np.array([0.5, 0.5])
    return fake


def _identity_tf():
    return (lambda x: x), (lambda x, lda_dim=128: x[:, :lda_dim])


def test_cluster_fn_none_is_the_old_constrained_ahc_path():
    prob_fn, embed_fn = _two_speaker_piece_fns()
    base = W.run_piece(60.0, prob_fn, embed_fn)
    same = W.run_piece(60.0, prob_fn, embed_fn, cluster_fn=None)
    assert same.segments == base.segments and same.n_clusters == base.n_clusters
    assert "vbx_fallback" not in same.stats


def test_custom_cluster_fn_gets_embeddings_windows_and_speaker_bounds():
    prob_fn, embed_fn = _two_speaker_piece_fns()
    seen = {}

    def cluster_fn(x, win_of, min_k, max_k):
        seen.update(n=len(x), win_of=np.asarray(win_of).copy(), min_k=min_k, max_k=max_k)
        return np.arange(len(x)) % 2

    res = W.run_piece(60.0, prob_fn, embed_fn, cluster_fn=cluster_fn, min_speakers=1, max_speakers=4)
    assert seen["n"] == res.stats["strong_labels"] and seen["min_k"] == 1 and seen["max_k"] == 4
    assert len(seen["win_of"]) == seen["n"]
    assert res.n_clusters == 2 and res.stats["cannot_link_violations"] == 0


def test_vbx_cluster_fn_separates_swapped_local_labels():
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([0.0, 1.0, 0.0])
    rng = np.random.default_rng(0)

    def prob_fn(s, e):
        idx = int(round(s / 10.0))
        fns = [lambda t: t < 40.0, lambda t: t >= 40.0]
        return _frame_probs(s, e, fns[::-1] if idx % 2 else fns)     # local order flips per window

    def embed_fn(spans):
        return np.array([(vec_a if (s + e) / 2.0 < 40.0 else vec_b) + rng.normal(0.0, 0.01, 3)
                         for s, e in spans])

    calls = []
    xtf, ptf = _identity_tf()
    fn = W.make_vbx_cluster_fn(_oracle_vbx(calls), xtf, ptf, np.ones(3), lda_dim=3, Fa=0.07, Fb=0.8,
                               max_iters=20, init_smoothing=7.0)
    res = W.run_piece(80.0, prob_fn, embed_fn, cluster_fn=fn)
    assert res.n_clusters == 2
    assert res.stats["cannot_link_violations"] == 0 and res.stats["vbx_fallback"] is False
    at_20 = {s["cluster"] for s in res.segments if s["start"] <= 20 <= s["end"]}
    at_60 = {s["cluster"] for s in res.segments if s["start"] <= 60 <= s["end"]}
    assert len(at_20) == 1 and len(at_60) == 1 and at_20 != at_60
    assert calls and calls[0]["Fa"] == 0.07 and calls[0]["Fb"] == 0.8 and calls[0]["maxIters"] == 20
    assert calls[0]["init_smoothing"] == 7.0 and calls[0]["phi_len"] == 3


def test_hungarian_reassign_removes_a_vbx_merge_inside_one_window():
    # VBx may put the two labels of window 0 into one cluster. The per-window Hungarian step must split them.
    prob_fn, embed_fn = _two_speaker_piece_fns()
    base = _oracle_vbx()

    def merging_vbx(ahc_init, fea, Phi, Fa, Fb, loopProb=0.0, maxIters=20, init_smoothing=7.0):
        g, pi = base(ahc_init, fea, Phi, Fa, Fb)
        g = g.copy()
        g[1] = g[0]          # items 0 and 1 are window 0's two labels (sorted by window, then label)
        return g, pi

    xtf, ptf = _identity_tf()
    fn = W.make_vbx_cluster_fn(merging_vbx, xtf, ptf, np.ones(3), lda_dim=3)
    res = W.run_piece(60.0, prob_fn, embed_fn, cluster_fn=fn)
    assert res.stats["cannot_link_violations"] == 0
    assert res.n_clusters == 2
    w0 = res.windows[0]
    assert w0.cluster_of[0] >= 0 and w0.cluster_of[1] >= 0 and w0.cluster_of[0] != w0.cluster_of[1]


def test_vbx_falls_back_to_constrained_ahc_above_max_k():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(9, 4))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    win_of = np.arange(9)       # one label per window: with 3 labels per window cannot-link would force 3 clusters

    def three_speakers(ahc_init, fea, Phi, Fa, Fb, loopProb=0.0, maxIters=20, init_smoothing=7.0):
        return np.eye(3)[np.arange(len(fea)) % 3], np.ones(3) / 3

    xtf, ptf = _identity_tf()
    fn = W.make_vbx_cluster_fn(three_speakers, xtf, ptf, np.ones(4), lda_dim=4, ahc_threshold=0.6)
    labels = fn(x, win_of, 1, 2)
    assert fn.last_info == {"vbx_fallback": True}
    expected = W.constrained_ahc(x, win_of, 0.6 ** 2 / 2.0, 1, 2)     # euclid 0.6 -> cosine 0.18
    assert np.array_equal(labels, expected)
    assert len(set(labels.tolist())) <= 2


def test_vbx_with_fewer_than_two_embeddings_puts_everything_in_cluster_zero():
    calls = []
    xtf, ptf = _identity_tf()
    fn = W.make_vbx_cluster_fn(_oracle_vbx(calls), xtf, ptf, np.ones(3), lda_dim=3)
    assert fn(np.array([[1.0, 0.0, 0.0]]), np.array([0]), 1, 8).tolist() == [0]
    assert fn(np.zeros((0, 3)), np.zeros(0, dtype=int), 1, 8).size == 0
    assert calls == [] and fn.last_info == {"vbx_fallback": False}


def test_save_and_load_piece_probs_round_trip(tmp_path):
    piece = W.Piece(core_start=0.0, core_end=240.0, audio_start=0.0, audio_end=270.0)
    spans = W.window_spans(piece.audio_duration)
    rng = np.random.default_rng(5)
    probs = [rng.random((int(round(30.0 / 0.08)), n)) for n in (2, 0, 3)] + \
            [rng.random((375, 1)) for _ in range(len(spans) - 3)]
    path = tmp_path / "piece_000.npz"
    W.save_piece_probs(str(path), piece, spans, probs, 0.08)
    got_piece, got_spans, got_probs, got_fs = W.load_piece_probs(str(path))
    assert got_piece == piece
    assert got_spans == spans and got_fs == pytest.approx(0.08)
    assert len(got_probs) == len(probs)
    for a, b in zip(got_probs, probs):
        assert a.shape == b.shape and np.array_equal(a, b)
    with pytest.raises(ValueError):
        W.save_piece_probs(str(tmp_path / "bad.npz"), piece, spans, probs[:-1], 0.08)


def test_run_file_accepts_planned_pieces_and_skips_planning():
    def boom(*args, **kwargs):
        raise AssertionError("planning must be skipped when pieces are given")

    prob_fn, embed_fn = _two_speaker_piece_fns()
    xtf, ptf = _identity_tf()
    fn = W.make_vbx_cluster_fn(_oracle_vbx(), xtf, ptf, np.ones(3), lda_dim=3)
    pieces = [W.Piece(0.0, 60.0, 0.0, 60.0)]
    out = W.run_file(60.0, None, make_prob_fn=lambda p: prob_fn, make_embed_fn=lambda p: embed_fn,
                     vad_fn=boom, pieces=pieces, frame_s=0.08, cluster_fn=fn)
    assert len(out["pieces"]) == 1
    assert {s["speaker"] for s in out["segments"]} == {"SPEAKER_00", "SPEAKER_01"}


def test_reference_vbx_separates_two_speakers_through_make_vbx_cluster_fn():
    """Uses the DiariZen VBx.py copied to the scratchpad (outside the repo), so it runs only on that machine.

    Same formulas as diarizen.clustering.VBx (cluster_vbx, Apache-2.0, numpy/scipy only). The toy embeddings
    have the scale vbx_setup gives them (xvec_tf multiplies unit vectors by sqrt(lda_dim)); with unit vectors
    VBx with Fa 0.07 stays at responsibilities of about 0.5 and merges the speakers.
    """
    import importlib.util
    if not os.path.isfile(REF_VBX):
        pytest.skip("reference VBx.py not present on this machine")
    spec = importlib.util.spec_from_file_location("vbx_reference_copy", REF_VBX)
    vbx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vbx)

    dim, n = 48, 60
    rng = np.random.default_rng(1)
    ca = rng.normal(size=dim)
    ca /= np.linalg.norm(ca)
    speakers = np.arange(n) % 2
    X = []
    for s in speakers:
        v = (ca if s == 0 else -ca) + rng.normal(0.0, 0.05, dim)
        X.append(v / np.linalg.norm(v))
    X = np.array(X)
    xvec_tf = lambda x: np.sqrt(dim) * x                      # noqa: E731  as vbx_setup's xvec_tf scales
    plda_tf = lambda x0, lda_dim=dim: x0[:, :lda_dim]         # noqa: E731
    fn = W.make_vbx_cluster_fn(vbx.cluster_vbx, xvec_tf, plda_tf, np.ones(dim), lda_dim=32)
    labels = fn(X, np.arange(n) // 2, 1, 8)
    assert fn.last_info == {"vbx_fallback": False}
    assert len(set(labels.tolist())) == 2
    assert len(set(zip(labels.tolist(), speakers.tolist()))) == 2     # each label is one speaker


def test_plda_folder_is_the_newest_complete_snapshot(tmp_path):
    import importlib.util
    import time
    spec = importlib.util.spec_from_file_location(
        "windowed_vbx_stage_under_test", os.path.join(os.path.dirname(__file__), "..", "tools",
                                                      "windowed_vbx_stage.py"))
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    base = tmp_path / "models--BUT-FIT--diarizen-wavlm-large-s80-md-v2" / "snapshots"
    old, new, broken = (base / h / "plda" for h in ("aaa", "bbb", "ccc"))
    for d in (old, new):
        d.mkdir(parents=True)
        (d / "xvec_transform.npz").write_bytes(b"")
        (d / "plda.npz").write_bytes(b"")
    broken.mkdir(parents=True)
    (broken / "plda.npz").write_bytes(b"")
    os.utime(old, (time.time() - 100, time.time() - 100))
    assert stage.plda_candidates(str(tmp_path)) == [str(new), str(old)]     # broken snapshot is left out


def test_vbx_cluster_fn_gets_raw_embeddings_not_unit_ones():
    # The PLDA transform subtracts a mean learned on raw vectors, so run_piece must hand it the raw
    # embeddings (norm kept), while AHC and the Hungarian step use the unit ones.
    seen = {}

    def fake_vbx(ahc_init, fea, phi, Fa, Fb, maxIters, init_smoothing):
        q = np.zeros((len(fea), 2))
        q[np.arange(len(fea)), np.asarray(ahc_init) % 2] = 1.0
        return q, np.array([0.5, 0.5])

    def xvec_tf(x):
        seen["x"] = np.array(x)
        return x

    fn = W.make_vbx_cluster_fn(fake_vbx, xvec_tf, lambda x, lda_dim: x[:, :lda_dim], np.ones(8), lda_dim=4)
    raw = np.array([[10.0, 0, 0, 0], [0, 10.0, 0, 0], [10.0, 0.5, 0, 0], [0, 10.0, 0.5, 0]])
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    fn(unit, np.array([0, 0, 1, 1]), 1, 8, raw=raw)
    assert np.allclose(seen["x"], raw)
    fn(unit, np.array([0, 0, 1, 1]), 1, 8)           # no raw given: falls back to the unit vectors
    assert np.allclose(seen["x"], unit)


def test_run_piece_passes_raw_only_to_a_cluster_fn_that_wants_it():
    calls = []

    def prob_fn(s, e):
        n = int(round((e - s) / FS))
        t = s + (np.arange(n) + 0.5) * FS
        return np.stack([(t < 12).astype(float) * 0.9, (t >= 12).astype(float) * 0.9], axis=1)

    def embed_fn(spans):
        return np.array([[5.0, 0.0] if t0 < 12 else [0.0, 5.0] for t0, _ in spans])

    def plain(x, win_of, min_k, max_k):
        calls.append(("plain", None))
        return np.zeros(len(x), dtype=int)

    def wants(x, win_of, min_k, max_k, raw=None):
        calls.append(("wants", raw))
        return np.zeros(len(x), dtype=int)
    wants.wants_raw = True

    W.run_piece(30.0, prob_fn, embed_fn, cluster_fn=plain)
    W.run_piece(30.0, prob_fn, embed_fn, cluster_fn=wants)
    assert calls[0] == ("plain", None)
    assert calls[1][0] == "wants" and calls[1][1] is not None
    assert np.allclose(np.linalg.norm(calls[1][1], axis=1), 5.0)     # raw norm kept


def test_max_embed_s_caps_the_clean_seconds_embedded_per_window_label():
    spans_seen = []

    def prob_fn(s, e):
        n = int(round((e - s) / FS))
        return np.full((n, 1), 0.9)             # one label, clean speech through the whole window

    def embed_fn(spans):
        spans_seen.append(list(spans))
        return np.tile([[3.0, 0.0]], (len(spans), 1))

    W.run_piece(30.0, prob_fn, embed_fn, max_embed_s=6.0)
    per_window = len(spans_seen[0])
    assert sum(t1 - t0 for t0, t1 in spans_seen[0]) <= 6.0 + 3.0 + 1e-6 and per_window >= 2
    spans_seen.clear()
    W.run_piece(30.0, prob_fn, embed_fn, max_embed_s=0.0)       # 0 = no cap
    assert len(spans_seen[0]) > per_window
