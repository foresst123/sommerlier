"""Embedder trials (cosine AHC, --clustering dz-ahc) with fakes only: the embedding cache and the stage's argument checks.

The real models were NOT run. tools/windowed_vbx_stage.py imports torch only inside main(), after its argument checks,
so the refusals below run without torch.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import windowed_vbx_stage as VS  # noqa: E402
from utils import dz_pipeline as DZ  # noqa: E402

FS = 0.08


def _two_speakers_alternating(counter):
    """Fake local segmentation (speakers swap every 6 s) and a fake embedder that counts its calls."""
    centres = np.eye(4)[:2]

    def prob_fn(s, e):
        n = int(round((e - s) / FS))
        t = s + (np.arange(n) + 0.5) * FS
        out = np.full((n, 2), 0.05)
        out[np.arange(n), np.floor(t / 6.0).astype(int) % 2] = 0.95
        return out

    def embed_fn(span, mask):
        counter["n"] += 1
        return centres[int(span[0] // 6) % 2] * 3.0

    return prob_fn, embed_fn


def test_second_cos_threshold_reads_the_cache_and_calls_no_embedder(tmp_path):
    counter = {"n": 0}
    prob_fn, embed_fn = _two_speakers_alternating(counter)
    path = VS.embedding_cache_path(str(tmp_path / "embcache"), "fake-embedder", {"thr": 0.5})

    cache = VS.EmbeddingCache(path).load()                 # no file yet: an empty table
    first = DZ.run_piece_dz(30.0, prob_fn, cache.wrap(0.0, embed_fn), vbx_fn=DZ.make_dz_ahc_fn(0.3),
                            window_s=16.0, hop_s=0.8, frame_s=FS)
    assert first.n_clusters == 2 and counter["n"] > 0
    assert cache.misses == counter["n"] and cache.hits == 0
    assert cache.save() is True and os.path.isfile(path)
    made = counter["n"]

    again = VS.EmbeddingCache(path).load()                 # what a second run with another --cos-threshold does
    assert len(again.table) == made
    second = DZ.run_piece_dz(30.0, prob_fn, again.wrap(0.0, embed_fn), vbx_fn=DZ.make_dz_ahc_fn(1.5),
                             window_s=16.0, hop_s=0.8, frame_s=FS)
    assert counter["n"] == made                            # no model call on the second run
    assert again.hits == made and again.misses == 0
    assert again.save() is False                           # nothing new, so the file is not rewritten
    # the other threshold reached the clustering (one AHC cluster), but DiariZen's Hungarian step then leaves
    # slots unassigned and np.unique shifts them to id 0, so the output still has two ids: not a merge
    assert second.stats["ahc_init_clusters"] == 1 and second.stats["active_unassigned"] > 0


def test_cached_none_is_a_result_not_a_miss(tmp_path):
    path = str(tmp_path / "c.npz")
    cache = VS.EmbeddingCache(path).load()
    calls = {"n": 0}

    def refuse(span, mask):
        calls["n"] += 1
        return None

    wrapped = cache.wrap(1.0, refuse)
    assert wrapped((0.0, 16.0), np.ones(200)) is None and wrapped((0.0, 16.0), np.ones(200)) is None
    assert calls["n"] == 1 and cache.hits == 1
    assert cache.save() is True
    reread = VS.EmbeddingCache(path).load()
    assert reread.table and list(reread.table.values()) == [None]


def test_cache_file_name_depends_on_the_embedder_and_its_inputs_not_on_cos(tmp_path):
    base = VS.embedding_cache_path(str(tmp_path), "r34lm-x", {"thr": 0.5})
    assert base == VS.embedding_cache_path(str(tmp_path), "r34lm-x", {"thr": 0.5})
    assert base != VS.embedding_cache_path(str(tmp_path), "espnet_spk-y", {"thr": 0.5})
    assert base != VS.embedding_cache_path(str(tmp_path), "r34lm-x", {"thr": 0.45})
    assert os.path.dirname(base) == str(tmp_path) and os.path.basename(base).startswith("r34lm-x_")
    assert base.endswith(".npz")


@pytest.mark.parametrize("extra", [
    ["--cos-threshold", "0"],
    ["--cos-threshold", "2"],
    ["--cos-threshold", "nan"],
    ["--embedder", "pyannote"],
    ["--embedder-kind", "wespeaker_simam100"],                     # needs --simam-onnx or --simam-dir
])
def test_dz_ahc_argument_errors_stop_before_torch_is_needed(tmp_path, extra):
    argv = ["--audio", str(tmp_path / "a.wav"), "--probs-dir", str(tmp_path), "--out", str(tmp_path / "o.json"),
            "--clustering", "dz-ahc", *extra]
    with pytest.raises(SystemExit):
        VS.main(argv)
    assert not (tmp_path / "o.json").exists()


def test_embedder_kind_is_refused_for_the_other_clusterings(tmp_path):
    argv = ["--audio", str(tmp_path / "a.wav"), "--probs-dir", str(tmp_path), "--out", str(tmp_path / "o.json"),
            "--clustering", "dz", "--embedder-kind", "r34lm"]
    with pytest.raises(SystemExit):
        VS.main(argv)


class _CountingAdapter:
    """Stands in for an embedder kind: the two speakers (DC levels 0.1 and 0.5) get orthogonal vectors."""

    id_string = "fake-embedder-abc"

    def __init__(self, counter):
        self.counter = counter

    def embed(self, wave):
        self.counter["n"] += 1
        return np.array([1.0, 0.0]) if float(np.mean(wave)) < 0.3 else np.array([0.0, 1.0])


def _probs_dir(tmp_path, duration=30.0):
    """A plan.json and one piece_000.npz as windowed_nemotron_run.py --dump-probs writes them, two speakers."""
    from utils import windowed_clustering as wc
    probs = tmp_path / "probs"
    probs.mkdir()
    spans = wc.window_spans(duration, 16.0, 0.8)
    probs_list = []
    for s, e in spans:
        n = int(round((e - s) / FS))
        t = s + (np.arange(n) + 0.5) * FS
        out = np.full((n, 2), 0.05)
        out[np.arange(n), np.floor(t / 6.0).astype(int) % 2] = 0.95
        probs_list.append(out)
    piece = wc.Piece(0.0, duration, 0.0, duration)
    wc.save_piece_probs(str(probs / "piece_000.npz"), piece, spans, probs_list, FS)
    import json
    (probs / "plan.json").write_text(json.dumps({
        "duration": duration, "frame_s": FS, "speech_intervals": 1,
        "params": {"piece_s": 240.0, "window_s": 16.0, "hop_s": 0.8, "overlap_s": 30.0},
        "pieces": [{"file": "piece_000.npz", "core": [0.0, duration], "audio": [0.0, duration]}]}), encoding="utf-8")
    return probs


def _wave(tmp_path, duration=30.0):
    import soundfile as sf
    t = (np.arange(int(duration * 16000)) + 0.5) / 16000
    wave = np.where(np.floor(t / 6.0).astype(int) % 2 == 0, 0.1, 0.5).astype(np.float32)
    path = tmp_path / "chunk.wav"
    sf.write(str(path), wave, 16000, subtype="FLOAT")
    return path


def test_main_dz_ahc_second_cos_threshold_hits_the_cache_and_calls_no_embedder(tmp_path, monkeypatch):
    import json
    import types
    counter = {"n": 0}
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(VS.EMB, "make_embedder", lambda kind, **kw: _CountingAdapter(counter))
    audio, probs = _wave(tmp_path), _probs_dir(tmp_path)
    cache_dir = tmp_path / "embcache"

    def run(cos, out):
        argv = ["--audio", str(audio), "--probs-dir", str(probs), "--out", str(tmp_path / out),
                "--clustering", "dz-ahc", "--embedder-kind", "wespeaker_r293", "--r293-onnx", "unused.onnx",
                "--cos-threshold", str(cos), "--device", "cpu", "--embcache-dir", str(cache_dir)]
        assert VS.main(argv) == 0
        return json.loads((tmp_path / out).read_text(encoding="utf-8"))

    first = run(0.3, "a.json")
    made = counter["n"]
    assert made > 0 and first["params"]["embedder"] == "wespeaker_r293"
    assert first["params"]["embedder_id"] == "fake-embedder-abc"
    assert [p["clusters"] for p in first["pieces"]] == [2]          # two speakers, two clusters at cos 0.3
    assert first["params"]["cos_threshold"] == 0.3 and first["params"]["embedding_cache_written"] is True
    assert first["params"]["embedding_cache_misses"] == made and first["params"]["embedding_cache_hits"] == 0
    files = sorted(os.listdir(cache_dir))
    assert len(files) == 1 and files[0].startswith("fake-embedder-abc_") and files[0].endswith(".npz")

    second = run(0.6, "b.json")                              # another cos threshold, same inputs, same cache file
    assert counter["n"] == made                              # no embedder call
    assert second["params"]["embedding_cache_hits"] == made and second["params"]["embedding_cache_misses"] == 0
    assert second["params"]["cos_threshold"] == 0.6 and second["params"]["embedding_cache_written"] is False
    assert sorted(os.listdir(cache_dir)) == files            # still one file for this embedder and these inputs
    assert second["segments"] and {"start", "end", "speaker"} <= set(second["segments"][0])
