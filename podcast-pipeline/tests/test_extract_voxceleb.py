import os
import random
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from tools import extract_voxceleb_embeddings as X  # noqa: E402


def write_wav(path, seconds, freq=220.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(int(16000 * seconds)) / 16000
    pcm = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(pcm.tobytes())


class FakeAdapter:
    """Vector = (mean, std, length) of the wave. Refuses a clip whose length is exactly 2.0 s, as a real model might."""

    def __init__(self):
        self.calls = 0

    def load(self):
        return self

    def embed(self, wave_):
        self.calls += 1
        if len(wave_) == 32000:
            raise ValueError("refused")
        return np.array([float(wave_.mean()), float(wave_.std()), float(len(wave_))])


@pytest.fixture
def voxroot(tmp_path):
    root = tmp_path / "aac"
    # id00001: two videos, four clips of 1 s, one clip of 0.2 s (too short)
    write_wav(str(root / "id00001" / "vid1" / "a.wav"), 1.0)
    write_wav(str(root / "id00001" / "vid1" / "b.wav"), 1.0)
    write_wav(str(root / "id00001" / "vid2" / "c.wav"), 1.0)
    write_wav(str(root / "id00001" / "vid2" / "short.wav"), 0.2)
    # id00002: one clip refused by the adapter (2.0 s)
    write_wav(str(root / "id00002" / "vid3" / "ok.wav"), 1.0)
    write_wav(str(root / "id00002" / "vid3" / "refused.wav"), 2.0)
    return root


def test_list_speakers_walks_nested_videos(voxroot):
    spk = X.list_speakers(str(voxroot), "wav")
    assert sorted(spk) == ["id00001", "id00002"]
    assert len(spk["id00001"]) == 4
    assert X.list_speakers(str(voxroot), "m4a") == {}


def test_pick_utterances_spreads_over_videos_and_is_seeded():
    paths = [f"/s/v{v}/u{u}.m4a" for v in range(3) for u in range(4)]
    first = X.pick_utterances(paths, 3, random.Random("0:s"))
    again = X.pick_utterances(paths, 3, random.Random("0:s"))
    assert first == again
    assert len(first) == 3
    assert len({os.path.dirname(p) for p in first}) == 3          # one from each video before a second
    assert X.pick_utterances(paths, 0, random.Random(0)) == paths  # 0 keeps all
    assert len(X.pick_utterances(paths, 100, random.Random(0))) == len(paths)


def test_embed_paths_counts_short_and_refused(voxroot):
    paths = [str(voxroot / "id00001" / "vid1" / "a.wav"), str(voxroot / "id00001" / "vid2" / "short.wav"),
             str(voxroot / "id00002" / "vid3" / "refused.wav")]
    vecs, stats = X.embed_paths(paths, FakeAdapter(), min_s=0.5)
    assert stats == {"offered": 3, "used": 1, "too_short": 1, "refused": 1}
    assert vecs.shape == (1, 3)


def test_extract_writes_chunks_resumes_and_merges(voxroot, tmp_path):
    out = str(tmp_path / "out")
    adapter = FakeAdapter()
    totals = X.extract(str(voxroot), out, adapter, ext="wav", per_speaker=0, min_s=0.5, seed=0, log=lambda s: None)
    assert totals["done_now"] == 2 and totals["done_before"] == 0
    assert totals["embeddings"] == 3 + 1          # id00001: a, b, c (short dropped); id00002: ok (refused dropped)
    assert os.path.exists(os.path.join(out, "chunks", "id00001.npz"))
    with open(os.path.join(out, "progress.jsonl"), encoding="utf-8") as fh:
        assert sum(1 for _ in fh) == 2

    calls_before = adapter.calls
    again = X.extract(str(voxroot), out, adapter, ext="wav", per_speaker=0, min_s=0.5, seed=0, log=lambda s: None)
    assert again["done_before"] == 2 and again["done_now"] == 0
    assert adapter.calls == calls_before          # resumed: nothing re-embedded

    meta = X.merge(out, "r34lm")
    assert meta["n_speakers"] == 2 and meta["n_embeddings"] == 4 and meta["embedder_dim"] == 3
    with np.load(os.path.join(out, "train.npz"), allow_pickle=False) as z:
        assert z["X"].shape == (4, 3)
        assert sorted(set(z["y"].tolist())) == ["id00001", "id00002"]


def test_merge_refuses_mixed_dimensions(tmp_path):
    out = tmp_path / "out"
    (out / "chunks").mkdir(parents=True)
    np.savez(out / "chunks" / "a.npz", X=np.zeros((2, 3)), y=np.array(["a", "a"]))
    np.savez(out / "chunks" / "b.npz", X=np.zeros((2, 4)), y=np.array(["b", "b"]))
    with pytest.raises(SystemExit):
        X.merge(str(out), "x")


def test_decode_returns_mono_float32_at_16k(voxroot):
    wave_ = X.decode(str(voxroot / "id00001" / "vid1" / "a.wav"))
    assert wave_.dtype == np.float32
    assert abs(len(wave_) - 16000) <= 1


def test_extract_limit_speakers(voxroot, tmp_path):
    out = str(tmp_path / "out2")
    totals = X.extract(str(voxroot), out, FakeAdapter(), ext="wav", per_speaker=0, min_s=0.5, seed=0,
                       limit_speakers=1, log=lambda s: None)
    assert totals["speakers"] == 1
    assert os.listdir(os.path.join(out, "chunks")) == ["id00001.npz"]


def test_parallel_decode_gives_the_same_embeddings(voxroot):
    paths = X.list_speakers(str(voxroot), "wav")["id00001"]
    serial, s_stats = X.embed_paths(paths, FakeAdapter(), min_s=0.5, workers=1)
    parallel, p_stats = X.embed_paths(paths, FakeAdapter(), min_s=0.5, workers=4)
    assert np.array_equal(serial, parallel)
    assert s_stats == p_stats


def test_shards_cover_each_speaker_exactly_once(voxroot, tmp_path):
    out = str(tmp_path / "sharded")
    done = []
    for k in range(3):
        t = X.extract(str(voxroot), out, FakeAdapter(), ext="wav", per_speaker=0, min_s=0.5, seed=0,
                      shard=(k, 3), log=lambda s: None)
        done.append(t["done_now"])
    assert sum(done) == 2                          # two speakers in the fixture
    assert sorted(os.listdir(os.path.join(out, "chunks"))) == ["id00001.npz", "id00002.npz"]


def test_parse_shard_rejects_bad_values():
    assert X.parse_shard("2/4") == (2, 4)
    for bad in ("4/4", "a/b", "1", "0/0", "-1/3"):
        with pytest.raises(ValueError):
            X.parse_shard(bad)


class CountingAdapter(FakeAdapter):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def embed(self, wave_):
        return super().embed(wave_) * self.scale


def test_multi_decodes_each_clip_once_and_writes_one_chunk_set_per_embedder(voxroot, tmp_path):
    decoded = []

    def counting_decode(path, sr=16000):
        decoded.append(path)
        return X.decode(path, sr)

    outs = {"a": str(tmp_path / "a"), "b": str(tmp_path / "b")}
    adapters = {"a": CountingAdapter(1.0), "b": CountingAdapter(2.0)}
    totals = X.extract_multi(str(voxroot), outs, adapters, ext="wav", per_speaker=0, min_s=0.5, seed=0,
                             decode_fn=counting_decode, log=lambda s: None)
    assert len(decoded) == len(set(decoded)) == 6            # 6 wav files, each decoded once for both
    assert totals["a"]["embeddings"] == totals["b"]["embeddings"] == 4
    for k in ("a", "b"):
        assert sorted(os.listdir(os.path.join(outs[k], "chunks"))) == ["id00001.npz", "id00002.npz"]
    with np.load(os.path.join(outs["b"], "chunks", "id00001.npz")) as za, \
            np.load(os.path.join(outs["a"], "chunks", "id00001.npz")) as zb:
        assert np.allclose(za["X"], 2.0 * zb["X"])


def test_multi_resumes_per_embedder(voxroot, tmp_path):
    outs = {"a": str(tmp_path / "ra"), "b": str(tmp_path / "rb")}
    X.extract_multi(str(voxroot), outs, {"a": CountingAdapter(1.0)}, ext="wav", per_speaker=0, min_s=0.5,
                    seed=0, log=lambda s: None)
    t = X.extract_multi(str(voxroot), outs, {"a": CountingAdapter(1.0), "b": CountingAdapter(1.0)},
                        ext="wav", per_speaker=0, min_s=0.5, seed=0, log=lambda s: None)
    assert t["a"]["done_before"] == 2 and t["a"]["done_now"] == 0
    assert t["b"]["done_before"] == 0 and t["b"]["done_now"] == 2


def test_parse_outs_one_or_many():
    assert X.parse_outs("/x", ["r34lm"]) == {"r34lm": "/x"}
    assert X.parse_outs("r34lm=/x,simam=/y", ["r34lm", "simam"]) == {"r34lm": "/x", "simam": "/y"}
    for bad, kinds in [("/x", ["r34lm", "simam"]), ("r34lm=/x", ["r34lm", "simam"]),
                       ("r34lm=/x,r34lm=/y", ["r34lm"]), ("r34lm=", ["r34lm"])]:
        with pytest.raises(ValueError):
            X.parse_outs(bad, kinds)
