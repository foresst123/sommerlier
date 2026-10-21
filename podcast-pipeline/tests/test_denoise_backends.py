import sys
import textwrap

import numpy as np
import pytest
from scipy.signal import firwin

from models import denoise
from models.denoise import (available_backends, chain_id, get_backend, parse_chain)
from models.denoise.base import DenoiseBackend, DenoiseError
from models.denoise.chunking import (chunk_apply, chunk_spans, compensate_delay, estimate_delay,
                                     fit_length, resample_exact, resample_to, resampled_length)


def _speechlike(n, sr, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    noise = np.convolve(rng.standard_normal(n + 200), firwin(201, 3000, fs=sr), mode="valid")  # < 3 kHz: survives 16 kHz round trips
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t)) + 0.2 * noise
    return x.astype(np.float32)


# ---------------------------------------------------------------- chain parsing

@pytest.mark.parametrize("spec", [None, "", "none", "None", "  ", " NONE "])
def test_parse_chain_none(spec):
    assert parse_chain(spec) == []
    assert chain_id(spec) == "none"


def test_parse_chain_values():
    assert parse_chain("dfn3") == ["dfn3"]
    assert parse_chain("dfn3>mf2se48") == ["dfn3", "mf2se48"]
    assert parse_chain(" DFN3 > mpsenet ") == ["dfn3", "mpsenet"]
    assert parse_chain(["a", "b"]) == ["a", "b"]
    assert chain_id("dfn3") == "dfn3"
    assert chain_id(["dfn3", "mf2se48"]) == "dfn3+mf2se48"
    assert chain_id("dfn3>mf2se48") == "dfn3+mf2se48"


@pytest.mark.parametrize("spec", ["a>>b", ">a", "a>", ">", "none>a", "a>none", "a b", "a/b", "a>b>>"])
def test_parse_chain_errors(spec):
    with pytest.raises(ValueError):
        parse_chain(spec)
    with pytest.raises(ValueError):
        chain_id(spec)


def test_chain_id_is_filename_safe():
    cid = chain_id("dfn3>mf2se48>mpsenet")
    assert cid == "dfn3+mf2se48+mpsenet"
    assert all(c.isalnum() or c in "+_" for c in cid)


# --------------------------------------------------------------------- registry

def test_get_backend_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_backend("nope")


def test_get_backend_names_and_cfg_merge():
    b = get_backend("IDENTITY")
    assert b.name == "identity"
    b = get_backend("fake_hp", {"chunk_s": 5, "overlap_s": 1, "fake_hp": {"chunk_s": 8}})
    assert (b.chunk_s, b.overlap_s) == (8.0, 1.0)
    for name in ("identity", "fake_hp", "dfn3", "mf2se48", "mpsenet"):
        assert get_backend(name).name == name


def test_available_backends_never_raises_and_reports_missing(monkeypatch):
    res = available_backends()
    assert set(res) == {"identity", "fake_hp", "dfn3", "mf2se48", "mpsenet"}
    assert res["identity"] == (True, "") and res["fake_hp"][0] is True
    for ok, reason in res.values():
        assert isinstance(ok, bool) and isinstance(reason, str)
        assert ok or reason
    # a broken check is reported, not raised
    monkeypatch.setattr(denoise.BACKENDS["dfn3"], "check_available", classmethod(lambda cls, cfg=None: 1 / 0))
    assert available_backends()["dfn3"][0] is False


def test_available_backends_missing_deps(monkeypatch):
    import models.denoise.dfn3 as dfn3
    import models.denoise.mf2se48 as mf
    import models.denoise.mpsenet as mp
    for mod in (dfn3, mf, mp):
        monkeypatch.setattr(mod, "missing_modules", lambda *names: list(names))
    res = available_backends()
    assert all(not res[k][0] and "missing python package" in res[k][1] for k in ("dfn3", "mf2se48", "mpsenet"))
    with pytest.raises(DenoiseError):
        get_backend("dfn3").load()


def test_weights_missing_reported(monkeypatch, tmp_path):
    import models.denoise.dfn3 as dfn3
    import models.denoise.mf2se48 as mf
    import models.denoise.mpsenet as mp
    for mod in (dfn3, mf, mp):
        monkeypatch.setattr(mod, "missing_modules", lambda *names: [])
    monkeypatch.setattr(dfn3, "has_module", lambda n: False)
    monkeypatch.setenv("DFN_MODEL_DIR", str(tmp_path / "x"))
    monkeypatch.setenv("MF2SE48_DIR", str(tmp_path / "y"))
    monkeypatch.delenv("MPSENET_DIR", raising=False)
    res = available_backends()
    assert not res["dfn3"][0] and "weights not found" in res["dfn3"][1]
    assert not res["mf2se48"][0] and "weights not found" in res["mf2se48"][1]
    assert not res["mpsenet"][0] and "MPSENET_DIR" in res["mpsenet"][1]
    monkeypatch.setenv("MPSENET_DIR", str(tmp_path))
    assert "does not look like" in available_backends()["mpsenet"][1]
    assert available_backends({"dfn3": {"auto_download": True}})["dfn3"][0] is True


def test_import_is_light():
    # models.denoise must import without pulling in the heavy model packages
    for name in ("df", "clearvoice"):
        assert name not in sys.modules


def test_mf2se48_rejects_long_chunks():
    with pytest.raises(ValueError):
        get_backend("mf2se48", {"chunk_s": 30})


# ----------------------------------------------------------- identity / fake_hp

def test_identity_returns_equal_arrays():
    x = _speechlike(24000, 24000)
    y = get_backend("identity").enhance(x, 24000)
    assert y.dtype == np.float32 and y.shape == x.shape
    np.testing.assert_array_equal(y, x)


def test_input_not_mutated_and_output_independent():
    x = _speechlike(12000, 24000)
    before = x.copy()
    for name in ("identity", "fake_hp"):
        y = get_backend(name).enhance(x, 24000)
        np.testing.assert_array_equal(x, before)
        assert not np.shares_memory(x, y)
        y[:] = 0
        np.testing.assert_array_equal(x, before)
    ro = x.copy()
    ro.setflags(write=False)
    get_backend("fake_hp").enhance(ro, 24000)  # read-only input is fine


def test_fake_hp_attenuates_low_frequencies_deterministically():
    sr = 24000
    t = np.arange(sr * 2) / sr
    low = np.sin(2 * np.pi * 50 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 2000 * t).astype(np.float32)
    b = get_backend("fake_hp", {"cutoff_hz": 300})
    lo_out, hi_out = b.enhance(low, sr), b.enhance(high, sr)
    mid = slice(sr // 2, -sr // 2)
    rms = lambda a: float(np.sqrt(np.mean(a[mid] ** 2)))  # noqa: E731
    assert rms(lo_out) < 0.05 * rms(low)
    assert rms(hi_out) > 0.95 * rms(high)
    np.testing.assert_array_equal(b.enhance(low, sr), lo_out)
    assert lo_out.shape == low.shape and lo_out.dtype == np.float32


@pytest.mark.parametrize("n", [1, 2, 5, 17, 100])
def test_fake_hp_keeps_length_on_tiny_inputs(n):
    x = _speechlike(n, 24000)
    y = get_backend("fake_hp").enhance(x, 24000)
    assert y.shape == (n,) and np.all(np.isfinite(y))


def test_enhance_empty_and_bad_inputs():
    b = get_backend("identity")
    assert b.enhance(np.zeros(0, np.float32), 24000).shape == (0,)
    with pytest.raises(ValueError):
        b.enhance(np.zeros((2, 10), np.float32), 24000)
    with pytest.raises(ValueError):
        b.enhance(np.zeros(10, np.int16), 24000)
    with pytest.raises(ValueError):
        b.enhance(np.array([0.0, np.nan], np.float32), 24000)
    with pytest.raises(ValueError):
        get_backend("fake_hp", {"cutoff_hz": 20000}).enhance(np.zeros(100, np.float32), 24000)


# ------------------------------------------------------------------- chunking

@pytest.mark.parametrize("chunk,overlap", [(1000, 0), (1000, 100), (1000, 250), (64, 16), (37, 9)])
def test_chunk_apply_identity_exact_around_boundaries(chunk, overlap):
    rng = np.random.default_rng(1)
    hop = chunk - overlap
    lengths = {0, 1, 2, overlap, chunk - 1, chunk, chunk + 1, chunk + overlap, 2 * chunk - overlap,
               2 * chunk - overlap + 1, 2 * chunk, 3 * hop + overlap, 3 * hop + overlap + 1, 5 * chunk + 7}
    for n in sorted(lengths):
        x = rng.standard_normal(n).astype(np.float32)
        spans = chunk_spans(n, chunk, overlap) if n else []
        assert all(e - s <= chunk for s, e in spans)
        if spans:
            assert spans[0][0] == 0 and spans[-1][1] == n
            assert all(spans[i + 1][0] <= spans[i][1] for i in range(len(spans) - 1))  # no holes
        y = chunk_apply(x, lambda c: c.copy(), chunk, overlap)
        assert y.shape == x.shape
        np.testing.assert_allclose(y, x, atol=1e-6, rtol=0)


def test_chunk_apply_fn_sees_bounded_chunks_and_wrong_length_is_fixed():
    seen = []

    def fn(c):
        seen.append(len(c))
        return c[:-1]  # model that loses a sample

    x = np.ones(1000, np.float32)
    y = chunk_apply(x, fn, 300, 50)
    assert y.shape == x.shape and max(seen) <= 300 and len(seen) > 3


def test_chunk_apply_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_apply(np.zeros(10, np.float32), lambda c: c, 100, 60)
    with pytest.raises(ValueError):
        chunk_apply(np.zeros(10, np.float32), lambda c: c, 0, 0)


def test_chunk_apply_long_input_matches_unchunked_linear_fn():
    x = _speechlike(100003, 16000)
    y = chunk_apply(x, lambda c: 0.5 * c, 16000, 2000)
    np.testing.assert_allclose(y, 0.5 * x, atol=1e-6)


# ------------------------------------------------------------------ resampling

@pytest.mark.parametrize("n", [1, 7, 480, 24000, 24001, 99991])
@pytest.mark.parametrize("native", [48000, 16000])
def test_resample_round_trip_restores_length_exactly(n, native):
    x = _speechlike(n, 24000)
    up = resample_to(x, 24000, native)
    assert len(up) == resampled_length(n, 24000, native)
    back = resample_exact(up, native, 24000, n)
    assert back.shape == (n,) and back.dtype == np.float32


@pytest.mark.parametrize("native", [48000, 16000])
def test_resample_round_trip_is_close_for_in_band_signal(native):
    sr = 24000
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    back = resample_exact(resample_to(x, sr, native), native, sr, len(x))
    np.testing.assert_allclose(back[2000:-2000], x[2000:-2000], atol=5e-3)


def test_resample_same_rate_copies():
    x = _speechlike(100, 24000)
    y = resample_to(x, 24000, 24000)
    np.testing.assert_array_equal(x, y)
    assert not np.shares_memory(x, y)
    with pytest.raises(ValueError):
        resample_to(x, 0, 16000)


# --------------------------------------------------------------- delay handling

@pytest.mark.parametrize("delay", [0, 1, 37, 160, -23])
def test_delay_compensation_recovers_fixed_delay(delay):
    sr = 16000
    ref = _speechlike(sr * 6, sr, seed=3)
    out = np.zeros_like(ref)
    if delay >= 0:
        out[delay:] = ref[:len(ref) - delay]
    else:
        out[:delay] = ref[-delay:]
    out = out * 0.7 + 0.001 * np.random.default_rng(5).standard_normal(len(ref)).astype(np.float32)
    lag, score = estimate_delay(ref, out, max_lag=400, probe=sr)
    assert lag == delay and score > 0.9
    fixed = compensate_delay(out, lag)
    assert fixed.shape == out.shape
    keep = slice(abs(delay) + 1, len(ref) - abs(delay) - 1)
    assert np.corrcoef(fixed[keep], ref[keep])[0, 1] > 0.99


def test_estimate_delay_degenerate_inputs():
    assert estimate_delay(np.zeros(1000, np.float32), np.zeros(1000, np.float32), 50) == (0, 0.0)
    assert estimate_delay(np.ones(10, np.float32), np.ones(10, np.float32), 50) == (0, 0.0)
    x = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(compensate_delay(x, 0), x)
    assert compensate_delay(x, 100).shape == x.shape


class _Native16k(DenoiseBackend):
    """Fake 16 kHz model that is a pure delay, optionally NaN-ing or changing length."""
    name = "fake16k"
    native_sr = 16000
    default_chunk_s = 0.5
    default_overlap_s = 0.1

    def __init__(self, cfg=None, logger=None, lag=0, mode="ok"):
        super().__init__(cfg, logger)
        self.lag, self.mode, self.calls = lag, mode, 0

    def _enhance_native(self, x):
        def run(c):
            self.calls += 1
            if self.mode == "nan":
                return np.full_like(c, np.nan)
            if self.mode == "short":
                return c[:-3]
            return compensate_delay(c, -self.lag) if self.lag else c.copy()
        return self._chunked(run, x) if self.mode != "delay_whole" else compensate_delay(x, -self.lag)


@pytest.mark.parametrize("sr", [24000, 48000, 16000, 22050])
@pytest.mark.parametrize("n_sec", [0.0003, 0.2, 1.7])
def test_enhance_end_to_end_identity_model_keeps_length_and_content(sr, n_sec):
    n = max(1, int(sr * n_sec))
    x = _speechlike(n, sr)
    b = _Native16k()
    y = b.enhance(x, sr)
    assert y.shape == x.shape and y.dtype == np.float32
    if n > 2000:
        np.testing.assert_allclose(y[500:-500], x[500:-500], atol=5e-3)
    assert b.loaded


def test_enhance_removes_model_delay_when_auto():
    sr = 24000
    x = _speechlike(sr * 3, sr, seed=9)
    delayed = _Native16k({"delay_compensation": "auto"}, lag=40, mode="delay_whole")
    y = delayed.enhance(x, sr)
    assert np.corrcoef(y[2000:-2000], x[2000:-2000])[0, 1] > 0.99
    plain = _Native16k({"delay_compensation": False}, lag=40, mode="delay_whole")
    z = plain.enhance(x, sr)
    assert np.corrcoef(z[2000:-2000], x[2000:-2000])[0, 1] < 0.9
    fixed = _Native16k({"delay_compensation": 40}, lag=40, mode="delay_whole")
    assert np.corrcoef(fixed.enhance(x, sr)[2000:-2000], x[2000:-2000])[0, 1] > 0.99


def test_non_finite_model_output_raises_clear_error():
    x = _speechlike(24000, 24000)
    with pytest.raises(DenoiseError, match="non-finite"):
        _Native16k(mode="nan").enhance(x, 24000)


def test_model_output_with_wrong_length_is_fixed():
    x = _speechlike(24000, 24000)
    assert _Native16k(mode="short").enhance(x, 24000).shape == x.shape


def test_enhance_does_not_mutate_input_with_resampling():
    x = _speechlike(24000, 24000)
    before = x.copy()
    _Native16k().enhance(x, 24000)
    np.testing.assert_array_equal(x, before)


def test_enhance_is_deterministic():
    x = _speechlike(36000, 24000)
    a, b = _Native16k().enhance(x, 24000), _Native16k().enhance(x, 24000)
    np.testing.assert_array_equal(a, b)


def test_load_unload_lifecycle():
    class B(_Native16k):
        loads = unloads = 0

        def _load(self):
            B.loads += 1

        def _unload(self):
            B.unloads += 1

    b = B()
    b.load(); b.load()
    assert B.loads == 1
    b.unload(); b.unload()
    assert B.unloads == 1 and not b.loaded
    with B() as c:
        assert c.loaded
    assert not c.loaded


# ---------------------------------------------- MP-SENet wrapper with a fake repo

_FAKE_MODEL = textwrap.dedent("""
    import torch
    import torch.nn as nn
    from models.transformer import Tag
    from utils import LearnableSigmoid2d

    class MPNet(nn.Module):
        def __init__(self, h, num_tsblocks=4):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1))
            self.tag = Tag
            self.h = h

        def forward(self, amp, pha):
            return amp * self.w, pha, None
""")


def _make_fake_mpsenet(tmp_path):
    torch = pytest.importorskip("torch")
    repo = tmp_path / "MP-SENet"
    (repo / "models").mkdir(parents=True)
    (repo / "best_ckpt").mkdir()
    (repo / "models" / "transformer.py").write_text("Tag = 'fake'\n")
    (repo / "models" / "model.py").write_text(_FAKE_MODEL)
    (repo / "utils.py").write_text("class LearnableSigmoid2d: pass\nimport matplotlib\nmatplotlib.use('Agg')\n")
    (repo / "best_ckpt" / "config.json").write_text(
        '{"sampling_rate": 16000, "n_fft": 400, "hop_size": 100, "win_size": 400, "compress_factor": 0.3}')

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))
    torch.save({"generator": M().state_dict()}, repo / "best_ckpt" / "g_best_vb")
    return repo


def test_mpsenet_import_isolation_restores_project_modules(tmp_path):
    from models.denoise.mpsenet import load_mpnet_class
    import models
    import utils
    before = (sys.modules["models"], sys.modules["utils"], list(sys.path))
    MPNet = load_mpnet_class(str(_make_fake_mpsenet(tmp_path)))
    assert MPNet.__name__ == "MPNet"
    assert (sys.modules["models"], sys.modules["utils"], list(sys.path)) == before
    assert models is sys.modules["models"] and utils is sys.modules["utils"]
    assert "models.model" not in sys.modules and "models.transformer" not in sys.modules


@pytest.mark.parametrize("sr,n", [(24000, 24000 * 3), (16000, 5000), (24000, 50)])
def test_mpsenet_wrapper_with_pass_through_model(tmp_path, monkeypatch, sr, n):
    pytest.importorskip("torch")
    import models.denoise.mpsenet as mp
    monkeypatch.setattr(mp, "missing_modules", lambda *names: [])
    repo = _make_fake_mpsenet(tmp_path)
    monkeypatch.setenv("MPSENET_DIR", str(repo))
    assert available_backends()["mpsenet"] == (True, "")
    b = get_backend("mpsenet", {"chunk_s": 1.0, "overlap_s": 0.1, "device": "cpu"})
    x = _speechlike(n, sr, seed=2)
    before = x.copy()
    y = b.enhance(x, sr)
    np.testing.assert_array_equal(x, before)
    assert y.shape == x.shape and y.dtype == np.float32 and np.all(np.isfinite(y))
    if n > 4000:  # an identity mask must reproduce the signal (STFT round trip + resampling)
        np.testing.assert_allclose(y[1000:-1000], x[1000:-1000], atol=0.02)
    # silent input stays silent instead of dividing by zero
    assert np.all(b.enhance(np.zeros(sr, np.float32), sr) == 0)
    b.unload()
