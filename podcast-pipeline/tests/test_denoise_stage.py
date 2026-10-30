"""The optional noise-filter stage: no-op without a chain, aligned and cached with one."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from schemas.audio import AudioData
from services import denoise_service
from services.denoise_service import DenoiseService
from services.pipeline_service import PipelineService
from services.stage_output_service import StageOutputService
from utils.checkpoint import CheckpointManager

SR = 24000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeBackend:
    """Test backend; `behaviour` picks what enhance() does."""
    calls = []

    def __init__(self, name, behaviour):
        self.name, self.native_sr, self.behaviour = name, 16000, behaviour
        self.unloaded = False

    def load(self):
        pass

    def unload(self):
        self.unloaded = True

    def enhance(self, wav, sr):
        FakeBackend.calls.append(self.name)
        assert wav.dtype == np.float32 and wav.ndim == 1
        if self.behaviour == "short":
            return wav[:-10]
        if self.behaviour == "nan":
            out = wav.copy()
            out[5] = np.nan
            return out
        if self.behaviour == "silent":
            return np.zeros_like(wav)
        if self.behaviour == "boom":
            raise RuntimeError("out of memory")
        if self.behaviour == "half":
            return (wav * 0.5).astype(np.float32)
        if self.behaviour == "plus1":
            return (wav + 0.01).astype(np.float32)
        return wav.copy()


@pytest.fixture(autouse=True)
def fake_api(monkeypatch):
    """A stand-in for models.denoise, so these tests do not depend on it."""
    FakeBackend.calls = []
    made = []

    def get_backend(name, cfg=None, logger=None):
        if name not in ("identity", "half", "plus1", "short", "nan", "silent", "boom"):
            raise KeyError(name)
        backend = FakeBackend(name, name)
        made.append(backend)
        return backend

    def parse_chain(spec):
        if spec is None or str(spec).strip().lower() in ("", "none"):
            return []
        names = [t.strip() for t in str(spec).split(">")]
        if not all(names):
            raise ValueError(f"empty token in {spec!r}")
        return names

    def chain_id(spec):
        names = spec if isinstance(spec, list) else parse_chain(spec)
        return "+".join(names) or "none"

    monkeypatch.setattr(denoise_service, "_load_api",
                        lambda: (get_backend, parse_chain, chain_id))
    return made


def speech(seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    wav = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)
    return (wav + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def audio(wav=None):
    wav = speech() if wav is None else wav
    return AudioData(waveform=wav, sample_rate=SR, name="x.wav", audio_segment=None,
                     duration=len(wav) / SR)


def args(**kw):
    base = dict(denoise="none", music_scope="spans", job_id="j", cache_dir="c")
    base.update(kw)
    return SimpleNamespace(**base)


# --- the service -----------------------------------------------------------

def test_identity_chain_keeps_the_waveform_and_reports(fake_api):
    data = audio()
    before = data.waveform.copy()
    info = DenoiseService().apply(data, "identity", None, None)
    assert np.array_equal(data.waveform, before)
    assert info["errors"] == [] and info["chain"] == ["identity"]
    assert info["chain_id"] == "identity" and info["lag_samples"] == 0
    assert info["samples"] == len(before) and info["sample_rate"] == SR
    assert info["corr_with_input"] == pytest.approx(1.0, abs=1e-3)
    assert info["noise_floor_db_before"] == pytest.approx(info["noise_floor_db_after"])
    assert all(b.unloaded for b in fake_api)


def test_a_delayed_output_reports_its_lag():
    wav = speech()
    delayed = np.concatenate([np.zeros(48, np.float32), wav[:-48]])
    lag, corr = denoise_service.estimate_lag(wav, delayed, SR)
    assert lag == 48 and corr > 0.99


def test_length_changing_backend_is_rejected_and_input_kept():
    data = audio()
    before = data.waveform.copy()
    info = DenoiseService().apply(data, "short", None, None)
    assert np.array_equal(data.waveform, before)
    assert len(info["errors"]) == 1 and "samples" in info["errors"][0]


def test_nan_backend_is_rejected():
    data = audio()
    before = data.waveform.copy()
    info = DenoiseService().apply(data, "nan", None, None)
    assert np.array_equal(data.waveform, before)
    assert "NaN" in info["errors"][0]


def test_silent_output_is_rejected_where_input_has_speech():
    data = audio()
    info = DenoiseService().apply(data, "silent", None, None)
    assert "silent" in info["errors"][0] and np.abs(data.waveform).max() > 0.1


def test_a_crashing_backend_never_raises_and_drops_the_whole_chain(fake_api):
    data = audio()
    before = data.waveform.copy()
    info = DenoiseService().apply(data, "half>boom", None, None)
    assert np.array_equal(data.waveform, before), "half a chain is not the chain"
    assert "out of memory" in info["errors"][0]
    assert all(b.unloaded for b in fake_api)


def test_unknown_backend_is_an_error_entry_not_a_crash():
    info = DenoiseService().apply(audio(), "nonesuch", None, None)
    assert info["errors"]


def test_chain_runs_in_order_each_on_the_previous_output():
    data = audio()
    before = data.waveform.copy()
    DenoiseService().apply(data, "half>plus1", None, None)
    assert FakeBackend.calls == ["half", "plus1"]
    assert np.allclose(data.waveform, before * 0.5 + 0.01, atol=1e-6)
    data = audio()
    FakeBackend.calls = []
    DenoiseService().apply(data, "plus1>half", None, None)
    assert FakeBackend.calls == ["plus1", "half"]
    assert np.allclose(data.waveform, (before + 0.01) * 0.5, atol=1e-6)


def test_bad_chain_syntax_is_reported_by_parse_spec():
    with pytest.raises(ValueError):
        denoise_service.parse_spec("a>>b")
    assert denoise_service.parse_spec("none") == []


# --- the pipeline step -----------------------------------------------------

def pipeline():
    return PipelineService(*(None,) * 8)


def run_step(tmp_path, spec, data=None, ns="v1-abc", stage_out=None, computed=None):
    ck = CheckpointManager(str(tmp_path / "cache"), "job")
    ck.namespaces["processed_audio"] = ns
    data = data or audio()
    stage_out = stage_out or StageOutputService(str(tmp_path / "out"))
    computed = set() if computed is None else computed
    pipeline()._apply_denoise(args(denoise=spec), {}, ck, data, stage_out, computed)
    return ck, data, stage_out, computed


def test_no_chain_is_a_noop_with_no_cache_or_output(tmp_path):
    for spec in ("none", "", None, "NONE"):
        data = audio()
        before = data.waveform
        ck, data, _, computed = run_step(tmp_path, spec, data)
        assert data.waveform is before
        assert ck.namespaces == {"processed_audio": "v1-abc"}
        assert not computed
    assert os.listdir(tmp_path / "cache" / "job") == []
    assert not (tmp_path / "out").exists()


def test_processed_audio_namespace_is_pinned_without_a_chain():
    """Golden value: adding --denoise must not move the existing cache key."""
    ns = PipelineService._processed_audio_namespace(
        args(), {}, {"fps": 100.0, "spans": []}, {"fade": 0.0, "kept": []},
        "/nonexistent/x.wav")
    assert ns == GOLDEN_NAMESPACE
    # a chain never reaches it: the arg is not an input of the namespace
    assert PipelineService._processed_audio_namespace(
        args(denoise="identity"), {}, {"fps": 100.0, "spans": []},
        {"fade": 0.0, "kept": []}, "/nonexistent/x.wav") == ns


def test_identity_chain_writes_the_stage_files_and_caches(tmp_path):
    ck, data, _, computed = run_step(tmp_path, "identity")
    stage = tmp_path / "out" / "01b_denoise"
    assert (stage / "after_denoise.wav").is_file()
    info = json.loads((stage / "denoise.json").read_text())
    for key in ("chain", "chain_id", "sample_rate", "samples", "lag_samples",
                "noise_floor_db_before", "noise_floor_db_after", "corr_with_input",
                "errors"):
        assert key in info
    assert info["chain"] == ["identity"] and info["samples"] == len(data.waveform)
    assert "denoise" in computed
    assert ck.namespaces["denoised_audio"] == "identity-v1-abc"
    assert "denoised_audio" in os.listdir(tmp_path / "cache" / "job")
    assert not (tmp_path / "out" / "01_music").exists(), "after_music.wav is not ours"


def test_second_entry_loads_from_cache_and_does_not_rerun(tmp_path):
    run_step(tmp_path, "half")
    assert FakeBackend.calls == ["half"]
    shutil_out = tmp_path / "out"
    first = audio()
    ck, data, _, computed = run_step(tmp_path, "half", first)
    assert FakeBackend.calls == ["half"], "backend ran again on a cache hit"
    assert np.allclose(data.waveform, speech() * 0.5, atol=1e-6)
    assert "denoise" not in computed, "stage output is written only when computed"
    assert shutil_out.exists()


def test_a_different_chain_or_input_does_not_reuse_the_cache(tmp_path):
    run_step(tmp_path, "half")
    run_step(tmp_path, "plus1")
    assert FakeBackend.calls == ["half", "plus1"]
    run_step(tmp_path, "half", ns="v1-other")
    assert FakeBackend.calls == ["half", "plus1", "half"]


def test_failed_chain_is_cached_as_the_unfiltered_audio(tmp_path):
    run_step(tmp_path, "boom")
    ck, data, _, _ = run_step(tmp_path, "boom")
    assert FakeBackend.calls == ["boom"], "a failing chain must not retry every stage pass"
    assert np.array_equal(data.waveform, speech())
    info = json.loads((tmp_path / "out" / "01b_denoise" / "denoise.json").read_text())
    assert info["errors"]


def test_stage_dir_name():
    assert StageOutputService.STAGES["denoise"] == "01b_denoise"


# --- arguments --------------------------------------------------------------

def _parse(argv, config=None):
    """main.py's own parsing in a subprocess (it runs at module scope)."""
    probe = (
        "import sys, json\n"
        "sys.argv = ['main.py'] + " + repr(argv) + "\n"
        "src = open('main.py').read().split('# 2. DELAYED IMPORTS')[0]\n"
        "ns = {'__name__': 'probe', '__file__': 'main.py'}\n"
        "exec(compile(src, 'main.py', 'exec'), ns)\n"
        "print(json.dumps(ns['args'].denoise))\n"
    )
    path = os.path.join(ROOT, "_probe_denoise_args.py")
    with open(path, "w") as fh:
        fh.write(probe)
    try:
        env = dict(os.environ, SOMMELIER_SKIP_BOOTSTRAP="1")
        return subprocess.run([sys.executable, path], cwd=ROOT, env=env,
                              capture_output=True, text=True)
    finally:
        os.unlink(path)


BASE = ["--audio", "x.wav", "--env", "kaggle"]


def test_denoise_flag_defaults_to_none_from_the_profile():
    out = _parse(BASE)
    assert out.returncode == 0, out.stderr[-1500:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == "none"


def test_denoise_flag_overrides_and_accepts_stop_after_asr():
    out = _parse(BASE + ["--denoise", "dfn3>mf2se48", "--stop_after", "asr"])
    # without models.denoise the spec cannot be validated; with it, it parses
    if out.returncode == 0:
        assert json.loads(out.stdout.strip().splitlines()[-1]) == "dfn3>mf2se48"
    else:
        assert "--denoise" in out.stderr


def test_both_profiles_carry_the_key():
    # a100 filters the recording before diarization (wpe, then rf_crowd, then a 50% mix back of the
    # raw audio); kaggle is unchanged. A new profile must state its choice here.
    expected = {"kaggle": "none", "a100": "wpe>rf_crowd>blend50"}
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    for name, profile in cfg["environments"].items():
        if "pipeline" in profile:
            assert profile["pipeline"].get("denoise") == expected[name], name


GOLDEN_NAMESPACE = "v1-8f1af1a315b51aaaa1ec"
