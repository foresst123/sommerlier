import os
import re
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

import models.denoise.audio_separator as mod
from models.denoise import BACKENDS
from models.denoise.audio_separator import (ALIASES, AudioSeparatorBackend, RFCrowdBackend, RFDenoiseBackend,
                                            RFDereverbBackend, UVRDeEchoBackend, RFVocals3005Backend, pick_stem,
                                            parse_stem)
from models.denoise.base import DenoiseError

SR = 44100


def _sig(n, seed=0, amp=0.4):
    return np.random.default_rng(seed).uniform(-amp, amp, n).astype(np.float32)


class FakeSeparator:
    """Mimics audio_separator.separator.Separator: writes '<input>_(<Stem>)_<model>.wav' per stem into output_dir."""
    instances = []
    stems = {}            # stem name as the model reports it -> gain applied to the input
    transform = None      # optional fn(name, stereo) -> stereo, to inject NaN / silence / wrong length
    extra_samples = 0

    def __init__(self, log_level=20, model_file_dir="/tmp/x", output_dir=None, output_format="WAV",
                 normalization_threshold=0.9, chunk_duration=None, mdxc_params=None, use_autocast=False,
                 use_native_fp16=False):
        self.kwargs = dict(model_file_dir=model_file_dir, output_dir=output_dir, normalization_threshold=normalization_threshold,
                           chunk_duration=chunk_duration, mdxc_params=mdxc_params, use_native_fp16=use_native_fp16,
                           env=os.environ.get("AUDIO_SEPARATOR_MODEL_DIR"))
        self.output_dir = output_dir
        self.loaded = None
        self.calls = 0
        FakeSeparator.instances.append(self)

    def load_model(self, model_filename):
        self.loaded = model_filename

    def separate(self, path):
        self.calls += 1
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        assert sr == SR and x.shape[1] == 2
        base = os.path.splitext(os.path.basename(path))[0]
        out = []
        for name, gain in self.stems.items():
            y = x * np.float32(gain)
            if self.transform:
                y = self.transform(name, y)
            if self.extra_samples:
                y = np.concatenate([y, np.zeros((self.extra_samples, 2), dtype=np.float32)])
            fn = f"{base}_({name})_{os.path.splitext(self.loaded)[0]}.wav"
            sf.write(os.path.join(self.output_dir, fn), y, SR, subtype="FLOAT")
            out.append(fn)  # the library returns names relative to output_dir
        return out


@pytest.fixture
def fake(monkeypatch, tmp_path):
    FakeSeparator.instances = []
    FakeSeparator.stems = {}
    FakeSeparator.transform = None
    FakeSeparator.extra_samples = 0
    monkeypatch.setattr(mod, "_import_separator", lambda: FakeSeparator)
    monkeypatch.setattr(mod, "missing_modules", lambda *names: [])
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("BS_ROFORMER_MODEL_DIR", raising=False)

    def prepare(cls, **cfg):
        for f in [cls.checkpoint or cfg.get("checkpoint"), cls.config_file, "download_checks.json"]:
            if f:
                (tmp_path / f).write_bytes(b"x")
        if not cls.config_file and not str(cls.checkpoint or cfg.get("checkpoint")).endswith(".ckpt"):
            for f in mod._HASH_FILES:
                (tmp_path / f).write_bytes(b"x")
        return cls(cfg)
    return prepare


# name, class, stems the model reports (name -> gain), expected gain of the kept signal
CASES = [
    ("rf_vocals_3005", RFVocals3005Backend, {"Vocals": 0.7, "Instrumental": 0.3}, 0.7),   # YAML says 'Vocals'
    ("rf_vocals_kim", mod.RFVocalsKimBackend, {"vocals": 0.6, "other": 0.4}, 0.6),
    ("rf_denoise", RFDenoiseBackend, {"dry": 0.9, "other": 0.1}, 0.9),
    ("rf_denoise_aggr", mod.RFDenoiseAggrBackend, {"other": 0.2, "dry": 0.8}, 0.8),        # order must not matter
    ("rf_dereverb", RFDereverbBackend, {"reverb": 0.25, "noreverb": 0.75}, 0.75),          # 'reverb' is a substring of 'noreverb'
    ("uvr_deecho", UVRDeEchoBackend, {"Reverb": 0.3, "No Reverb": 0.7}, 0.7),               # VR names have a space
    ("rf_crowd", RFCrowdBackend, {"crowd": 0.2, "other": 0.8}, 0.8),                        # speech is the NON-crowd stem
]


@pytest.mark.parametrize("name,cls,stems,gain", CASES)
def test_stem_selection_per_alias(fake, name, cls, stems, gain):
    FakeSeparator.stems = stems
    b = fake(cls)
    assert b.name == name and b.native_sr == SR
    x = _sig(SR)
    y = b.enhance(x, SR)
    assert np.allclose(y, x * gain, atol=1e-6)
    assert FakeSeparator.instances[0].loaded == cls.checkpoint


def test_crowd_invert_matches_other_stem(fake):
    FakeSeparator.stems = {"crowd": 0.2, "other": 0.8}
    b = fake(RFCrowdBackend, stem_keep="crowd", invert=True)
    x = _sig(SR)
    assert np.allclose(b.enhance(x, SR), x * 0.8, atol=1e-6)


def test_stem_keep_order_and_missing_stem(fake):
    FakeSeparator.stems = {"vocals": 0.5, "instrumental": 0.5}
    b = fake(AudioSeparatorBackend, checkpoint="m.ckpt", stem_keep=["speech", "vocals"])
    assert np.allclose(b.enhance(_sig(1000), SR), _sig(1000) * 0.5, atol=1e-6)
    b2 = fake(AudioSeparatorBackend, checkpoint="m.ckpt", stem_keep="nothing")
    with pytest.raises(DenoiseError, match="no output stem matches"):
        b2.enhance(_sig(1000), SR)


def test_generic_requires_stem_keep():
    with pytest.raises(ValueError, match="stem_keep"):
        AudioSeparatorBackend({"checkpoint": "m.ckpt"})


def test_pick_stem_exact_before_substring():
    paths = ["in_(Reverb)_m.wav", "in_(No Reverb)_m.wav"]
    assert pick_stem(paths, ["noreverb"])[0] == paths[1]
    assert pick_stem(paths, ["reverb"])[0] == paths[0]
    assert pick_stem(["in_(Vocals)_m.wav", "in_(Instrumental)_m.wav"], ["voc"])[0] == "in_(Vocals)_m.wav"  # unique substring
    assert pick_stem(paths, ["zzz"])[0] is None
    assert parse_stem("/a/b/in_(No Reverb)_model_x.wav") == "No Reverb"
    assert parse_stem("junk.wav") == ""


@pytest.mark.parametrize("sr", [24000, 16000, 48000, 22050])
@pytest.mark.parametrize("n", [1, 777, 24001, 100003])
def test_exact_length_at_input_rates(fake, sr, n):
    FakeSeparator.stems = {"dry": 1.0, "other": 0.0}
    b = fake(RFDenoiseBackend)
    y = b.enhance(_sig(n, seed=n), sr)
    assert y.shape == (n,) and y.dtype == np.float32 and np.all(np.isfinite(y))


def test_output_longer_or_shorter_is_fitted(fake):
    FakeSeparator.stems = {"dry": 1.0, "other": 0.0}
    FakeSeparator.extra_samples = 37
    b = fake(RFDenoiseBackend)
    x = _sig(SR)
    y = b.enhance(x, SR)
    assert len(y) == len(x) and np.allclose(y, x, atol=1e-6)
    FakeSeparator.extra_samples = 5000   # far more than rounding: refuse
    with pytest.raises(DenoiseError, match="returned"):
        fake(RFDenoiseBackend).enhance(x, SR)


def test_sample_aligned_identity(fake):
    FakeSeparator.stems = {"dry": 1.0, "other": 0.0}
    x = _sig(2 * SR)
    y = fake(RFDenoiseBackend).enhance(x, SR)
    assert np.array_equal(y, x)


def test_input_not_mutated_and_deterministic(fake):
    FakeSeparator.stems = {"dry": 0.5, "other": 0.5}
    b = fake(RFDenoiseBackend)
    x = _sig(24000, amp=1.5)  # above the threshold: exercises the pre-scale path
    ref = x.copy()
    y1 = b.enhance(x, 24000)
    y2 = b.enhance(x, 24000)
    assert np.array_equal(x, ref)
    assert np.array_equal(y1, y2)


def test_loud_input_level_is_restored(fake):
    FakeSeparator.stems = {"dry": 0.5, "other": 0.5}
    x = _sig(SR, amp=3.0)
    y = fake(RFDenoiseBackend).enhance(x, SR)
    assert np.allclose(y, x * 0.5, atol=1e-5)
    assert FakeSeparator.instances[0].kwargs["normalization_threshold"] == 1.0


def test_temp_files_cleaned_and_model_reused(fake):
    FakeSeparator.stems = {"dry": 1.0, "other": 0.0}
    b = fake(RFDenoiseBackend)
    b.enhance(_sig(5000), SR)
    work = FakeSeparator.instances[0].output_dir
    assert os.path.isdir(work) and os.listdir(work) == []
    b.enhance(_sig(5000), SR)
    assert len(FakeSeparator.instances) == 1 and FakeSeparator.instances[0].calls == 2
    b.unload()
    assert not os.path.exists(work) and b._sep is None


def test_temp_files_cleaned_after_failure(fake):
    FakeSeparator.stems = {"dry": 1.0, "other": 0.0}
    b = fake(AudioSeparatorBackend, checkpoint="m.ckpt", stem_keep="nothing")
    with pytest.raises(DenoiseError):
        b.enhance(_sig(5000), SR)
    assert os.listdir(FakeSeparator.instances[-1].output_dir) == []


def test_non_finite_output_raises(fake):
    FakeSeparator.stems = {"dry": 1.0, "other": 0.0}
    FakeSeparator.transform = staticmethod(lambda name, y: np.where(np.arange(len(y))[:, None] == 10, np.nan, y).astype(np.float32))
    with pytest.raises(DenoiseError, match="non-finite"):
        fake(RFDenoiseBackend).enhance(_sig(SR), SR)


def test_silent_output_where_input_has_speech_raises(fake):
    FakeSeparator.stems = {"dry": 0.0, "other": 1.0}
    with pytest.raises(DenoiseError, match="silent"):
        fake(RFDenoiseBackend).enhance(_sig(SR), SR)
    # silent input stays silent without an error
    y = fake(RFDenoiseBackend).enhance(np.zeros(SR, dtype=np.float32), SR)
    assert y.shape == (SR,) and not y.any()


def test_empty_and_bad_input(fake):
    b = fake(RFDenoiseBackend)
    assert b.enhance(np.zeros(0, dtype=np.float32), 24000).shape == (0,)
    with pytest.raises(ValueError):
        b.enhance(np.zeros((2, 10), dtype=np.float32), 24000)


def test_config_passthrough_and_env_override(fake, tmp_path, monkeypatch):
    FakeSeparator.stems = {"dry": 1.0, "other": 0.0}
    other = tmp_path / "cfgdir"
    other.mkdir()
    for f in (RFDenoiseBackend.checkpoint, RFDenoiseBackend.config_file, "download_checks.json"):
        (other / f).write_bytes(b"x")
    b = RFDenoiseBackend({"model_file_dir": str(other), "segment_size": 128, "overlap": 2, "batch_size": 3,
                          "chunk_duration": 60, "native_fp16": True})
    b.enhance(_sig(1000), SR)
    kw = FakeSeparator.instances[0].kwargs
    assert kw["model_file_dir"] == str(other) and kw["env"] == str(other)   # configured folder beats the env var
    assert kw["mdxc_params"] == {"segment_size": 128, "override_model_segment_size": False, "overlap": 2, "batch_size": 3}
    assert kw["chunk_duration"] == 60.0
    assert kw["use_native_fp16"] is False   # CPU box: fp16 is CUDA-only
    assert os.environ["AUDIO_SEPARATOR_MODEL_DIR"] == str(tmp_path)         # restored after load


def test_unknown_separator_kwargs_dropped(fake, monkeypatch):
    class Old:
        instances = FakeSeparator.instances
        def __init__(self, output_dir=None, model_file_dir=""):
            self.output_dir = output_dir
        def load_model(self, model_filename):
            self.loaded = model_filename
        def separate(self, path):
            return FakeSeparator.separate(self, path)
    FakeSeparator.stems = {"dry": 1.0, "other": 0.0}
    Old.calls, Old.stems, Old.transform, Old.extra_samples = 0, FakeSeparator.stems, None, 0
    monkeypatch.setattr(mod, "_import_separator", lambda: Old)
    assert np.allclose(fake(RFDenoiseBackend, segment_size=64, chunk_duration=30).enhance(_sig(1000), SR), _sig(1000), atol=1e-6)


# ------------------------------------------------------------ check_available

def test_check_missing_package(monkeypatch):
    monkeypatch.setattr(mod, "missing_modules", lambda *names: ["audio_separator"])
    ok, why = RFDenoiseBackend.check_available({})
    assert not ok and "audio_separator" in why and "requirements-denoise-audio-separator.txt" in why


def test_check_missing_checkpoint_without_auto_download(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "missing_modules", lambda *names: [])
    monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR", raising=False)
    monkeypatch.delenv("BS_ROFORMER_MODEL_DIR", raising=False)
    ok, why = RFDenoiseBackend.check_available({})
    assert not ok and "AUDIO_SEPARATOR_MODEL_DIR" in why
    ok, why = RFDenoiseBackend.check_available({"model_file_dir": str(tmp_path / "nope")})
    assert not ok and "does not exist" in why
    ok, why = RFDenoiseBackend.check_available({"model_file_dir": str(tmp_path)})
    assert not ok and RFDenoiseBackend.checkpoint in why and "auto_download" in why
    ok, why = RFDenoiseBackend.check_available({"model_file_dir": str(tmp_path), "auto_download": True})
    assert ok and why == ""


def test_check_dir_fallback_and_vr_hash_files(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "missing_modules", lambda *names: [])
    monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR", raising=False)
    monkeypatch.setenv("BS_ROFORMER_MODEL_DIR", str(tmp_path))
    (tmp_path / UVRDeEchoBackend.checkpoint).write_bytes(b"x")
    (tmp_path / "download_checks.json").write_bytes(b"{}")
    ok, why = UVRDeEchoBackend.check_available({})
    assert not ok and "vr_model_data.json" in why
    for f in mod._HASH_FILES:
        (tmp_path / f).write_bytes(b"{}")
    assert UVRDeEchoBackend.check_available({}) == (True, "")


def test_load_without_files_raises_denoise_error(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "missing_modules", lambda *names: [])
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path))
    with pytest.raises(DenoiseError, match="missing in"):
        RFDenoiseBackend({}).enhance(_sig(100), SR)


def test_load_failure_is_wrapped_and_cleans_up(fake, monkeypatch):
    class Broken(FakeSeparator):
        def load_model(self, model_filename):
            raise RuntimeError("boom")
    monkeypatch.setattr(mod, "_import_separator", lambda: Broken)
    b = fake(RFDenoiseBackend)
    with pytest.raises(DenoiseError, match="boom"):
        b.enhance(_sig(100), SR)
    assert b._work is None and not b.loaded


# ------------------------------------------------------------------- registry

def test_alias_names_valid_unique_and_distinct_from_registry():
    names = [c.name for c in ALIASES] + [AudioSeparatorBackend.name]
    assert all(re.fullmatch(r"[a-z0-9_]+", n) for n in names)
    assert len(names) == len(set(names))
    mine = set(ALIASES) | {AudioSeparatorBackend}
    existing = {n for n, cls in BACKENDS.items() if cls not in mine}
    assert not existing & set(names)


@pytest.mark.parametrize("cls", ALIASES)
def test_alias_metadata(cls):
    assert cls.native_sr == 44100 and cls.checkpoint and cls.stem_keep
    assert cls.checkpoint.endswith((".ckpt", ".pth"))
    assert (cls.config_file == "") == cls.checkpoint.endswith(".pth")


def test_import_does_not_pull_in_torch_or_audio_separator():
    code = ("import sys; import models.denoise.audio_separator as m; "
            "assert m.AudioSeparatorBackend.name == 'audio_separator'; "
            "assert 'torch' not in sys.modules, 'torch imported'; "
            "assert 'audio_separator' not in sys.modules, 'audio_separator imported'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
    assert r.returncode == 0, r.stderr
