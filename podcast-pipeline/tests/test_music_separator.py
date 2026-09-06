"""What the vocal-isolation stage must not get wrong.

BS-RoFormer replaced htdemucs here. These pin the four things that would fail
silently if the wrapper drifted: the interface `MusicService` calls, the two
failure paths where a wrong answer enters the dataset as speech, the stem
choice, and -- new -- the profile knobs actually reaching the library instead
of being swallowed, which is how the previous Demucs-era `segment`/`overlap`
block sat in config.json doing nothing.

Worth recording alongside: on this corpus music removal ran on 1 of 932
segments in one file and 0 of 200 in the other -- there is 30s of music in 50
minutes. The swap matters for a corpus with more music, not for this one.

Run:  python -m pytest tests/test_music_separator.py -q   (from podcast-pipeline/)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bs_roformer import (DEFAULT_MODEL, NATIVE_SAMPLE_RATE,
                                BSRoformerRemover, _level_ratio, _match_length,
                                _to_mono)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _remover(**kw):
    return BSRoformerRemover(device=None, logger=None, **kw)


# --- the interface MusicService relies on -----------------------------------

def test_it_answers_the_questions_music_service_asks():
    for name in ("separate_full", "separate_segment", "separate_span", "unload"):
        assert hasattr(BSRoformerRemover, name), name


def test_the_checkpoint_is_named_not_left_to_the_library_default():
    """audio-separator's default model changes between releases; a run has to
    be reproducible from the profile alone."""
    assert DEFAULT_MODEL.endswith(".ckpt")
    assert "bs_roformer" in DEFAULT_MODEL


def test_the_checkpoint_can_be_overridden():
    assert _remover(model_filename="other.ckpt").model_filename == "other.ckpt"


def test_unknown_profile_keys_are_still_swallowed():
    """A stale key in a profile must not crash a run that is otherwise fine."""
    assert _remover(some_retired_knob=3).model_filename == DEFAULT_MODEL


# --- the knobs have to actually arrive --------------------------------------
# The bug this replaces: config.json carried Demucs' `segment`/`overlap`, the
# constructor swallowed them in **_ignored, and both profiles ran identical
# library defaults while looking tuned.

class _FakeSeparator:
    """Stands in for audio_separator.separator.Separator."""
    last_kwargs = None

    def __init__(self, output_dir=None, output_format=None, use_autocast=False,
                 use_native_fp16=False, use_torch_compile=False, log_level=20,
                 model_file_dir=None, chunk_duration=None,
                 normalization_threshold=0.9, mdxc_params=None):
        if use_autocast and use_native_fp16:
            raise ValueError("mutually exclusive precision modes")
        _FakeSeparator.last_kwargs = dict(locals())


def _install_fake_separator(monkeypatch, cls=_FakeSeparator):
    import types
    module = types.ModuleType("audio_separator.separator")
    module.Separator = cls
    pkg = types.ModuleType("audio_separator")
    pkg.separator = module
    monkeypatch.setitem(sys.modules, "audio_separator", pkg)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", module)


def test_the_windowing_knobs_reach_the_library_when_a_profile_does_set_them(monkeypatch):
    """None of them are set by default -- the checkpoint's config is better
    tuned than a guess -- but the plumbing has to work for the runs that do."""
    _install_fake_separator(monkeypatch)
    remover = _remover(batch_size=4, overlap=8, override_model_segment_size=True,
                       segment_size=256)
    remover._work_dir = "/tmp/x"
    params = remover._separator_kwargs()["mdxc_params"]
    assert params == {"segment_size": 256, "override_model_segment_size": True,
                      "overlap": 8, "batch_size": 4}


def test_every_profile_bounds_the_overlap_add_buffers(monkeypatch):
    """audio-separator allocates the overlap-add result and counter for the
    whole track -- ~1.4MB per second of 44.1kHz stereo, so ~4GB for a
    50-minute recording. On CUDA it puts them in host RAM, so an unbounded run
    is a Kaggle OOM. chunk_duration is what splits the input first."""
    _install_fake_separator(monkeypatch)
    for name, profile in _profiles().items():
        block = profile["models"]["bs_roformer"]
        assert block.get("chunk_duration"), f"{name} does not bound its buffers"
        remover = _remover(**{k: v for k, v in block.items() if k != "model"})
        remover._work_dir = "/tmp/x"
        assert remover._separator_kwargs()["chunk_duration"] == block["chunk_duration"]


def test_batch_size_is_not_mistaken_for_a_speed_knob():
    """audio-separator's MDXC separator: "for Roformer models, batch_size is
    not utilized due to negligible performance improvements". It runs one
    window per iteration. A profile setting it looks tuned and is not."""
    for name, profile in _profiles().items():
        assert "batch_size" not in profile["models"]["bs_roformer"], name


def test_the_overlap_is_left_to_the_checkpoint():
    """step = chunk_size // overlap, so this is the dominant cost knob -- and
    the checkpoint's own inference.num_overlap is the value UVR tuned for
    these weights. Overriding it here would replace a measured number with a
    guessed one; the library falls back to it when nothing is passed."""
    for name, profile in _profiles().items():
        assert "overlap" not in profile["models"]["bs_roformer"], name


def test_output_stems_are_not_quietly_attenuated():
    """The library normalizes each stem down to normalization_threshold, 0.9
    by default. That lands after separate_span has measured the level it is
    correcting for, so a loud vocal stem would come back quiet and the seam
    would show."""
    for name, profile in _profiles().items():
        assert profile["models"]["bs_roformer"]["normalization_threshold"] == 1.0, name


def test_the_verified_fp16_path_is_used_and_excludes_autocast(monkeypatch):
    """audio-separator lists (cuda, bs_roformer) as a verified native-fp16
    capability, and raises if autocast is asked for at the same time."""
    _install_fake_separator(monkeypatch)
    remover = BSRoformerRemover(device="cuda:0", native_fp16=True, logger=None)
    remover._work_dir = "/tmp/x"
    kwargs = remover._separator_kwargs()
    assert kwargs["use_native_fp16"] is True
    assert kwargs["use_autocast"] is False
    _FakeSeparator(**{k: v for k, v in kwargs.items()})     # would raise if both


def test_fp16_is_dropped_on_cpu_rather_than_warned_about_every_run(monkeypatch):
    _install_fake_separator(monkeypatch)
    remover = BSRoformerRemover(device="cpu", native_fp16=True,
                                torch_compile=True, logger=None)
    remover._work_dir = "/tmp/x"
    kwargs = remover._separator_kwargs()
    assert kwargs["use_native_fp16"] is False
    assert kwargs["use_torch_compile"] is False
    assert kwargs["use_autocast"] is False


def test_keys_the_installed_library_rejects_are_dropped_not_raised(monkeypatch):
    """Pinned only to >=0.47: an unknown key must not become a TypeError deep
    into a run, after the weights have already downloaded."""
    class Old:
        def __init__(self, output_dir=None, output_format=None, log_level=20):
            pass
    _install_fake_separator(monkeypatch, Old)
    remover = _remover(batch_size=4, model_file_dir="/weights")
    remover._work_dir = "/tmp/x"
    kwargs = remover._separator_kwargs()
    assert set(kwargs) == {"output_dir", "output_format", "log_level"}


def test_autocast_follows_the_device_when_fp16_is_off(monkeypatch):
    _install_fake_separator(monkeypatch)
    for device, expected in (("cuda:0", True), ("cpu", False), (None, False)):
        remover = BSRoformerRemover(device=device, logger=None)
        remover._work_dir = "/tmp/x"
        assert remover._separator_kwargs()["use_autocast"] is expected, device


def test_the_offline_weight_dir_can_come_from_the_environment(monkeypatch):
    """Checkpoints are fetched from UVR, not the HF hub, so HF_HUB_OFFLINE does
    nothing for them; the dir is what download_offline_weights.py exports."""
    monkeypatch.setenv("BS_ROFORMER_MODEL_DIR", "/weights/audio-separator")
    assert _remover().model_file_dir == "/weights/audio-separator"
    assert _remover(model_file_dir="/explicit").model_file_dir == "/explicit"


# --- the two paths where a wrong answer would be silent ---------------------

def test_a_failed_segment_keeps_the_mixture_rather_than_going_silent():
    """Silence here would enter the dataset labelled as speech."""
    remover = _remover()
    remover._run = lambda audio, sr: None          # separation unavailable
    audio = np.full(1000, 0.25, dtype=np.float32)
    assert np.array_equal(remover.separate_segment(audio, 16000), audio)


def test_a_failed_full_pass_returns_none_so_the_caller_can_fall_back():
    remover = _remover()
    remover._run = lambda audio, sr: None
    assert remover.separate_full(np.zeros(1000, dtype=np.float32), 16000) is None


def test_empty_audio_is_refused_before_the_model_loads():
    """_get_model would download weights; an empty array must not trigger that."""
    remover = _remover()
    remover._get_model = lambda: (_ for _ in ()).throw(AssertionError("must not load"))
    assert remover._run(np.array([], dtype=np.float32), 16000) is None


# --- stem selection ---------------------------------------------------------

def test_the_vocal_stem_is_picked_by_name_not_by_position():
    """Order is undocumented, and taking the instrumental stem would be a
    total, silent failure -- speech replaced by backing track."""
    import inspect
    source = inspect.getsource(BSRoformerRemover._run)
    assert '"vocal" in os.path.basename(p).lower()' in source


def test_unload_is_safe_before_anything_was_loaded():
    remover = _remover()
    remover.unload()
    assert remover._separator is None


# --- the 44.1kHz path -------------------------------------------------------

def test_the_native_rate_is_the_one_the_checkpoint_was_trained_at():
    """16kHz mono is what the pipeline carries and it throws away everything
    the band-split model uses above 8kHz."""
    assert NATIVE_SAMPLE_RATE == 44100


def test_hi_res_is_skipped_rather_than_guessed_when_the_source_is_gone():
    remover = _remover()
    ref = np.ones(1600, dtype=np.float32)
    assert remover.separate_span("/no/such/file.wav", 0.0, 0.1, 16000, ref) is None
    assert remover.separate_span(None, 0.0, 0.1, 16000, ref) is None


def test_hi_res_can_be_turned_off_from_the_profile(tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(b"not really a wav")
    remover = _remover(hi_res=False)
    ref = np.ones(1600, dtype=np.float32)
    assert remover.separate_span(str(src), 0.0, 0.1, 16000, ref) is None


def test_a_broken_decode_falls_back_instead_of_raising(tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(b"not really a wav")
    ref = np.ones(1600, dtype=np.float32)
    assert _remover().separate_span(str(src), 0.0, 0.1, 16000, ref) is None


def test_the_level_of_a_freshly_decoded_span_is_matched_back():
    """The pipeline normalizes each recording to -20 dBFS at decode and does
    not keep the gain, so a span read from the source is at the original level.
    Writing it back unscaled leaves a step at both seams."""
    source = np.full(4000, 0.1, dtype=np.float32)
    normalized = source * 3.7
    assert _level_ratio(normalized, source) == pytest.approx(3.7, rel=1e-6)


def test_a_silent_span_leaves_the_level_alone():
    """Nothing to measure means nothing to get wrong; a ratio here would be
    either a division by zero or noise amplified by a huge factor."""
    silence = np.zeros(4000, dtype=np.float32)
    assert _level_ratio(silence, np.full(4000, 0.1, dtype=np.float32)) == 1.0
    assert _level_ratio(np.full(4000, 0.1, dtype=np.float32), silence) == 1.0
    assert _level_ratio(np.array([]), np.array([])) == 1.0


def test_the_replacement_is_exactly_as_long_as_the_slice_it_overwrites():
    """MusicService assigns this into waveform[i:j]; one frame out is either an
    exception or a silently shifted span."""
    for n in (10, 100):
        assert len(_match_length(np.ones(97, dtype=np.float32), n)) == n
    padded = _match_length(np.ones(3, dtype=np.float32), 5)
    assert list(padded) == [1, 1, 1, 0, 0]
    stereo = _match_length(np.ones((3, 2), dtype=np.float32), 5)
    assert stereo.shape == (5, 2)


def test_stereo_is_downmixed_only_after_separation():
    """The checkpoint is stereo; collapsing the channels before it runs is the
    same information loss as feeding it 16kHz."""
    stereo = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert list(_to_mono(stereo)) == [0.5, 0.5]
    mono = np.array([1.0, 0.0], dtype=np.float32)
    assert list(_to_mono(mono)) == [1.0, 0.0]


# --- the profile switch -----------------------------------------------------

def _profiles():
    import json
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)["environments"]


def test_both_profiles_name_a_checkpoint():
    for name, profile in _profiles().items():
        model = profile["models"]["bs_roformer"].get("model")
        assert model and model.endswith(".ckpt"), name


def test_the_checkpoint_is_the_best_scoring_one_audio_separator_ships():
    """audio-separator bundles models-scores.json, which ranks checkpoints on
    a shared 40-track set. ep_368 beats the ep_317 this pipeline started with
    (11.63 vs 11.43 mean vocal SDR) at identical cost -- same architecture,
    same windowing. Pinned so a future edit has to argue with the number."""
    for name, profile in _profiles().items():
        assert "ep_368" in profile["models"]["bs_roformer"]["model"], name


def test_no_profile_still_carries_the_retired_demucs_knobs():
    """`segment` and `overlap`-as-a-fraction were Demucs'. They survived the
    swap in config.json for months, ignored, while both profiles looked tuned."""
    for name, profile in _profiles().items():
        block = profile["models"]["bs_roformer"]
        assert "segment" not in block, f"{name} still carries Demucs' segment"
        assert not isinstance(block.get("overlap"), float), (
            f"{name}: overlap is a window count for BS-RoFormer, not a fraction")


def test_every_profile_key_is_a_constructor_argument():
    """The check that would have caught the dead knobs: anything in the block
    other than `model` has to be a parameter the wrapper actually reads."""
    import inspect
    params = set(inspect.signature(BSRoformerRemover.__init__).parameters)
    for name, profile in _profiles().items():
        for key in profile["models"]["bs_roformer"]:
            if key == "model":
                continue
            assert key in params, f"{name}.{key} is swallowed by **_ignored"


def test_the_loader_strips_model_before_passing_kwargs():
    """`model` names the checkpoint; the rest of the block is constructor
    arguments. Forwarding it would reach BSRoformerRemover as an argument it
    does not take."""
    source = open(os.path.join(ROOT, "services", "model_loader.py"),
                  encoding="utf-8").read()
    assert 'bs_roformer_cfg.pop("model", None)' in source
    assert "bs_roformer_cfg = dict(" in source, "the profile dict must be copied before popping"
    assert 'bs_roformer_cfg["model_filename"] = checkpoint' in source, (
        "popping the checkpoint and then not passing it is how it got lost before")


def test_the_cli_override_reaches_the_loader():
    """--music_separator was a dead flag: duplicated choices, read by nobody."""
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert 'choices=["bs_roformer", "bs_roformer"]' not in main
    assert "MUSIC_SEPARATOR" not in main, "the env hop published a name nothing read"
    loader = open(os.path.join(ROOT, "services", "model_loader.py"),
                  encoding="utf-8").read()
    assert 'getattr(self.args, "music_separator", None)' in loader


# --- the hi-res path end to end ---------------------------------------------

def _write_source(path, seconds=2.0, sr=NATIVE_SAMPLE_RATE):
    """A stereo 44.1kHz file with content above 8kHz, so a 16kHz round trip is
    visibly lossy and the two paths cannot be confused."""
    import soundfile as sf
    t = np.arange(int(seconds * sr)) / sr
    left = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 14000 * t)
    right = 0.3 * np.sin(2 * np.pi * 330 * t)
    sf.write(path, np.stack([left, right], axis=1).astype(np.float32), sr)
    return np.stack([left, right], axis=1).astype(np.float32)


def test_a_hi_res_span_comes_back_at_the_pipeline_rate_length_and_level(tmp_path):
    """The whole point of the path: separate at 44.1kHz stereo, hand back
    something that drops straight into a 16kHz mono waveform slice."""
    import librosa
    src = tmp_path / "src.wav"
    _write_source(src)

    remover = _remover()
    # A perfect separator: whatever it is given is all vocals. That isolates
    # the resample / downmix / level / length arithmetic, which is what the
    # caller depends on and what a real model would not change.
    remover._run = lambda audio, sr: audio

    span = (0.25, 1.75)
    at16k, _ = librosa.load(str(src), sr=16000, mono=True,
                            offset=span[0], duration=span[1] - span[0])
    gain = 3.7                       # what AudioService's -20 dBFS pass applied
    reference = (at16k * gain).astype(np.float32)

    out = remover.separate_span(str(src), *span, 16000, reference)
    assert out is not None
    assert out.dtype == np.float32
    assert len(out) == len(reference), "must drop into waveform[i:j] unchanged"
    # Same level as the slice it replaces: a step at the seam is audible and
    # hands ASR a passage at the wrong loudness.
    rms = lambda a: float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))
    assert rms(out) == pytest.approx(rms(reference), rel=0.02)
    assert np.max(np.abs(out)) <= 1.0


def test_the_model_actually_sees_44_1khz_stereo(tmp_path):
    """If this regresses to the 16kHz mono slice, the checkpoint loses every
    band above 8kHz -- most of what it is chosen over Demucs for."""
    src = tmp_path / "src.wav"
    _write_source(src)
    seen = {}

    def spy(audio, sr):
        seen["sr"], seen["shape"] = sr, audio.shape
        return audio

    remover = _remover()
    remover._run = spy
    remover.separate_span(str(src), 0.25, 1.75, 16000,
                          np.ones(24000, dtype=np.float32))
    assert seen["sr"] == NATIVE_SAMPLE_RATE
    assert seen["shape"][1] == 2, "stereo was collapsed before separation"


def test_a_separator_that_fails_mid_span_falls_back_rather_than_silencing(tmp_path):
    src = tmp_path / "src.wav"
    _write_source(src)
    remover = _remover()
    remover._run = lambda audio, sr: None
    assert remover.separate_span(str(src), 0.25, 1.75, 16000,
                                 np.ones(24000, dtype=np.float32)) is None
