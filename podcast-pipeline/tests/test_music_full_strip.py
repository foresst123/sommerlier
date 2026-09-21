"""Music removal over the whole recording, and the noise measured after it.

`music_scope: spans` strips only the stretches the tagger called a bed; `full`
runs the separator over the entire recording. What has to hold for `full`:
whatever the separator returns replaces the waveform only when it is usable (a
NaN written over a recording poisons everything after it), the result is cached
and re-applied like any other patch, what was cached by one scope or checkpoint
is never reused under another, and the noise the export is judged by is the
noise of the audio it is cut from.

Run:  python -m pytest tests/test_music_full_strip.py -q     (from podcast-pipeline/)
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.audio import AudioData
from services.music_service import MusicService
from services.pipeline_service import PipelineService
from utils.checkpoint import CheckpointManager
from utils.music_map import MUSIC, MusicMap
from utils.noise_map import KINDS, NoiseTrack

SR = 1000


def _audio(seconds=20, value=1.0):
    return AudioData(name="t", waveform=np.full(seconds * SR, value, dtype=np.float32),
                     sample_rate=SR, duration=float(seconds), audio_segment=None)


class Halving:
    """A separator whose vocals are half the mixture, so a replaced sample is visible."""

    def __init__(self):
        self.full_calls, self.span_calls, self.segment_calls = [], [], []

    def separate_full(self, audio, sr):
        self.full_calls.append(len(audio))
        return (np.asarray(audio) * 0.5).astype(np.float32)

    def separate_segment(self, audio, sr):
        self.segment_calls.append(len(audio))
        return (np.asarray(audio) * 0.5).astype(np.float32)


class HiRes(Halving):
    """Decodes the source itself, as BSRoformerRemover.separate_span does."""

    def __init__(self, result="half"):
        super().__init__()
        self.result = result

    def separate_span(self, source_path, start, end, out_sr, reference):
        self.span_calls.append((source_path, start, end, out_sr, len(reference)))
        return None if self.result is None else (reference * 0.5).astype(np.float32)


class Logger:
    def __init__(self):
        self.errors, self.infos = [], []

    def error(self, msg):
        self.errors.append(msg)

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.infos.append(msg)

    def debug(self, msg):
        pass


# --- MusicService.strip_full_recording ----------------------------------------------

def test_the_whole_recording_becomes_its_vocals_and_one_patch_covers_it():
    audio = _audio()
    patches = MusicService(bs_roformer_model=Halving()).strip_full_recording(audio)
    assert np.all(audio.waveform == 0.5)
    assert len(patches) == 1 and patches[0][0] == 0
    assert len(patches[0][1]) == len(audio.waveform)


def test_the_source_is_decoded_from_its_own_file_for_the_whole_length():
    model = HiRes()
    audio = _audio(seconds=20)
    MusicService(bs_roformer_model=model).strip_full_recording(audio, source_path="/x/in.mp3")
    assert model.span_calls == [("/x/in.mp3", 0.0, 20.0, SR, 20 * SR)]
    assert model.full_calls == [], "the 16kHz waveform is only the fallback"


def test_it_falls_back_to_the_waveform_it_holds_when_the_source_cannot_be_used():
    model = HiRes(result=None)
    audio = _audio()
    patches = MusicService(bs_roformer_model=model).strip_full_recording(
        audio, source_path="/x/in.mp3")
    assert model.full_calls == [20 * SR] and patches
    assert np.all(audio.waveform == 0.5)


def test_without_a_source_path_the_held_waveform_is_used():
    model = HiRes()
    MusicService(bs_roformer_model=model).strip_full_recording(_audio())
    assert model.span_calls == [] and model.full_calls == [20 * SR]


def test_no_model_leaves_the_recording_alone():
    audio = _audio()
    assert MusicService(bs_roformer_model=None).strip_full_recording(audio) == []
    assert np.all(audio.waveform == 1.0)


def test_a_recording_shorter_than_a_second_is_left_alone():
    audio = AudioData(name="t", waveform=np.ones(SR // 2, dtype=np.float32),
                      sample_rate=SR, duration=0.5, audio_segment=None)
    assert MusicService(bs_roformer_model=Halving()).strip_full_recording(audio) == []


@pytest.mark.parametrize("bad, why", [
    (lambda ref: None, "produced nothing"),
    (lambda ref: ref[:-5] * 0.5, "samples long"),
    (lambda ref: np.where(np.arange(len(ref)) == 7, np.nan, ref * 0.5).astype(np.float32),
     "NaN or Inf"),
    (lambda ref: np.where(np.arange(len(ref)) == 7, np.inf, ref * 0.5).astype(np.float32),
     "NaN or Inf"),
    (lambda ref: (ref * 1e-4).astype(np.float32), "almost silent"),
])
def test_an_unusable_result_leaves_the_recording_exactly_as_it_was(bad, why):
    class Broken:
        def separate_full(self, audio, sr):
            return bad(np.asarray(audio))

    audio, log = _audio(), Logger()
    patches = MusicService(bs_roformer_model=Broken()).strip_full_recording(audio, logger=log)
    assert patches == []
    assert np.all(audio.waveform == 1.0)
    assert log.errors and why in log.errors[0]


def test_the_result_can_be_cached_and_reapplied_to_freshly_loaded_audio():
    service = MusicService(bs_roformer_model=Halving())
    once = _audio()
    patches = service.strip_full_recording(once)
    again = _audio()
    service.apply_music_patches(again, patches)
    service.apply_music_patches(again, patches)
    assert np.array_equal(once.waveform, again.waveform)


def test_the_separator_is_handed_back_so_a_second_recording_can_use_it():
    service = MusicService(bs_roformer_model=Halving())
    service.strip_full_recording(_audio())
    audio = _audio()
    assert service.strip_full_recording(audio) and np.all(audio.waveform == 0.5)


def test_cached_read_only_audio_is_stripped_too():
    audio = _audio()
    audio.waveform.flags.writeable = False
    assert MusicService(bs_roformer_model=Halving()).strip_full_recording(audio)
    assert np.all(audio.waveform == 0.5)


# --- PipelineService._music_scope / _music_checkpoint_name ---------------------------

def test_the_scope_defaults_to_the_old_way_and_is_read_from_the_profile():
    assert PipelineService._music_scope(SimpleNamespace()) == "spans"
    assert PipelineService._music_scope(SimpleNamespace(music_scope=" FULL ")) == "full"


def test_an_unknown_scope_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError):
        PipelineService._music_scope(SimpleNamespace(music_scope="everything"))


def test_the_checkpoint_name_comes_from_the_command_line_then_the_profile():
    config = {"environments": {"kaggle": {"models": {"bs_roformer": {"model": "a.ckpt"}}}}}
    assert PipelineService._music_checkpoint_name(SimpleNamespace(env="kaggle"), config) == "a.ckpt"
    args = SimpleNamespace(env="kaggle", music_separator="b.ckpt")
    assert PipelineService._music_checkpoint_name(args, config) == "b.ckpt"
    assert PipelineService._music_checkpoint_name(SimpleNamespace(env="x"), {}) == "default"


def test_the_shipped_profiles_run_the_separator_over_the_whole_recording():
    import json
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.json"), encoding="utf-8") as fh:
        profiles = json.load(fh)["environments"]
    for name, profile in profiles.items():
        assert profile["pipeline"]["music_scope"] == "full", name
        assert PipelineService._music_scope(
            SimpleNamespace(music_scope=profile["pipeline"]["music_scope"])) == "full"


# --- PipelineService._strip_music -----------------------------------------------------

def _pipeline(model, tmp_path, config_model="kim.ckpt"):
    pipe = PipelineService.__new__(PipelineService)
    pipe.music_svc = MusicService(bs_roformer_model=model)
    pipe.model_loader = SimpleNamespace(get=lambda name: model)
    pipe.logger = None
    pipe.noise_track = None
    pipe.loads = []
    pipe._load = pipe.loads.append
    pipe._free = lambda args, *names: None
    config = {"environments": {"kaggle": {"models": {"bs_roformer": {"model": config_model}}}}}
    return pipe, config, CheckpointManager(str(tmp_path), "job")


def _args(**kw):
    base = dict(env="kaggle", music_scope="full", step_music_removal=True,
                step_music_analysis=True, music_separator=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_full_scope_strips_even_when_the_tagger_found_no_music(tmp_path):
    model = Halving()
    pipe, config, ckpt = _pipeline(model, tmp_path)
    audio = _audio()
    assert pipe._strip_music(_args(), config, ckpt, audio, "/x.mp3", MusicMap()) is True
    assert np.all(audio.waveform == 0.5)


def test_spans_scope_does_nothing_when_there_is_no_bed_to_strip(tmp_path):
    model = Halving()
    pipe, config, ckpt = _pipeline(model, tmp_path)
    audio = _audio()
    assert pipe._strip_music(_args(music_scope="spans"), config, ckpt, audio, "/x.mp3",
                             MusicMap()) is False
    assert np.all(audio.waveform == 1.0)
    assert model.full_calls == [] and model.segment_calls == []
    assert pipe.loads == [], "the separator is not even loaded when there is nothing for it"


def test_spans_scope_still_strips_only_the_bed(tmp_path):
    model = Halving()
    pipe, config, ckpt = _pipeline(model, tmp_path)
    audio = _audio()
    music_map = MusicMap([(5.0, 8.0, MUSIC)])
    assert pipe._strip_music(_args(music_scope="spans"), config, ckpt, audio, None, music_map)
    assert audio.waveform[6 * SR] == 0.5 and audio.waveform[0] == 1.0


def test_a_switched_off_step_touches_nothing(tmp_path):
    model = Halving()
    pipe, config, ckpt = _pipeline(model, tmp_path)
    audio = _audio()
    assert pipe._strip_music(_args(step_music_removal=False), config, ckpt, audio, None,
                             MusicMap()) is False
    assert np.all(audio.waveform == 1.0) and model.full_calls == []


def test_a_later_entry_reapplies_the_cached_result_without_asking_the_model(tmp_path):
    model = Halving()
    pipe, config, ckpt = _pipeline(model, tmp_path)
    first, second = _audio(), _audio()
    pipe._strip_music(_args(), config, ckpt, first, None, MusicMap())
    pipe._strip_music(_args(), config, CheckpointManager(str(tmp_path), "job"), second,
                      None, MusicMap())
    assert len(model.full_calls) == 1
    assert np.array_equal(first.waveform, second.waveform)


def test_a_failed_separation_leaves_the_audio_and_says_it_changed_nothing(tmp_path):
    class Broken:
        def separate_full(self, audio, sr):
            return np.full(len(audio), np.nan, dtype=np.float32)

    pipe, config, ckpt = _pipeline(Broken(), tmp_path)
    audio = _audio()
    assert pipe._strip_music(_args(), config, ckpt, audio, None, MusicMap()) is False
    assert np.all(audio.waveform == 1.0)


def test_what_one_checkpoint_or_scope_cached_is_not_reused_by_another(tmp_path):
    model = Halving()
    pipe, config, ckpt = _pipeline(model, tmp_path, config_model="kim.ckpt")
    pipe._strip_music(_args(), config, ckpt, _audio(), None, MusicMap())
    assert len(model.full_calls) == 1

    # Another checkpoint: separated again.
    other, other_config, other_ckpt = _pipeline(model, tmp_path, config_model="other.ckpt")
    other._strip_music(_args(), other_config, other_ckpt, _audio(), None, MusicMap())
    assert len(model.full_calls) == 2

    # Another scope, same checkpoint: separated again, and only over the bed.
    spans, spans_config, spans_ckpt = _pipeline(model, tmp_path, config_model="kim.ckpt")
    audio = _audio()
    spans._strip_music(_args(music_scope="spans"), spans_config, spans_ckpt, audio, None,
                       MusicMap([(5.0, 8.0, MUSIC)]))
    assert len(model.full_calls) == 2 and len(model.segment_calls) == 1
    assert audio.waveform[0] == 1.0, "the full-recording result must not have been reused"


# --- PipelineService._measure_processed_noise -------------------------------------------

def _track(level, seconds=20):
    return NoiseTrack({k: np.full(seconds * 100, level, dtype=np.float32) for k in KINDS},
                      fps=100.0)


@pytest.fixture
def sweeps(monkeypatch):
    """Replaces the tagger sweep; records what it was asked to listen to."""
    seen = []

    def fake_build_maps(waveform, sample_rate, detector, logger=None, **_):
        seen.append(float(np.asarray(waveform)[0]))
        return MusicMap(), _track(0.02)

    monkeypatch.setattr("services.pipeline_service.build_maps", fake_build_maps)
    return seen


def test_the_noise_track_becomes_that_of_the_stripped_audio(tmp_path, sweeps):
    pipe, _, ckpt = _pipeline(Halving(), tmp_path)
    pipe.noise_track = _track(0.30)
    pipe._measure_processed_noise(_args(), ckpt, _audio(value=0.5))
    assert sweeps == [0.5], "the sweep listens to the waveform as it is now"
    assert float(pipe.noise_track.combined.max()) == pytest.approx(0.02)


def test_a_later_entry_loads_the_measurement_instead_of_sweeping_again(tmp_path, sweeps):
    pipe, config, ckpt = _pipeline(Halving(), tmp_path)
    pipe._strip_music(_args(), config, ckpt, _audio(), None, MusicMap())
    pipe._measure_processed_noise(_args(), ckpt, _audio(value=0.5))

    again, _, again_ckpt = _pipeline(Halving(), tmp_path)
    again.noise_track = _track(0.30)
    again._strip_music(_args(), config, again_ckpt, _audio(), None, MusicMap())
    again._measure_processed_noise(_args(), again_ckpt, _audio(value=0.5))
    assert len(sweeps) == 1
    assert float(again.noise_track.combined.max()) == pytest.approx(0.02, abs=1e-3)


def test_the_measurement_before_stripping_is_kept_where_it_was_saved(tmp_path, sweeps):
    pipe, config, ckpt = _pipeline(Halving(), tmp_path)
    ckpt.save("noise_track", _track(0.30).to_json(), fmt="json")
    pipe._strip_music(_args(), config, ckpt, _audio(), None, MusicMap())
    pipe._measure_processed_noise(_args(), ckpt, _audio(value=0.5))
    raw = NoiseTrack.from_json(ckpt.load("noise_track", fmt="json"))
    assert float(raw.combined.max()) == pytest.approx(0.30)


def test_no_tagger_step_means_the_earlier_measurement_stands(tmp_path, sweeps):
    pipe, _, ckpt = _pipeline(Halving(), tmp_path)
    pipe.noise_track = _track(0.30)
    pipe._measure_processed_noise(_args(step_music_analysis=False), ckpt, _audio())
    assert sweeps == [] and float(pipe.noise_track.combined.max()) == pytest.approx(0.30)


def test_an_empty_sweep_does_not_replace_a_real_measurement(tmp_path, monkeypatch):
    monkeypatch.setattr("services.pipeline_service.build_maps",
                        lambda *a, **k: (MusicMap(), NoiseTrack()))
    pipe, _, ckpt = _pipeline(Halving(), tmp_path)
    pipe.noise_track = _track(0.30)
    pipe._measure_processed_noise(_args(), ckpt, _audio())
    assert float(pipe.noise_track.combined.max()) == pytest.approx(0.30)
    assert not ckpt.exists("noise_track_processed", fmt="json")
