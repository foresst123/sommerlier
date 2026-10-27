import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nemotron_two_pass import (
    AudioWindow,
    ChunkResult,
    build_vad_windows,
    crop_to_core,
    globalize_pass1,
    map_pass2_to_global,
    guide_windows_with_pass1,
)


def _window(start, end, context=2.0):
    return AudioWindow(
        core_start=start,
        core_end=end,
        context_start=max(0.0, start - context),
        context_end=end + context,
    )


def test_vad_boundaries_cover_the_original_audio_and_use_speech_gaps():
    windows = build_vad_windows(
        72.0,
        [(0.0, 10.0), (11.0, 24.0), (25.0, 37.0), (39.0, 55.0), (56.0, 72.0)],
        target_seconds=24.0,
        min_seconds=20.0,
        max_seconds=30.0,
    )

    assert windows[0].core_start == 0.0
    assert windows[-1].core_end == 72.0
    assert all(left.core_end == right.core_start
               for left, right in zip(windows, windows[1:]))
    assert all(20.0 <= w.duration <= 30.0 for w in windows[:-1])
    assert any(abs(w.core_end - 24.0) < 1e-6 or abs(w.core_end - 25.0) < 1e-6
               for w in windows)


@pytest.mark.parametrize("duration", [40.0, 48.0, 55.0, 61.0, 75.0, 91.0])
def test_vad_planner_rebalances_feasible_durations_without_short_tails(duration):
    windows = build_vad_windows(
        duration,
        [],
        target_seconds=25.0,
        min_seconds=20.0,
        max_seconds=30.0,
    )

    assert windows[0].core_start == 0.0
    assert windows[-1].core_end == duration
    assert all(left.core_end == right.core_start
               for left, right in zip(windows, windows[1:]))
    assert all(20.0 <= window.duration <= 30.0 for window in windows)


def test_vad_planner_has_deterministic_short_chunks_only_when_constraints_are_impossible():
    first = build_vad_windows(19.0, [], min_seconds=20.0, target_seconds=25.0,
                              max_seconds=30.0)
    impossible = build_vad_windows(31.0, [], min_seconds=20.0, target_seconds=25.0,
                                   max_seconds=30.0)

    assert [window.duration for window in first] == [19.0]
    assert [window.duration for window in impossible] == [15.5, 15.5]


def test_context_is_cropped_back_to_the_core_without_shifting_timestamps():
    window = _window(20.0, 40.0)
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": "speaker_0"},
        {"start": 2.0, "end": 24.0, "speaker": "speaker_1"},
    ]

    assert crop_to_core(segments, window) == [
        {"start": 20.0, "end": 23.0, "speaker": "speaker_0"},
        {"start": 20.0, "end": 40.0, "speaker": "speaker_1"},
    ]


def test_pass1_overlap_gives_one_global_identity_across_chunk_local_labels():
    first = ChunkResult(
        _window(0.0, 20.0),
        [{"start": 2.0, "end": 20.0, "speaker": "speaker_0"}],
    )
    second = ChunkResult(
        _window(20.0, 40.0),
        [{"start": 0.0, "end": 5.0, "speaker": "speaker_7"}],
    )

    out = globalize_pass1([first, second], min_overlap=0.1)

    assert out[0][0]["speaker"] == "SPEAKER_00"
    assert out[1][0]["speaker"] == "SPEAKER_00"
    assert out[1][0]["start"] == 20.0
    assert out[1][0]["end"] == 23.0
    assert out[1][0]["speaker_source"] == "pass1_overlap"


def test_pass1_guides_pass2_context_around_a_boundary_speaker():
    windows = [_window(0.0, 20.0), _window(20.0, 40.0)]
    pass1 = [
        {"start": 18.5, "end": 22.0, "speaker": "SPEAKER_00"},
    ]

    guided = guide_windows_with_pass1(windows, pass1, context_seconds=1.0)

    assert guided[1].context_start <= 17.5
    assert guided[0].context_end >= 23.0


def test_pass1_reconciles_the_same_speaker_across_a_silence_with_wespeaker():
    first = ChunkResult(
        _window(0.0, 30.0),
        [{"start": 0.0, "end": 12.0, "speaker": "speaker_0"}],
    )
    second = ChunkResult(
        _window(30.0, 60.0),
        [{"start": 12.0, "end": 16.0, "speaker": "speaker_7"}],
    )
    calls = []

    out = globalize_pass1(
        [first, second],
        speaker_similarity=lambda *args: calls.append(args) or 0.95,
        clean_speech_seconds={
            ("pass1", "speaker_7"): 2.0,
            ("pass1", "SPEAKER_00"): 2.0,
        },
        min_clean_speech_seconds=1.5,
    )

    assert out[1][0]["speaker"] == "SPEAKER_00"
    assert out[1][0]["speaker_source"] == "wespeaker"
    assert calls == [("speaker_7", "SPEAKER_00")]


def test_pass1_reconciliation_is_one_to_one_within_a_chunk():
    first = ChunkResult(
        _window(0.0, 30.0),
        [{"start": 0.0, "end": 36.0, "speaker": "speaker_0"}],
    )
    second = ChunkResult(
        _window(30.0, 60.0),
        [
            {"start": 1.0, "end": 3.0, "speaker": "local_a"},
            {"start": 3.0, "end": 5.0, "speaker": "local_b"},
        ],
    )

    out = globalize_pass1([first, second], min_overlap=0.1)
    speakers = [segment["speaker"] for segment in out[1]]

    assert speakers.count("SPEAKER_00") == 1
    assert len(set(speakers)) == 2


def test_short_unmatched_backchannel_is_uncertain_and_does_not_call_wespeaker():
    pass1 = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}]
    pass2 = ChunkResult(
        _window(10.0, 30.0),
        [{"start": 3.0, "end": 3.25, "speaker": "speaker_9"}],
    )
    calls = []

    out = map_pass2_to_global(
        [pass2],
        pass1,
        speaker_similarity=lambda *args: calls.append(args) or 0.99,
        clean_speech_seconds={("pass2", "speaker_9"): 0.25},
        min_clean_speech_seconds=1.5,
    )

    assert out[0][0]["speaker"] == "SPEAKER_UNKNOWN"
    assert out[0][0]["speaker_confidence"] is False
    assert calls == []


def test_unknown_pass1_backchannel_cannot_become_a_confident_pass2_identity():
    pass1 = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_UNKNOWN"}]
    pass2 = ChunkResult(
        _window(0.0, 20.0),
        [{"start": 2.0, "end": 3.0, "speaker": "speaker_9"}],
    )

    out = map_pass2_to_global([pass2], pass1)

    assert out[0][0]["speaker"] == "SPEAKER_UNKNOWN"
    assert out[0][0]["speaker_confidence"] is False


def test_speaker_similarity_is_only_used_after_clean_speech_threshold():
    pass1 = [{"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"}]
    pass2 = ChunkResult(
        _window(10.0, 30.0),
        [{"start": 3.0, "end": 6.0, "speaker": "speaker_9"}],
    )
    calls = []

    out = map_pass2_to_global(
        [pass2],
        pass1,
        speaker_similarity=lambda *args: calls.append(args) or 0.95,
        clean_speech_seconds={
            ("pass1", "SPEAKER_00"): 2.0,
            ("pass2", "speaker_9"): 2.0,
        },
        min_clean_speech_seconds=1.5,
    )

    assert calls == [("speaker_9", "SPEAKER_00")]
    assert out[0][0]["speaker"] == "SPEAKER_00"
    assert out[0][0]["speaker_source"] == "wespeaker"


def test_pass2_aggregates_all_segments_before_mapping_a_local_label():
    pass1 = [{"start": 13.0, "end": 18.0, "speaker": "SPEAKER_00"}]
    pass2 = ChunkResult(
        _window(10.0, 30.0),
        [
            {"start": 3.0, "end": 4.0, "speaker": "speaker_9"},
            {"start": 5.0, "end": 9.0, "speaker": "speaker_9"},
        ],
    )

    out = map_pass2_to_global([pass2], pass1)

    assert [segment["speaker"] for segment in out[0]] == [
        "SPEAKER_00", "SPEAKER_00"]


def test_pass2_mapping_is_one_to_one_when_two_local_labels_compete_for_one_global():
    pass1 = [{"start": 12.0, "end": 20.0, "speaker": "SPEAKER_00"}]
    pass2 = ChunkResult(
        _window(10.0, 30.0),
        [
            {"start": 2.0, "end": 5.0, "speaker": "local_a"},
            {"start": 5.0, "end": 8.0, "speaker": "local_b"},
        ],
    )

    out = map_pass2_to_global([pass2], pass1)
    speakers = [segment["speaker"] for segment in out[0]]

    assert speakers.count("SPEAKER_00") == 1
    assert speakers.count("SPEAKER_UNKNOWN") == 1


def test_wespeaker_cache_keys_keep_pass1_and_pass2_labels_separate(monkeypatch):
    import types
    import torch
    import nemotron_worker

    calls = []

    class FakeEmbedder:
        def __init__(self, **kwargs):
            pass

        def embed(self, audio, sample_rate):
            calls.append(audio.copy())
            return torch.tensor([float(len(calls)), 1.0])

    fake_module = types.SimpleNamespace(WeSpeakerONNXEmbedder=FakeEmbedder)
    monkeypatch.setitem(sys.modules, "models.wespeaker_embedding", fake_module)

    helpers = __import__("utils.nemotron_two_pass", fromlist=["ChunkResult"])
    pass2 = helpers.ChunkResult(
        _window(10.0, 30.0),
        [{"start": 3.0, "end": 5.0, "speaker": "SPEAKER_00"}],
    )
    similarity, clean = nemotron_worker._lazy_wespeaker_similarity(
        [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}],
        [pass2],
        __import__("numpy").zeros(30 * 16000, dtype="float32"),
        16000,
        1.0,
    )

    assert clean[("pass1", "SPEAKER_00")] >= 1.0
    assert clean[("pass2", "SPEAKER_00")] >= 1.0
    similarity("SPEAKER_00", "SPEAKER_00")
    assert len(calls) == 2


def test_silero_loader_uses_the_installed_package_without_torch_hub(monkeypatch):
    import types
    import torch
    import nemotron_worker

    model = object()
    fake_package = types.SimpleNamespace(
        load_silero_vad=lambda **kwargs: model,
        get_speech_timestamps=lambda audio, loaded, **kwargs: [
            {"start": 0.1, "end": 0.2}
        ],
    )
    monkeypatch.setitem(sys.modules, "silero_vad", fake_package)
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs:
                        pytest.fail("torch.hub.load must not be called"))

    vad = nemotron_worker.load_silero_vad()
    assert vad(torch.zeros(3200), 16000, {}) == [(0.1, 0.2)]


def test_missing_silero_package_fails_with_an_installation_hint(monkeypatch):
    import builtins
    import nemotron_worker

    real_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "silero_vad":
            raise ModuleNotFoundError("No module named 'silero_vad'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(RuntimeError, match="silero-vad"):
        nemotron_worker.load_silero_vad()


def test_worker_two_passes_both_use_batch_size_16_and_preserve_the_file(tmp_path):
    import numpy as np
    import soundfile as sf

    import nemotron_worker

    audio_path = tmp_path / "input.wav"
    sf.write(audio_path, np.zeros(48 * 16000, dtype=np.float32), 16000)

    class FakeModel:
        def __init__(self):
            self.calls = []

        def diarize(self, *, audio, batch_size, **kwargs):
            self.calls.append((list(audio), batch_size))
            return [["0 4 speaker_0"] for _ in audio]

    model = FakeModel()
    output = nemotron_worker.run_two_pass(
        model,
        str(audio_path),
        {
            "batch_size": 16,
            "chunk_target_seconds": 24,
            "chunk_min_seconds": 20,
            "chunk_max_seconds": 30,
            "context_seconds": 2,
            "wespeaker_fallback": False,
        },
        vad_intervals=lambda waveform, sample_rate, cfg: [(0.0, 10.0), (12.0, 48.0)],
    )

    assert [batch_size for _, batch_size in model.calls] == [16, 16]
    assert output
    assert min(segment["start"] for segment in output) >= 0.0
    assert max(segment["end"] for segment in output) <= 48.0


def test_model_pass_keeps_more_than_16_paths_in_one_batch_request():
    import nemotron_worker

    class FakeModel:
        def __init__(self):
            self.request = None

        def diarize(self, *, audio, batch_size, **kwargs):
            self.request = (list(audio), batch_size)
            return [[] for _ in audio]

    model = FakeModel()
    paths = [f"chunk_{index:03d}.wav" for index in range(40)]

    assert len(nemotron_worker._model_pass(model, paths, {})) == 40
    assert model.request == (paths, 16)


def test_legacy_one_pass_is_an_explicit_configurable_fallback():
    import nemotron_worker

    class FakeModel:
        def __init__(self):
            self.calls = []

        def diarize(self, **kwargs):
            self.calls.append(kwargs)
            return [["0 1 speaker_0"]]

    model = FakeModel()
    output = nemotron_worker.diarize(model, "unused.wav", {"legacy_one_pass": True})

    assert output == [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    assert len(model.calls) == 1 and model.calls[0]["batch_size"] == 1


def test_a100_profile_enables_the_two_pass_experiment_at_batch_16():
    import json

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json"),
              encoding="utf-8") as handle:
        cfg = json.load(handle)["environments"]["a100"]["models"]["nemotron"]

    assert cfg["two_pass"] is True
    assert cfg["legacy_one_pass"] is False
    assert cfg["batch_size"] == 16
    assert 20.0 <= cfg["chunk_min_seconds"] <= cfg["chunk_target_seconds"] <= cfg["chunk_max_seconds"] <= 30.0
