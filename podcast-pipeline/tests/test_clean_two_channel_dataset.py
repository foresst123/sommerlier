import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.segment import SpeechSegment
from services.clean_two_channel_dataset_service import CleanTwoChannelDatasetService
from services.pipeline_service import PipelineService
from services.separation_service import SeparationService
from utils.excise import TimelineMap


SR = 8000
A, B, C = "SPEAKER_00", "SPEAKER_01", "SPEAKER_02"


def _transcript(index, speaker, start, end):
    return SimpleNamespace(
        index=f"{index:05d}", speaker=speaker, start=start, end=end,
        text=f"cau noi day du cua nguoi {speaker} o luot {index}", words=None)


def _speech(index, speaker, start, end, value):
    return SpeechSegment(
        index=f"{index:05d}", speaker=speaker, start=start, end=end,
        audio=np.full(round((end - start) * SR), value, dtype=np.float32))


def _settings():
    return {
        "min_seconds": 3.0,
        "max_seconds": 20.0,
        "plateau_min": 3.0,
        "plateau_max": 10.0,
        "max_silence": 2.0,
        "max_monologue": 20.0,
        "boundary_gap": 0.0,
        "min_turns_each": 1,
        "min_score": 0.0,
        "require_noise": False,
        "require_semantic": False,
        "require_word_alignment": False,
        "pad_seconds": 0.0,
    }


def test_blank_output_path_does_no_filesystem_work(tmp_path):
    service = CleanTwoChannelDatasetService()
    result = service.export(
        root="   ", source_path=str(tmp_path / "input.wav"),
        transcripts=None, speech_segments=None, separation_service=None,
        timeline=None, noise=None, music_map=None, sample_rate=SR,
        audio_duration=0.0, selection_settings={})
    assert result == {"enabled": False, "item_count": 0}
    assert list(tmp_path.iterdir()) == []


def test_pipeline_does_not_call_exporter_when_config_path_is_blank():
    class Spy(CleanTwoChannelDatasetService):
        called = False

        def export(self, **_kwargs):
            self.called = True
            raise AssertionError("blank config must not reach the exporter")

    spy = Spy()
    pipeline = PipelineService(*(None,) * 8, clean_dataset_svc=spy)
    args = SimpleNamespace(env="kaggle", bss=True)
    config = {"environments": {"kaggle": {
        "outputs": {"clean_two_channel_dir": ""}}}}
    result = pipeline._export_clean_two_channel_dataset(
        args, config, [], [], None, None, "input.wav")
    assert result is None
    assert spy.called is False


def test_parallel_files_receive_distinct_numeric_source_ids(tmp_path):
    root = tmp_path / "clean"
    root.mkdir()
    sources = [tmp_path / "one.wav", tmp_path / "two.wav"]
    for source in sources:
        source.write_bytes(source.name.encode())
    service = CleanTwoChannelDatasetService()
    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(
            lambda path: service._reserve_source(str(root), str(path))[0],
            sources))
    assert sorted(ids) == ["000001", "000002"]
    with open(root / "index.json", encoding="utf-8") as handle:
        registry = json.load(handle)
    assert sorted(registry["sources"]) == ["000001", "000002"]


def test_explicit_pair_ignores_other_speakers():
    service = SeparationService(logger=None)
    segments = [
        _speech(0, B, 0.0, 1.0, 0.2),
        _speech(1, C, 1.0, 2.0, 0.3),
        _speech(2, A, 2.0, 3.0, 0.4),
    ]
    left, right = service.export_sdlm_dual_channel(
        segments, 3.0, SR, speakers=(C, A))
    assert np.allclose(left[SR:2 * SR], 0.3)
    assert np.allclose(right[2 * SR:3 * SR], 0.4)
    assert np.allclose(left[:SR], 0.0) and np.allclose(right[:SR], 0.0)

    clipped_left, clipped_right = service.export_sdlm_dual_channel(
        segments, 3.0, SR, speakers=(C, A), time_range=(1.0, 3.0))
    assert len(clipped_left) == len(clipped_right) == 2 * SR
    assert np.allclose(clipped_left[:SR], 0.3)
    assert np.allclose(clipped_right[SR:], 0.4)


def test_export_writes_aligned_clean_tracks_and_stable_source_id(tmp_path):
    source = tmp_path / "episode.wav"
    source.write_bytes(b"source identity")
    root = tmp_path / "clean"
    transcripts = [
        _transcript(0, A, 0.0, 2.0),
        _transcript(1, B, 2.2, 4.2),
        _transcript(2, A, 4.4, 6.4),
        _transcript(3, B, 6.6, 8.6),
    ]
    speech = [
        _speech(0, A, 0.0, 2.0, 0.2),
        _speech(1, B, 2.2, 4.2, 0.4),
        _speech(2, A, 4.4, 6.4, 0.2),
        _speech(3, B, 6.6, 8.6, 0.4),
    ]
    exporter = CleanTwoChannelDatasetService()
    separation = SeparationService(logger=None)

    kwargs = dict(
        root=str(root), source_path=str(source), transcripts=transcripts,
        speech_segments=speech, separation_service=separation,
        timeline=TimelineMap(), noise=None, music_map=None,
        sample_rate=SR, audio_duration=9.0, selection_settings=_settings())
    first = exporter.export(**kwargs)
    second = exporter.export(**kwargs)

    assert first["source_id"] == second["source_id"] == "000001"
    assert second["item_count"] >= 1
    with open(root / "index.json", encoding="utf-8") as handle:
        registry = json.load(handle)
    assert list(registry["sources"]) == ["000001"]
    assert registry["sources"]["000001"]["original_path"] == str(source)

    source_dir = root / "000001"
    with open(source_dir / "source.json", encoding="utf-8") as handle:
        source_meta = json.load(handle)
    item = source_meta["items"][0]
    item_dir = source_dir / item["folder"]
    sp1, sr1 = sf.read(item_dir / "sp1_clean.wav")
    sp2, sr2 = sf.read(item_dir / "sp2_clean.wav")
    stereo, sr_stereo = sf.read(item_dir / "audio_2ch.wav")
    with open(item_dir / "metadata.json", encoding="utf-8") as handle:
        metadata = json.load(handle)

    assert sr1 == sr2 == sr_stereo == SR
    assert len(sp1) == len(sp2) == len(stereo) == metadata["num_samples"]
    assert stereo.shape == (len(sp1), 2)
    assert np.allclose(stereo[:, 0], sp1, atol=1 / 32768)
    assert np.allclose(stereo[:, 1], sp2, atol=1 / 32768)
    assert metadata["audio"]["method"] == "strict_separation_tracks"
    assert metadata["separated_spans"] == []
    assert metadata["failed_separation_spans"] == []
    assert metadata["speakers"] == {
        "SP1": A, "SP2": B, "left": "SP1", "right": "SP2"}


def test_pooled_export_matches_sequential_tree(tmp_path, monkeypatch):
    import services.clean_two_channel_dataset_service as mod

    def cand(k):
        return SimpleNamespace(
            speakers=(A, B), pad_start=k * 4.0, pad_end=k * 4.0 + 3.0, tier="S",
            score=90 - k, metrics={}, components={}, noise=None,
            music_patched_share=0.0)

    speech = [_speech(2 * k, A, k * 4.0, k * 4.0 + 1.5, 0.2) for k in range(5)]
    speech += [_speech(2 * k + 1, B, k * 4.0 + 1.5, k * 4.0 + 3.0, 0.4) for k in range(5)]
    speech.append(_speech(99, A, 30.0, 33.0, 0.2))     # candidate 6 has no B voice
    monkeypatch.setattr(mod, "pick_non_overlapping", lambda c: [cand(k) for k in range(6)])
    monkeypatch.setattr(CleanTwoChannelDatasetService, "_conversation",
                        staticmethod(lambda finder, candidate: []))

    def run(workers, name):
        source = tmp_path / f"{name}.wav"
        source.write_bytes(b"x")
        root = tmp_path / name
        out = CleanTwoChannelDatasetService(workers=workers).export(
            root=str(root), source_path=str(source), transcripts=[], speech_segments=speech,
            separation_service=SeparationService(logger=None), timeline=TimelineMap(),
            noise=None, music_map=None, sample_rate=SR, audio_duration=40.0,
            selection_settings=_settings())
        files = {}
        for path in sorted((root / "000001").rglob("*")):
            if path.is_file():
                files[str(path.relative_to(root))] = path.read_bytes()
        return out, files

    seq, seq_files = run(1, "seq")
    par, par_files = run(4, "par")
    assert seq["item_count"] == par["item_count"] == 5
    assert seq["skipped_empty_channel"] == par["skipped_empty_channel"]
    assert seq_files.keys() == par_files.keys()
    assert all(seq_files[k] == par_files[k] for k in seq_files if not k.endswith("source.json"))
