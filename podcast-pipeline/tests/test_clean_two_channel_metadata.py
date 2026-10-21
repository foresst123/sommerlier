import json
import os
import sys

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.clean_two_channel_dataset_service import CleanTwoChannelDatasetService
from services.separation_service import SeparationService
from utils.excise import TimelineMap
from tests.test_clean_two_channel_dataset import (
    SR, A, B, _settings, _speech, _transcript)

PRE_EXISTING_KEYS = {
    "schema_version", "item_id", "tier", "score", "source_start", "source_end",
    "duration", "orig_spans", "sample_rate", "num_samples", "speakers", "audio",
    "metrics", "components", "noise", "music_patched_share", "separated_spans",
    "failed_separation_spans", "conversation",
}


def _export(tmp_path):
    source = tmp_path / "my_episode.v2.wav"
    source.write_bytes(b"x")
    transcripts = [_transcript(i, s, 2.2 * i, 2.2 * i + 2.0)
                   for i, s in enumerate([A, B, A, B])]
    speech = [_speech(i, s, 2.2 * i, 2.2 * i + 2.0, 0.2 + 0.2 * (i % 2))
              for i, s in enumerate([A, B, A, B])]
    root = tmp_path / "clean"
    result = CleanTwoChannelDatasetService().export(
        root=str(root), source_path=str(source), transcripts=transcripts,
        speech_segments=speech, separation_service=SeparationService(logger=None),
        timeline=TimelineMap(), noise=None, music_map=None, sample_rate=SR,
        audio_duration=9.0, selection_settings=_settings())
    return root, result


def test_item_metadata_has_source_stem_and_written_audio_facts(tmp_path):
    root, result = _export(tmp_path)
    assert result["item_count"] >= 1
    source_dir = root / result["source_id"]
    with open(source_dir / "source.json", encoding="utf-8") as handle:
        rows = json.load(handle)["items"]
    for row in rows:
        assert row["source_stem"] == "my_episode.v2"
        item_dir = source_dir / row["folder"]
        with open(item_dir / "metadata.json", encoding="utf-8") as handle:
            metadata = json.load(handle)
        assert metadata["source_stem"] == "my_episode.v2"
        # The registry's numbered id keeps its old meaning: the new key is additive.
        assert metadata["source_id"] == result["source_id"]
        assert PRE_EXISTING_KEYS <= set(metadata)
        assert not {"role", "roles", "assistant", "user"} & set(metadata)
        for name in ("sp1_clean.wav", "sp2_clean.wav", "audio_2ch.wav"):
            data, sr = sf.read(item_dir / name)
            assert sr == metadata["sample_rate"]
            assert len(data) == metadata["num_samples"]
