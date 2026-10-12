"""sample_rate / num_samples / source_stem in conversation.json and manifest rows.

Run:  python -m pytest tests/test_conversation_export_metadata.py -q     (from podcast-pipeline/)
"""
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sf = pytest.importorskip("soundfile")

from services.conversation_export_service import ConversationExportService
from utils.excise import TimelineMap
from utils.noise_map import KINDS, NoiseTrack

A, B = "SPEAKER_00", "SPEAKER_01"
SR = 8000


def _talk(n):
    return [SimpleNamespace(
        index=f"{k:05d}", speaker=(A, B)[k % 2], start=k * 8.0, end=k * 8.0 + 7.0,
        text=f"đoạn {k} bàn về chuyện số {k} rất dài dòng", bss=False, unseparated=None)
        for k in range(n)]


class _LLM:
    model_name = "fake/model"
    batch_size = 100
    max_batch_tokens = 0

    def ensure_loaded(self):
        return True

    def count_tokens(self, text):
        return len(text.split())

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None, **extra):
        replies = []
        for message in user_messages:
            n = int(re.search(r"N=(\d+) dòng", message).group(1))
            replies.append(json.dumps([{
                "start_line": 1, "end_line": n, "keep": True, "self_contained": 5,
                "topic": "chủ đề", "reason": "ok", "assistant": "B", "role_conf": 0.9,
                "role_why": "B đáp", "interruptions": [], "summary": "Hai người trao đổi."}]))
        return True, replies


@pytest.fixture
def exported(tmp_path):
    rng = np.random.default_rng(0)
    wave = (rng.standard_normal(300 * SR) * 0.1).astype(np.float32)
    noise = NoiseTrack({k: np.full(60000, 0.001, dtype=np.float32) for k in KINDS}, fps=100.0)
    svc = ConversationExportService(_LLM())
    result = svc.run(_talk(30), timeline=TimelineMap(), noise=noise, music_map=None,
                     waveform=wave, sample_rate=SR, out_dir=str(tmp_path), base_name="ep 01")
    assert result.exports
    return tmp_path, result


def test_conversation_json_records_rate_samples_and_source(exported):
    root, result = exported
    docs = sorted(root.rglob("conversation.json"))
    assert len(docs) == len(result.exports)
    for path in docs:
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["source_stem"] == "ep 01"
        assert doc["sample_rate"] == SR and isinstance(doc["sample_rate"], int)
        assert isinstance(doc["num_samples"], int)
        for key in doc["files"]:
            if key in ("alignments",):
                continue
            wav = path.parent / doc["files"][key]
            if wav.suffix != ".wav":
                continue
            info = sf.info(str(wav))
            assert info.samplerate == doc["sample_rate"]
            assert info.frames == doc["num_samples"]
        for key in ("id", "folder", "files", "duration", "speakers", "role", "swappable"):
            assert key in doc


def test_manifest_rows_and_exports_carry_source_and_rate(exported):
    root, result = exported
    rows = [json.loads(line) for line in
            (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows
    for row in rows:
        assert row["source_stem"] == "ep 01" and row["sample_rate"] == SR
        assert "path" in row and "duration" in row and "swappable" in row
    for export in result.exports:
        assert export["source_stem"] == "ep 01" and export["sample_rate"] == SR
