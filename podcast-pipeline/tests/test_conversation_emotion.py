"""Per-turn emotion from the finder: parsing, checkpoint compatibility, conversation.json rows.

Run:  python -m pytest tests/test_conversation_emotion.py -q     (from podcast-pipeline/)
"""
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.conversation_export_service import (  # noqa: E402
    EMOTION_LABELS, ConversationExportService, _clean_emotions, parse_finder_reply)
from utils.excise import TimelineMap  # noqa: E402
from utils.noise_map import KINDS, NoiseTrack  # noqa: E402

A, B = "SPEAKER_00", "SPEAKER_01"
SR = 8000


def test_clean_emotions_keeps_known_labels_and_clamps_conf():
    out = _clean_emotions([{"line": 2, "emotion": " Happy ", "conf": 1.7},
                           {"line": "3", "emotion": "sad", "conf": -0.2}], 1, 5)
    assert out == [{"line": 2, "emotion": "happy", "conf": 1.0},
                   {"line": 3, "emotion": "sad", "conf": 0.0}]


@pytest.mark.parametrize("bad", [
    {"line": 9, "emotion": "happy", "conf": 0.5},        # outside the region
    {"line": 2, "emotion": "joyous", "conf": 0.5},       # unknown label
    {"line": 2, "emotion": "happy", "conf": "high"},     # unreadable conf
    {"line": 2, "emotion": "happy"},                     # no conf
    {"line": 2, "emotion": "happy", "conf": True},
    {"line": "x", "emotion": "happy", "conf": 0.5},
    "happy", None, 3,
])
def test_clean_emotions_drops_rubbish(bad):
    assert _clean_emotions([bad], 1, 5) == []


def test_clean_emotions_non_list_and_duplicate_lines():
    assert _clean_emotions("happy", 1, 5) == []
    assert _clean_emotions(None, 1, 5) == []
    dup = _clean_emotions([{"line": 1, "emotion": "sad", "conf": 0.4},
                           {"line": 1, "emotion": "happy", "conf": 0.9}], 1, 5)
    assert dup == [{"line": 1, "emotion": "sad", "conf": 0.4}]


def test_parse_finder_reply_reads_emotions_and_old_replies_have_none():
    region = {"start_line": 1, "end_line": 4, "keep": True, "self_contained": 4,
              "emotions": [{"line": 1, "emotion": "neutral", "conf": 0.8}]}
    new = parse_finder_reply(json.dumps([region]))[0]
    assert new["emotions"] == [{"line": 1, "emotion": "neutral", "conf": 0.8}]
    del region["emotions"]
    assert parse_finder_reply(json.dumps([region]))[0]["emotions"] == []


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
                "role_why": "B đáp", "interruptions": [], "summary": "Hai người trao đổi.",
                "emotions": [{"line": 1, "emotion": "excited", "conf": 0.8},
                             {"line": 2, "emotion": "bogus", "conf": 0.8}]}]))
        return True, replies


def test_prompt_lists_the_emotion_labels():
    prompt = ConversationExportService(_LLM())._finder_system_prompt()
    assert "{emotion_labels}" not in prompt and "emotions" in prompt
    assert all(label in prompt for label in EMOTION_LABELS)


def test_checkpoint_round_trip_and_old_checkpoint_without_the_key():
    from tests.test_conversation_selection_checkpoint import (
        _selection, _service)
    service = _service()
    selection, transcript = _selection(service)
    selection.judged[0].emotions = {"seg-a": {"label": "happy", "conf": 0.7}}
    payload = service.selection_checkpoint(selection)
    restored = service.selection_from_checkpoint(
        payload, transcript, timeline=TimelineMap(), noise=None, music_map=None)
    assert restored.judged[0].emotions == {"seg-a": {"label": "happy", "conf": 0.7}}

    del payload["judged"][0]["emotions"]                   # a checkpoint from before emotions
    old = service.selection_from_checkpoint(
        payload, transcript, timeline=TimelineMap(), noise=None, music_map=None)
    assert old.judged[0].emotions == {}


def test_conversation_json_rows_carry_the_emotion_of_their_line(tmp_path):
    rng = np.random.default_rng(0)
    wave = (rng.standard_normal(300 * SR) * 0.1).astype(np.float32)
    noise = NoiseTrack({k: np.full(60000, 0.001, dtype=np.float32) for k in KINDS}, fps=100.0)
    talk = [SimpleNamespace(
        index=f"{k:05d}", speaker=(A, B)[k % 2], start=k * 8.0, end=k * 8.0 + 7.0,
        text=f"đoạn {k} bàn về chuyện số {k} rất dài dòng", bss=False, unseparated=None)
        for k in range(30)]
    result = ConversationExportService(_LLM()).run(
        talk, timeline=TimelineMap(), noise=noise, music_map=None, waveform=wave,
        sample_rate=SR, out_dir=str(tmp_path), base_name="ep")
    assert result.exports
    for path in tmp_path.rglob("conversation.json"):
        rows = json.loads(path.read_text(encoding="utf-8"))["conversation"]
        assert rows[0]["emotion"] == {"label": "excited", "conf": 0.8}
        assert "emotion" not in rows[1]                    # unknown label was dropped
