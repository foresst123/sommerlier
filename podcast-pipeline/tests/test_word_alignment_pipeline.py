"""Final refined text is forced-aligned before clean conversation selection."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.word_alignment_service import WordAlignmentService, apply_word_alignments
from utils.conversation_selection import (
    ConversationSelectionConfig, ConversationSelectionFinder,
    has_complete_word_alignment,
)
from utils.excise import TimelineMap


def _segment(index="00001", start=10.0, end=12.0,
             text="xin chào bạn"):
    return SimpleNamespace(
        index=index, start=start, end=end, speaker="SPEAKER_00", text=text,
        words=None, unseparated=None,
    )


def test_wav2vec_aligns_the_final_text_and_restores_global_timestamps(monkeypatch):
    seen = []

    class _WhisperX:
        @staticmethod
        def align(requests, model, metadata, audio, device, **kwargs):
            seen.extend(request["text"] for request in requests)
            words = []
            for request in requests:
                tokens = request["text"].split()
                width = (request["end"] - request["start"]) / len(tokens)
                for number, token in enumerate(tokens):
                    words.append({
                        "word": token,
                        "start": request["start"] + number * width,
                        "end": request["start"] + (number + 1) * width,
                        "score": 0.9,
                    })
            return {"word_segments": words}

    monkeypatch.setitem(sys.modules, "whisperx", _WhisperX)
    svc = WordAlignmentService(language="vi", device="cpu")
    svc._model, svc._metadata = object(), {"language": "vi"}
    seg = _segment(text="văn bản đã sửa")
    audio = SimpleNamespace(
        waveform=np.zeros(14 * 16000, dtype=np.float32), sample_rate=16000)

    result = svc.align([seg], audio)

    assert seen == ["văn bản đã sửa"]
    assert [word["word"] for word in seg.words] == ["văn", "bản", "đã", "sửa"]
    assert seg.words[0]["start"] == 10.0
    assert seg.words[-1]["end"] == 12.0
    assert result.report["word_coverage"] == 1.0


def test_clean_clip_filter_rejects_a_segment_without_complete_word_times():
    seg = _segment(start=0.0, end=5.0)
    cfg = ConversationSelectionConfig(min_seconds=1, max_seconds=20, require_noise=False,
                     require_word_alignment=True)
    finder = ConversationSelectionFinder([seg], TimelineMap(), None, None, cfg)
    assert finder.report()["breaks"]["word_alignment_missing"] == 1

    seg.words = [
        {"word": "xin", "start": 0.2, "end": 0.5},
        {"word": "chào", "start": 0.6, "end": 0.9},
        {"word": "bạn", "start": 1.0, "end": 1.2},
    ]
    assert has_complete_word_alignment(seg)


def test_alignment_checkpoint_changes_when_final_text_changes():
    svc = WordAlignmentService()
    before = svc.checkpoint_namespace_for([_segment(text="bản cũ")])
    after = svc.checkpoint_namespace_for([_segment(text="bản mới")])
    assert before != after


def test_failed_final_alignment_cannot_leave_stale_asr_words_behind():
    good = _segment(index="good")
    bad = _segment(index="bad")
    for seg in (good, bad):
        seg.words = [{"word": "stale", "start": seg.start, "end": seg.end}]
    apply_word_alignments([good, bad], {
        "good": [{"word": "xin", "start": 10.1, "end": 10.3}],
    })
    assert good.words[0]["word"] == "xin"
    assert bad.words is None


class _RecordingLogger:
    def __init__(self):
        self.records = []

    def info(self, msg, **kw):
        self.records.append(("info", msg, kw))

    def warning(self, msg, **kw):
        self.records.append(("warning", msg, kw))

    def error(self, msg, **kw):
        self.records.append(("error", msg, kw))


def test_total_alignment_failure_surfaces_the_underlying_error(monkeypatch):
    class _WhisperX:
        @staticmethod
        def align(*args, **kwargs):
            raise RuntimeError("No available kernel. Aborting execution.")

    monkeypatch.setitem(sys.modules, "whisperx", _WhisperX)
    logger = _RecordingLogger()
    svc = WordAlignmentService(language="vi", device="cpu", logger=logger)
    svc._model, svc._metadata = object(), {"language": "vi"}
    audio = SimpleNamespace(
        waveform=np.zeros(14 * 16000, dtype=np.float32), sample_rate=16000)

    with pytest.raises(RuntimeError) as excinfo:
        svc.align([_segment()], audio)

    assert "No available kernel" in str(excinfo.value)
    warnings = [r for r in logger.records if r[0] == "warning"]
    # batch failure and the per-segment retry failure are both logged with a traceback
    assert len(warnings) >= 2
    assert all(r[2].get("exc_info") for r in warnings)
    assert any("segment 00001" in r[1] for r in warnings)
