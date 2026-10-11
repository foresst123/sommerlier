import os
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.conversation_export_service import (  # noqa: E402
    ConversationExportService,
    ConversationSelection,
    _Judged,
)
from utils.conversation_selection import Candidate  # noqa: E402
from utils.excise import TimelineMap  # noqa: E402


class _LLM:
    model_name = "test/model"


def _segments():
    return [
        SimpleNamespace(index="seg-a", speaker="A", text="xin chào", start=0.0,
                        end=40.0, review_flags=None),
        SimpleNamespace(index="seg-b", speaker="B", text="xin chào bạn", start=40.0,
                        end=80.0, review_flags=None),
    ]


def _service(**settings):
    return ConversationExportService(_LLM(), require_noise=False,
                                     min_seconds=1, max_seconds=100, **settings)


def _selection(service, report=None):
    transcript = _segments()
    from utils.conversation_selection import ConversationSelectionFinder
    finder = ConversationSelectionFinder(
        transcript, TimelineMap(), None, None, service.cfg)
    candidate = Candidate(
        first=0, last=1, start=0.0, end=80.0, pad_start=0.0, pad_end=80.0,
        speakers=("A", "B"),
        metrics={"duration": np.float32(80.0)},
        noise={"combined_p90": np.float64(0.01)},
        music_patched_share=None,
        components={"duration": np.float32(1.0)}, score=82.0, tier="A")
    item = _Judged(original=candidate, candidate=candidate, semantic=5,
                   topic="chủ đề", why="hai người trao đổi", assistant="B",
                   role_conf=np.float32(0.9), interruptions={"seg-a": "yield"},
                   summary="Một cuộc trao đổi.")
    complete = {
        "prompt_version": "conversation-export-v10-foreign-review",
        "thinking": False, "mode": "llm_finder_then_validator", "skipped": None,
        "windows": 1, "unanswered": 0, "coverage_retry_unanswered": 0,
        "unreadable": 0, "cut_in_thought": 0,
        "replies": [{"answered": True, "readable": True, "cut_in_thought": False,
                     "coverage_fraction": 1.0, "overlap_ranges": []}],
        "coverage": {"total_windows": 1, "complete_windows": 1,
                      "min_fraction": 1.0},
    }
    complete.update(report or {})
    return ConversationSelection(
        finder=finder, judged=[item], kept={id(candidate): item},
        final=[candidate], report=complete), transcript


def test_selection_checkpoint_round_trips_by_segment_index_and_json_values():
    service = _service()
    selection, transcript = _selection(service)

    payload = service.selection_checkpoint(selection)

    assert payload is not None
    assert payload["complete"] is True
    assert payload["final"] == [["seg-a", "seg-b"]]
    json.dumps(payload, ensure_ascii=False)
    assert isinstance(payload["judged"][0]["original"]["metrics"]["duration"], float)
    restored = service.selection_from_checkpoint(
        payload, transcript, timeline=TimelineMap(), noise=None, music_map=None)
    assert [seg.index for seg in restored.segments_to_align()] == ["seg-a", "seg-b"]
    assert restored.final[0].score == pytest.approx(82.0)
    assert restored.kept[id(restored.final[0])].topic == "chủ đề"
    assert restored.kept[id(restored.final[0])].interruptions == {"seg-a": "yield"}


def test_selection_checkpoint_refuses_incomplete_llm_coverage():
    service = _service()
    selection, _ = _selection(service, {"unanswered": 1})

    assert service.selection_is_complete(selection) is False
    assert service.selection_checkpoint(selection) is None


def test_selection_checkpoint_namespace_changes_with_prompt_settings():
    assert _service().checkpoint_namespace != _service(thinking=True).checkpoint_namespace
    assert _service(max_foreign_share=0.2).checkpoint_namespace != _service().checkpoint_namespace


def test_selection_checkpoint_rejects_missing_current_segment():
    service = _service()
    selection, transcript = _selection(service)
    payload = service.selection_checkpoint(selection)
    with pytest.raises(ValueError, match="missing segment index"):
        service.selection_from_checkpoint(
            payload, transcript[:1], timeline=TimelineMap(), noise=None, music_map=None)
