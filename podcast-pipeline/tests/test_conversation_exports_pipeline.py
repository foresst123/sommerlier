"""How PipelineService hands a finished file to the conversation-export pass.

The pass needs three things the pipeline already has in hand at that point --
the audio it worked on, the cut timeline, and the SSLAM noise and music maps --
and each has a way to go quietly wrong: the music map is in the ORIGINAL
timeline while clips are cut in the shortened one, and a missing noise track must
not read as a clean recording.

Run:  python -m pytest tests/test_conversation_exports_pipeline.py -q     (from podcast-pipeline/)
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("soundfile")

from services.conversation_export_service import ConversationExportRun, ConversationExportService
from services.pipeline_service import PipelineService
from services.stage_output_service import StageOutputService
from utils.excise import TimelineMap
from utils.music_map import MUSIC, MusicMap
from utils.noise_map import KINDS, NoiseTrack

A, B = "SPEAKER_00", "SPEAKER_01"
SR = 8000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _seg(i, speaker, start, end):
    return SimpleNamespace(
        index=f"{i:05d}", speaker=speaker, start=start, end=end,
        text=f"đoạn {i} bàn về chuyện số {i} rất dài dòng", bss=False, unseparated=None)


def _talk(n):
    return [_seg(k, (A, B)[k % 2], k * 8.0, k * 8.0 + 7.0) for k in range(n)]


def _quiet(total=600):
    return NoiseTrack({k: np.full(int(total * 100), 0.001, dtype=np.float32)
                       for k in KINDS}, fps=100.0)


class _LLM:
    model_name = "fake/model"
    batch_size = 100
    max_batch_tokens = 0

    def ensure_loaded(self):
        return True

    def count_tokens(self, text):
        return len(text.split())

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None):
        return True, [json.dumps({"self_contained": 5, "topic": "t"})
                      for _ in user_messages]


class _StageOut:
    def __init__(self):
        self.calls = []

    def write_conversation_exports(self, report, exports):
        self.calls.append((report, exports))


class _Capture:
    """A clip service that only records how it was called."""

    def __init__(self):
        self.kwargs = None

    def run(self, transcripts, **kwargs):
        self.kwargs = kwargs
        return ConversationExportRun(exports=[], report={"skipped": "captured"})


def _pipeline(conversation_export_svc, timeline=None, noise=None):
    pipe = PipelineService.__new__(PipelineService)
    pipe.conversation_export_svc = conversation_export_svc
    pipe.timeline = timeline if timeline is not None else TimelineMap()
    pipe.noise_track = noise
    pipe.logger = None
    return pipe


def _audio(seconds=300.0):
    return SimpleNamespace(waveform=np.zeros(int(seconds * SR), dtype=np.float32),
                           sample_rate=SR)


def test_a_finished_file_is_cut_into_the_output_folder(tmp_path):
    out = _StageOut()
    pipe = _pipeline(ConversationExportService(_LLM()), noise=_quiet())
    result = pipe._export_conversation_exports(out, _talk(30), _audio(), MusicMap(),
                                 str(tmp_path), "/data/episode 7.mp3")
    assert result.exports
    assert (tmp_path / "conversation_exports" / "audio").is_dir()
    assert (tmp_path / "conversation_exports" / "metadata").is_dir()
    assert result.exports[0]["id"].startswith("episode_7_conversation_")
    assert out.calls and out.calls[0][1] == result.exports


def test_without_a_noise_track_nothing_is_cut_and_it_says_so(tmp_path):
    out = _StageOut()
    pipe = _pipeline(ConversationExportService(_LLM()), noise=None)
    result = pipe._export_conversation_exports(out, _talk(30), _audio(), MusicMap(),
                                 str(tmp_path), "ep.mp3")
    assert result.exports == []
    assert out.calls[0][0]["skipped"] == "noise_not_measured"
    assert not (tmp_path / "conversation_exports" / "audio").exists()


def test_the_music_map_is_moved_into_the_cut_timeline_before_it_is_used(tmp_path):
    timeline = TimelineMap(kept=[(0, 50, 0), (80, 400, 50)])     # 30s removed
    music = MusicMap([(100.0, 110.0, MUSIC)])                    # original time
    capture = _Capture()
    _pipeline(capture, timeline=timeline, noise=_quiet())._export_conversation_exports(
        _StageOut(), [], _audio(), music, str(tmp_path), "ep.mp3")
    moved = capture.kwargs["music_map"]
    assert [(round(a), round(b)) for a, b, _ in moved.spans] == [(70, 80)]


def test_with_nothing_cut_the_music_map_is_passed_through_unchanged(tmp_path):
    music = MusicMap([(100.0, 110.0, MUSIC)])
    capture = _Capture()
    _pipeline(capture, noise=_quiet())._export_conversation_exports(
        _StageOut(), [], _audio(), music, str(tmp_path), "ep.mp3")
    assert capture.kwargs["music_map"] is music


def test_the_pass_gets_the_audio_the_pipeline_worked_on(tmp_path):
    audio = _audio(120.0)
    capture = _Capture()
    _pipeline(capture, noise=_quiet())._export_conversation_exports(
        _StageOut(), [], audio, None, str(tmp_path), "ep.mp3")
    assert capture.kwargs["waveform"] is audio.waveform
    assert capture.kwargs["sample_rate"] == SR
    assert capture.kwargs["music_map"] is None


# --- the gate and the shipped settings -----------------------------------------------

def test_the_step_is_opt_in_and_keeps_the_llm_loaded_for_it():
    source = open(os.path.join(ROOT, "services", "pipeline_service.py"),
                  encoding="utf-8").read()
    assert 'opt_in_step_enabled(args, "conversation_exports")' in source
    assert 'self.step_enabled(args, "conversation_exports")' not in source
    assert "relabel_on or conversation_exports_on" in source


def test_the_pass_runs_after_relabel_and_before_the_llm_is_released():
    source = open(os.path.join(ROOT, "services", "pipeline_service.py"),
                  encoding="utf-8").read()
    relabel = source.index('opt_in_step_enabled(args, "speaker_relabel")')
    alignment = source.index('opt_in_step_enabled(args, "word_alignment")')
    exports = source.index('opt_in_step_enabled(args, "conversation_exports")')
    unload = source.index("relabel_on or conversation_exports_on")
    assert relabel < alignment < exports < unload


def test_the_shipped_profiles_accept_their_own_conversation_selection_settings():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for name, profile in config["environments"].items():
        ConversationExportService(_LLM(), **profile["models"]["conversation_selection"])


def test_the_clip_output_has_a_numbered_stage_directory():
    assert StageOutputService.STAGES["word_alignment"] == "08_word_alignment"
    assert StageOutputService.STAGES["conversation_exports"] == "09_conversation_exports"
    assert StageOutputService.STAGES["relabel"] == "07_relabel"


def test_the_clip_report_is_written_with_its_counts(tmp_path):
    out = StageOutputService(str(tmp_path))
    report = {"candidates": 5, "shortlisted": 3, "accepted": 1, "exported": 1,
              "skipped": None, "finder": {"blocks": 4, "blocks_short": 2,
                                          "noise_measured": True}}
    out.write_conversation_exports(report, [{"id": "x_conv_000001"}])
    folder = tmp_path / "09_conversation_exports"
    assert json.loads((folder / "exports.json").read_text(encoding="utf-8")) == [
        {"id": "x_conv_000001"}]
    assert json.loads((folder / "report.json").read_text(encoding="utf-8"))["candidates"] == 5
    assert json.loads((folder / "stats.json").read_text(encoding="utf-8"))["exported"] == 1


def test_a_skipped_pass_leaves_a_warning_saying_why(tmp_path):
    out = StageOutputService(str(tmp_path))
    out.write_conversation_exports({"skipped": "noise_not_measured", "finder": {}}, [])
    stats = json.loads((tmp_path / "09_conversation_exports" / "stats.json").read_text(encoding="utf-8"))
    assert any("noise" in w for w in stats["warnings"])


def test_raw_replies_go_to_their_own_file_not_into_the_report(tmp_path):
    out = StageOutputService(str(tmp_path))
    rows = [{"first_index": "00021", "raw": "{\"self_contained\": 4}", "outcome": "accepted"}]
    out.write_conversation_exports({"finder": {}, "replies": rows, "unreadable": 1}, [])
    folder = tmp_path / "09_conversation_exports"
    assert json.loads((folder / "replies.json").read_text(encoding="utf-8")) == rows
    assert "replies" not in json.loads((folder / "report.json").read_text(encoding="utf-8"))
    stats = json.loads((folder / "stats.json").read_text(encoding="utf-8"))
    assert any("replies.json" in w for w in stats["warnings"])


def test_relabel_replies_go_to_their_own_file_and_unreadable_windows_warn(tmp_path):
    out = StageOutputService(str(tmp_path))
    rows = [{"window": 0, "raw": "Không có gì sai.", "readable": False}]
    out.write_relabel({"segments": 10, "windows": 1, "failed_windows": 0,
                       "unreadable_windows": 1, "proposed": 0, "changed": 0,
                       "replies": rows, "applied": [], "rejected": []})
    folder = tmp_path / "07_relabel"
    assert json.loads((folder / "replies.json").read_text(encoding="utf-8")) == rows
    assert "replies" not in json.loads((folder / "report.json").read_text(encoding="utf-8"))
    stats = json.loads((folder / "stats.json").read_text(encoding="utf-8"))
    assert stats["unreadable_windows"] == 1
    assert any("without any JSON" in w for w in stats["warnings"])


def test_running_out_of_tokens_while_thinking_warns_and_names_the_setting(tmp_path):
    out = StageOutputService(str(tmp_path))
    out.write_conversation_exports({"finder": {}, "cut_in_thought": 3, "unreadable": 3}, [])
    stats = json.loads((tmp_path / "09_conversation_exports" / "stats.json").read_text(encoding="utf-8"))
    assert any("max_new_tokens" in w and "conversation_selection" in w for w in stats["warnings"])

    out.write_relabel({"segments": 10, "windows": 2, "failed_windows": 0, "unreadable_windows": 2,
                       "cut_in_thought_windows": 2, "proposed": 0, "changed": 0, "thinking": True,
                       "replies": [], "applied": [], "rejected": []})
    stats = json.loads((tmp_path / "07_relabel" / "stats.json").read_text(encoding="utf-8"))
    assert any("models.relabel.max_new_tokens" in w for w in stats["warnings"])
    assert stats["thinking"] is True
