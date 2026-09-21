"""How PipelineService runs the relabel pass: decide once, re-apply on every entry.

run() is re-entered once per stage and reloads the refinement checkpoint, which
predates the relabel pass. So what is remembered is the decision (index ->
speaker), and it is applied again to whatever run() loaded.

Run:  python -m pytest tests/test_speaker_relabel_pipeline.py -q     (from podcast-pipeline/)
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.transcript import TranscriptSegment
from services.pipeline_service import PipelineService
from services.speaker_relabel_service import SpeakerRelabelService
from utils.checkpoint import CheckpointManager
from utils.excise import TimelineMap

A, B = "SPEAKER_00", "SPEAKER_01"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _seg(i, speaker):
    return TranscriptSegment(
        index=f"{i:05d}", start=i * 5.0, end=i * 5.0 + 4.0, speaker=speaker,
        text=f"câu số {i}", text_whisper="w", text_phowhisper="p", text_qwen3="q",
        language="vi", bs_roformer=False, bss=False)


def _conversation(n=40, wrong=(7,)):
    segs = []
    for i in range(n):
        speaker = A if i % 2 == 0 else B
        if i in wrong:
            speaker = B if speaker == A else A
        segs.append(_seg(i, speaker))
    return segs


class _LLM:
    model_name = "fake/model"
    batch_size = 100
    max_batch_tokens = 0

    def __init__(self, reply="[]", loaded=True, fail=False):
        self.reply, self.loaded, self.fail = reply, loaded, fail
        self.calls = 0

    def ensure_loaded(self):
        return self.loaded

    def count_tokens(self, text):
        return len(text.split())

    def generate_texts(self, system_prompt, user_messages, max_new_tokens=512,
                       use_prefix=False, labels=None):
        self.calls += 1
        if self.fail:
            return False, []
        return True, [self.reply for _ in user_messages]


class _StageOut:
    def __init__(self):
        self.reports = []

    def write_relabel(self, report):
        self.reports.append(report)


def _pipeline(llm, **cfg):
    pipe = PipelineService.__new__(PipelineService)
    pipe.relabel_svc = SpeakerRelabelService(llm, window_tokens=100000, **cfg)
    pipe.timeline = TimelineMap()
    pipe.noise_track = None
    pipe.logger = None
    return pipe


def _fix_seven():
    return json.dumps([{"i": "00007", "speaker": B, "conf": 0.9, "why": "trả lời"}])


def _speech(segs):
    out = []
    for s in segs:
        obj = type("S", (), {})()
        obj.index, obj.speaker = s.index, s.speaker
        out.append(obj)
    return out


def test_a_decision_is_applied_saved_and_reported(tmp_path):
    segs = _conversation()
    speech = _speech(segs)
    out = _StageOut()
    ckpt = CheckpointManager(str(tmp_path), "job")

    _pipeline(_LLM(_fix_seven()))._relabel_speakers(ckpt, out, segs, speech)

    assert segs[7].speaker == B and segs[7].speaker_original == A
    assert speech[7].speaker == B
    assert out.reports and out.reports[0]["changed"] == 1
    assert ckpt.exists("speaker_relabel")


def test_the_next_entry_reapplies_the_decision_without_asking_the_model_again(tmp_path):
    ckpt = CheckpointManager(str(tmp_path), "job")
    _pipeline(_LLM(_fix_seven()))._relabel_speakers(
        ckpt, _StageOut(), _conversation(), _speech(_conversation()))

    # run() reloads the refinement checkpoint, which has the diarizer's labels.
    fresh = _conversation()
    speech = _speech(fresh)
    llm = _LLM(_fix_seven())
    out = _StageOut()
    _pipeline(llm)._relabel_speakers(ckpt, out, fresh, speech)

    assert llm.calls == 0 and out.reports == []
    assert fresh[7].speaker == B and speech[7].speaker == B


def test_only_speaker_and_its_trace_differ_after_a_reapplied_decision(tmp_path):
    ckpt = CheckpointManager(str(tmp_path), "job")
    _pipeline(_LLM(_fix_seven()))._relabel_speakers(
        ckpt, _StageOut(), _conversation(), _speech(_conversation()))
    fresh = _conversation()
    before = copy.deepcopy(fresh)
    _pipeline(_LLM())._relabel_speakers(ckpt, _StageOut(), fresh, _speech(fresh))
    for old, new in zip(before, fresh):
        a, b = dict(old.__dict__), dict(new.__dict__)
        for key in ("speaker", "speaker_original"):
            a.pop(key), b.pop(key)
        assert a == b


def test_a_skipped_pass_is_not_remembered_as_done(tmp_path):
    for llm in (_LLM(loaded=False),):
        ckpt = CheckpointManager(str(tmp_path / "skip"), "job")
        _pipeline(llm)._relabel_speakers(ckpt, _StageOut(), _conversation(), None)
        assert not ckpt.exists("speaker_relabel")

    ckpt = CheckpointManager(str(tmp_path / "single"), "job")
    single = [_seg(i, A) for i in range(6)]
    _pipeline(_LLM())._relabel_speakers(ckpt, _StageOut(), single, None)
    assert not ckpt.exists("speaker_relabel")


def test_a_pass_where_no_window_was_answered_is_tried_again_next_time(tmp_path):
    ckpt = CheckpointManager(str(tmp_path), "job")
    out = _StageOut()
    segs = _conversation()
    _pipeline(_LLM(fail=True))._relabel_speakers(ckpt, out, segs, None)
    assert not ckpt.exists("speaker_relabel")
    assert out.reports and out.reports[0]["failed_windows"] >= 1
    assert all(s.speaker_original is None for s in segs)


def test_a_different_model_or_prompt_does_not_reuse_the_decision(tmp_path):
    ckpt = CheckpointManager(str(tmp_path), "job")
    _pipeline(_LLM(_fix_seven()))._relabel_speakers(
        ckpt, _StageOut(), _conversation(), None)

    other = _LLM(_fix_seven())
    other.model_name = "another/model"
    fresh = _conversation()
    _pipeline(other)._relabel_speakers(
        CheckpointManager(str(tmp_path), "job"), _StageOut(), fresh, None)
    assert other.calls == 1


def test_a_missing_speech_segment_list_is_fine(tmp_path):
    segs = _conversation()
    _pipeline(_LLM(_fix_seven()))._relabel_speakers(
        CheckpointManager(str(tmp_path), "job"), _StageOut(), segs, None)
    assert segs[7].speaker == B


# --- the gate ----------------------------------------------------------------

def test_the_step_is_opt_in_in_run_and_keeps_the_llm_loaded_for_it():
    """run() is too large to drive here; pin the two lines that matter."""
    source = open(os.path.join(ROOT, "services", "pipeline_service.py"),
                  encoding="utf-8").read()
    assert 'opt_in_step_enabled(args, "speaker_relabel")' in source
    assert 'self.step_enabled(args, "speaker_relabel")' not in source
    assert 'self.step_enabled(args, "refinement") or relabel_on' in source


def test_both_profiles_ship_the_pass_off_until_it_has_been_measured():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for name, profile in config["environments"].items():
        assert profile["steps"]["speaker_relabel"] is False, name
        assert profile["steps"]["dialogue_clips"] is False, name
        assert "relabel" in profile["models"], name


def test_the_shipped_relabel_settings_are_accepted_by_the_service():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        config = json.load(fh)
    for name, profile in config["environments"].items():
        SpeakerRelabelService(_LLM(), **profile["models"]["relabel"])
