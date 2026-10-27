"""The separation_plan stop point: a CPU-only process builds and checkpoints the
overlap window plan, and the GPU separation run restores it instead of rebuilding.
Run: python -m pytest tests/test_separation_plan.py -q
"""
import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas.segment import Segment
from services.pipeline_service import PipelineService
from services.separation_service import SeparationService
from utils.checkpoint import CheckpointManager
from utils.steps import will_run
from test_separation_prefetch import FakeTSE, _audio, _dialogue

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pipe(svc):
    pipe = PipelineService(*(None,) * 8)
    pipe.separation_svc = svc
    return pipe


def _ck(tmp_path):
    ck = CheckpointManager(str(tmp_path / "cache"), "job")
    ck.namespaces = {"separation": "policy-1", "processed_audio": "v1-audio"}
    return ck


def _counting(monkeypatch):
    calls = {"n": 0}
    original = SeparationService._build_overlap_plan

    def counted(self, segments, audio, overlap_threshold=0.1):
        calls["n"] += 1
        return original(self, segments, audio, overlap_threshold)

    monkeypatch.setattr(SeparationService, "_build_overlap_plan", counted)
    return calls


def _svc(**kw):
    return SeparationService(FakeTSE(), logger=None, **kw)


# --- wiring ----------------------------------------------------------------

def test_stop_order_puts_the_plan_between_pre_separation_and_separation():
    order = PipelineService._STOP_ORDER
    assert order.index("denoise_pre_separation") < order.index("separation_plan")
    assert order.index("separation_plan") == order.index("separation") - 1


def test_cli_accepts_the_plan_stop_point():
    with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as fh:
        src = fh.read()
    choices = re.search(r'"--stop_after".*?choices=\[(.*?)\]', src, re.S).group(1)
    assert '"separation_plan"' in choices


def test_plan_stop_loads_no_separation_or_asr_model():
    args = SimpleNamespace(stop_after="separation_plan", step_separation=True,
                           step_asr=True, step_diarization=True)
    assert will_run(args, "separation") is False
    assert will_run(args, "asr") is False
    assert will_run(args, "diarization") is True


def test_other_stops_are_unchanged_by_the_plan_stop():
    assert will_run(SimpleNamespace(stop_after="separation"), "separation") is True
    assert will_run(SimpleNamespace(stop_after=None), "asr") is True


# --- save / restore ----------------------------------------------------------

def test_saved_plan_is_restored_without_rebuilding(tmp_path, monkeypatch):
    calls = _counting(monkeypatch)
    segs, audio = _dialogue(), _audio()

    # Process 1 (CPU, --stop_after separation_plan)
    assert _pipe(_svc())._save_separation_plan(_ck(tmp_path), segs, audio, "b.mp3")
    assert calls["n"] == 1

    # Process 2 (GPU separation): a fresh service restores the plan.
    svc = _svc()
    assert _pipe(svc)._restore_separation_plan(_ck(tmp_path), segs, "b.mp3")
    plan = svc._take_prefetched_plan("b.mp3")
    assert plan is not None and len(plan.speech) == len(segs)
    assert calls["n"] == 1, "the saved plan is used; no second build"


def test_changed_segments_do_not_reuse_the_saved_plan(tmp_path, monkeypatch):
    _counting(monkeypatch)
    segs, audio = _dialogue(), _audio()
    _pipe(_svc())._save_separation_plan(_ck(tmp_path), segs, audio, "b.mp3")

    changed = _dialogue()
    changed[1] = Segment(index="00002", start=14.0, end=14.6, speaker="SPEAKER_01")
    assert _pipe(_svc())._restore_separation_plan(_ck(tmp_path), changed, "b.mp3") is False


def test_changed_audio_namespace_does_not_reuse_the_saved_plan(tmp_path, monkeypatch):
    _counting(monkeypatch)
    segs, audio = _dialogue(), _audio()
    _pipe(_svc())._save_separation_plan(_ck(tmp_path), segs, audio, "b.mp3")

    ck = _ck(tmp_path)
    ck.namespaces["processed_audio"] = "v1-other-audio"
    assert _pipe(_svc())._restore_separation_plan(ck, segs, "b.mp3") is False


def test_a_plan_from_another_window_policy_is_not_reused(tmp_path, monkeypatch):
    _counting(monkeypatch)
    segs, audio = _dialogue(), _audio()
    _pipe(_svc())._save_separation_plan(_ck(tmp_path), segs, audio, "b.mp3")

    ck = _ck(tmp_path)
    ck.namespaces["separation"] = "policy-2"
    svc = _svc()
    svc.checkpoint_version = "policy-2"
    assert _pipe(svc)._restore_separation_plan(ck, segs, "b.mp3") is False


def test_no_saved_plan_means_restore_is_a_miss(tmp_path):
    assert _pipe(_svc())._restore_separation_plan(
        _ck(tmp_path), _dialogue(), "none.mp3") is False


def test_lazy_plan_is_not_saved(tmp_path, monkeypatch):
    _counting(monkeypatch)
    svc = _svc(performance_config={"lazy_windows": True})
    ck = _ck(tmp_path)
    assert _pipe(svc)._save_separation_plan(ck, _dialogue(), _audio(), "lazy.mp3") is False
    assert not ck.exists("separation_plan", verify_digest=False)


def test_restore_logs_the_saved_plan_use(tmp_path, monkeypatch):
    _counting(monkeypatch)
    segs, audio = _dialogue(), _audio()
    _pipe(_svc())._save_separation_plan(_ck(tmp_path), segs, audio, "b.mp3")

    lines = []

    class _Log:
        def info(self, msg, *a, **k):
            lines.append(msg)

        warning = debug = error = info

    svc = _svc()
    svc.logger = _Log()
    assert _pipe(svc)._restore_separation_plan(_ck(tmp_path), segs, "b.mp3")
    assert "[TSE] using saved separation plan for b.mp3" in lines


def test_run_stops_at_the_plan_point_after_saving_it(tmp_path, monkeypatch):
    """run() with --stop_after separation_plan: diarization comes from the
    checkpoint, the plan is built and saved, separation is never run, and the
    manifest records the stop."""
    import json
    import types
    import numpy as np
    import services.pipeline_service as pipeline_module

    calls = _counting(monkeypatch)
    segs = _dialogue()
    diarization = types.SimpleNamespace(segments=segs, method="cached", raw_segments=None)
    saved = {}
    inner = CheckpointManager(str(tmp_path / "cache"), "job")

    class _Checkpoint:
        def __init__(self, *_a, **_k):
            self.namespaces = {}

        def exists(self, stage, **kw):
            return stage == "diarization" or inner.exists(stage, **kw)

        def load_if_committed(self, stage, **_k):
            return diarization if stage == "diarization" else None

        def load(self, stage, **kw):
            return inner.load(stage, **kw)

        def load_array(self, stage):
            return np.zeros(len(_audio().waveform), dtype=np.float32)

        def save(self, stage, data, **kw):
            inner.namespaces = dict(self.namespaces)
            saved[stage] = data
            return inner.save(stage, data, **kw)

        def save_array(self, *_a, **_k):
            pass

    monkeypatch.setattr(pipeline_module, "CheckpointManager", _Checkpoint)

    class _NoSeparation(SeparationService):
        def process_overlaps(self, *_a, **_k):
            raise AssertionError("separation must not run at the plan stop")

    pipe = PipelineService(
        audio_svc=SimpleNamespace(load_audio=lambda *a, **k: None),
        diarization_svc=SimpleNamespace(timeline=None),
        separation_svc=_NoSeparation(FakeTSE(), logger=None),
        music_svc=SimpleNamespace(),
        asr_svc=SimpleNamespace(),
        caption_svc=SimpleNamespace(),
        refinement_svc=SimpleNamespace(),
        export_svc=SimpleNamespace(),
    )
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"input")
    args = SimpleNamespace(
        stop_after="separation_plan", save_path=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"), job_id="job", env="test",
        keep_models=False, no_stage_output=False,
        step_music_analysis=False, step_cut_music=False, step_diarization=True,
        step_separation=True, bss=True,
    )

    assert pipe.run(args, {}, str(audio_path)) is None

    assert calls["n"] == 1, "the plan is built once, on this CPU process"
    assert saved["separation_plan"]["key"].startswith("plan-v1-")
    manifest = json.loads((tmp_path / "out" / "episode" / "manifest.json").read_text())
    assert manifest["metadata"]["stopped_after"] == "separation_plan"
