"""The LLM trace has to reach every Qwen call of the five LLM stages.

run() opens `llm_trace.trace_to(stage, output_dir)` around each stage; the fake services
below read `llm_trace.current()` from the thread that calls them, which is what the writer
(Track A) captures at submit/generate time.

Run:  python -m pytest tests/test_llm_trace_wiring.py -q   (from podcast-pipeline/)
"""
import os
import sys
import threading
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import llm_trace
import services.pipeline_service as pipeline_module
from services.pipeline_service import PipelineService


def _seg(index):
    return types.SimpleNamespace(index=index, speaker="A", text=f"t{index}",
                                 start=float(index), end=index + 0.9, review_flags=None)


class _Checkpoint:
    def __init__(self, *_args, **_kwargs):
        self.namespaces = {}
        self.store = {}

    def exists(self, stage, **_kw):
        return stage == "diarization"

    def load_if_committed(self, stage, **_kw):
        if stage == "diarization":
            return types.SimpleNamespace(
                segments=[types.SimpleNamespace(index="0", start=0.0, end=0.5, speaker="A")],
                method="cached", raw_segments=None)
        if stage == "asr":
            return [_seg(0), _seg(1)]
        return self.store.get((stage, self.namespaces.get(stage)))

    def load_array(self, stage):
        return np.zeros(2400, dtype=np.float32)

    def save(self, stage, payload, **_kw):
        self.store[(stage, self.namespaces.get(stage))] = payload

    def save_array(self, *_a, **_kw):
        pass


class _Recorder:
    """Records the trace seen by whichever thread calls it."""

    def __init__(self):
        self.seen = []

    def note(self, name):
        self.seen.append((name, llm_trace.current()))


class _Refine(_Recorder):
    def refine(self, transcripts, prompt=None):
        self.note("refine")
        return transcripts

    def reset_stats(self):
        pass

    def unload(self):
        pass


class _Result:
    def __init__(self, **kw):
        self.mapping, self.flags_by_index = {}, {}
        self.windows, self.failed_windows, self.skipped = 1, 0, None
        self.complete, self.profile, self.stats = True, {}, {}
        self.__dict__.update(kw)

    def to_report(self):
        return {}


class _Relabel(_Recorder):
    checkpoint_namespace = "relabel-ns"

    def relabel(self, transcripts):
        self.note("relabel")
        return _Result()


class _Review(_Recorder):
    checkpoint_namespace = "review-ns"

    def review(self, transcripts):
        self.note("review")
        return _Result()


class _Persona(_Recorder):
    def prepare(self, transcripts, timeline, noise):
        return transcripts

    def source_digest(self, prepared):
        return "d"

    def checkpoint_namespace_for(self, digest):
        return "persona-ns"

    def profile(self, prepared, recording_id):
        self.note("persona")
        return _Result()


class _Selection:
    report = {}

    def to_checkpoint(self):
        return {"x": 1}


class _Export(_Recorder):
    def checkpoint_namespace_for(self, transcripts):
        return "selection-ns"

    def select(self, transcripts, **_kw):
        self.note("select")
        return _Selection()


class _StageOut:
    def __getattr__(self, name):
        return lambda *a, **k: None


class _Separation:
    checkpoint_version = None

    def reset_stats(self):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None

    def passthrough(self, segments, audio):
        return list(segments)


def _drive(tmp_path, monkeypatch, stop_after, llm_trace_flag=None):
    monkeypatch.setattr(pipeline_module, "CheckpointManager", _Checkpoint)
    monkeypatch.setattr(pipeline_module, "StageOutputService",
                        lambda *a, **k: _StageOut())
    refine, relabel, review = _Refine(), _Relabel(), _Review()
    persona, export = _Persona(), _Export()
    pipe = PipelineService(
        audio_svc=types.SimpleNamespace(),
        diarization_svc=types.SimpleNamespace(timeline=None),
        separation_svc=_Separation(), music_svc=types.SimpleNamespace(),
        asr_svc=types.SimpleNamespace(), caption_svc=types.SimpleNamespace(),
        refinement_svc=refine, export_svc=types.SimpleNamespace(),
        relabel_svc=relabel, review_svc=review, persona_svc=persona,
        conversation_export_svc=export)
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"x")
    args = types.SimpleNamespace(
        stop_after=stop_after, save_path=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"), job_id="job", env="test",
        keep_models=True, no_stage_output=False,
        step_music_analysis=False, step_cut_music=False, step_diarization=True,
        step_separation=False, step_asr=True, step_captioning=False,
        step_refinement=True, step_speaker_relabel=True, step_transcript_review=True,
        step_persona_profile=True, step_conversation_exports=True)
    if llm_trace_flag is not None:
        args.llm_trace = llm_trace_flag
    pipe.run(args, {}, str(audio))
    return pipe, (refine, relabel, review, persona, export), str(tmp_path / "out" / "episode")


@pytest.mark.parametrize("stage,index", [
    ("refinement", 0), ("speaker_relabel", 1), ("transcript_review", 2),
    ("persona_profile", 3), ("conversation_selection", 4)])
def test_each_llm_stage_runs_under_its_own_trace(tmp_path, monkeypatch, stage, index):
    _, fakes, output_dir = _drive(tmp_path, monkeypatch, stage)
    (_, trace), = fakes[index].seen
    assert trace == llm_trace.Trace(stage=stage, output_dir=output_dir)


def test_trace_is_closed_after_the_stage(tmp_path, monkeypatch):
    _drive(tmp_path, monkeypatch, "refinement")
    assert llm_trace.current() is None


def test_single_file_run_traces_every_stage_with_its_own_name(tmp_path, monkeypatch):
    _, fakes, output_dir = _drive(tmp_path, monkeypatch, "conversation_selection")
    names = [fake.seen[0][1].stage for fake in fakes]
    assert names == ["refinement", "speaker_relabel", "transcript_review",
                     "persona_profile", "conversation_selection"]
    assert {fake.seen[0][1].output_dir for fake in fakes} == {output_dir}


def test_llm_trace_false_turns_the_trace_off(tmp_path, monkeypatch):
    _, fakes, _ = _drive(tmp_path, monkeypatch, "conversation_selection", llm_trace_flag=False)
    assert all(fake.seen and fake.seen[0][1] is None for fake in fakes)


def test_two_files_in_two_threads_keep_their_own_output_dir(tmp_path, monkeypatch):
    """Stage-major batch runs run() for several files at once on pool threads."""
    dirs = {}
    barrier = threading.Barrier(2)

    def one(name):
        with llm_trace.trace_to("refinement", str(tmp_path / name)):
            barrier.wait(timeout=5)
            dirs[name] = llm_trace.current().output_dir

    threads = [threading.Thread(target=one, args=(n,)) for n in ("a", "b")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert dirs == {"a": str(tmp_path / "a"), "b": str(tmp_path / "b")}


# --- hand-overs to other threads ------------------------------------------------

def test_ask_in_batches_pool_threads_see_the_callers_trace(tmp_path):
    from utils.llm_batches import ask_in_batches

    seen = []

    class _Llm:
        parallel_windows = True
        replica_count = 3

        def generate_texts(self, system_prompt, chunk, max_new_tokens, **kw):
            seen.append(llm_trace.current())
            return True, ["r"] * len(chunk)

    with llm_trace.trace_to("speaker_relabel", str(tmp_path)):
        ask_in_batches(_Llm(), "sys", [f"m{i}" for i in range(6)], per_call=2,
                       max_new_tokens=8, label="t")
    assert len(seen) == 3
    assert all(t == llm_trace.Trace("speaker_relabel", str(tmp_path)) for t in seen)


def _refinement_service():
    from services.diarization_refinement_service import DiarizationRefinementService
    svc = DiarizationRefinementService.__new__(DiarizationRefinementService)
    svc.logger = None
    svc._worker = None
    svc._local_executor = None
    return svc


def test_local_refinement_executor_carries_the_trace(tmp_path):
    svc = _refinement_service()
    svc._ensure_worker_started = lambda: False
    seen = []
    svc.generate_texts = lambda *a, **k: (seen.append(llm_trace.current()), (True, ["x"]))[1]
    with llm_trace.trace_to("refinement", str(tmp_path)):
        svc.submit_texts("sys", ["m"]).result(timeout=5)
    assert seen == [llm_trace.Trace("refinement", str(tmp_path))]


def test_message_building_pool_carries_the_trace(tmp_path):
    from services import diarization_refinement_service as module
    svc = _refinement_service()
    svc.ensure_loaded = lambda: True
    svc._activate_cpu_threads = lambda: 4
    svc.shared_queue = False
    svc.progress_interval = 0
    svc._worker = object()
    svc._refine_sequential = lambda pending, prompt, progress: (0, 0)
    svc.rejected = 0
    seen = []

    def build(segments, i):
        seen.append(llm_trace.current())
        return "m"

    svc._build_user_message = build
    segs = [types.SimpleNamespace(index=i, start=0.0, end=5.0, text_whisper="một hai ba bốn năm sáu",
                                  text_phowhisper="", text_qwen3="") for i in range(12)]
    with llm_trace.trace_to("refinement", str(tmp_path)):
        svc.refine(segs)
    assert len(seen) == 12
    assert all(t == llm_trace.Trace("refinement", str(tmp_path)) for t in seen)
