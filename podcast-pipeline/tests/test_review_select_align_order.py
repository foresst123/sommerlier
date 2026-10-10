"""Stage order: review -> select -> align the selected segments -> write.

PipelineService.run() needs every earlier stage's checkpoint, so these drive the
stage helpers with fake services (no GPU, no models) and read the order of
run()'s body from its source, the way test_conversation_exports_pipeline does.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.pipeline_service import PipelineService

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _seg(index):
    return types.SimpleNamespace(index=index, speaker="A", text=f"t{index}",
                                 start=float(index), end=index + 0.9, review_flags=None)


class _Checkpoint:
    def __init__(self):
        self.namespaces = {}
        self.store = {}

    def _key(self, name):
        return (name, self.namespaces.get(name))

    def load_if_committed(self, name, **kw):
        return self.store.get(self._key(name))

    def exists(self, name, **kw):
        return self._key(name) in self.store

    def save(self, name, payload, **kw):
        self.store[self._key(name)] = payload


class _StageOut:
    def __init__(self, log=None):
        self.log = log if log is not None else []

    def write_review(self, report):
        self.log.append("write_review")

    def write_word_alignment(self, transcripts, report):
        self.log.append("write_word_alignment")

    def write_conversation_exports(self, report, exports):
        self.log.append("write_conversation_exports")


class _ReviewResult:
    def __init__(self, flags, windows=1, failed=0, skipped=None):
        self.flags_by_index = flags
        self.windows, self.failed_windows, self.skipped = windows, failed, skipped

    def to_report(self):
        return {"windows": self.windows}


class _Review:
    checkpoint_namespace = "review-ns"

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def review(self, segments):
        self.calls += 1
        return self.result


def _pipeline(**services):
    pipe = PipelineService(*(None,) * 8, **services)
    pipe.timeline = []
    return pipe


# --- review checkpoint ---------------------------------------------------------------

def test_review_flags_are_applied_saved_and_reapplied_without_the_llm():
    flag = {"kind": "speaker_turn", "phrase": None, "conf": 0.9, "why": "x"}
    review = _Review(_ReviewResult({"0": [flag], "1": []}))
    pipe = _pipeline(review_svc=review)
    ckpt, out = _Checkpoint(), _StageOut()

    segs = [_seg(0), _seg(1)]
    pipe._review_transcript(ckpt, out, segs)
    assert review.calls == 1
    assert segs[0].review_flags == [flag] and segs[1].review_flags == []
    saved = ckpt.store[("transcript_review", "review-ns")]
    assert set(saved) == {"flags_by_index", "report"}

    fresh = [_seg(0), _seg(1)]              # resume: transcripts reloaded without flags
    pipe._review_transcript(ckpt, out, fresh)
    assert review.calls == 1                # no second LLM pass
    assert fresh[0].review_flags == [flag] and fresh[1].review_flags == []


def test_a_review_that_got_no_answer_is_not_remembered():
    review = _Review(_ReviewResult({}, windows=2, failed=2))
    pipe = _pipeline(review_svc=review)
    ckpt = _Checkpoint()
    pipe._review_transcript(ckpt, _StageOut(), [_seg(0)])
    assert not ckpt.store
    pipe._review_transcript(ckpt, _StageOut(), [_seg(0)])
    assert review.calls == 2

    skipped = _Review(_ReviewResult({}, windows=0, skipped="no segments"))
    pipe = _pipeline(review_svc=skipped)
    pipe._review_transcript(ckpt, _StageOut(), [_seg(0)])
    assert not ckpt.store


# --- select -> align subset -> write ----------------------------------------------------

class _Selection:
    def __init__(self, segs):
        self._segs = segs

    def segments_to_align(self):
        return self._segs


class _Export:
    def __init__(self, log, selected):
        self.log, self.selected = log, selected

    def select(self, transcripts, **kw):
        self.log.append("select")
        self.seen = list(transcripts)
        return _Selection(self.selected)

    def write(self, selection, **kw):
        self.log.append("write")
        return types.SimpleNamespace(report={}, exports=[])


class _Align:
    def __init__(self, log):
        self.log = log
        self.aligned = None

    def checkpoint_namespace_for(self, segments):
        return "align-" + ",".join(str(s.index) for s in segments)

    def align(self, segments, audio, speech_segments):
        self.log.append("align")
        self.aligned = list(segments)
        return types.SimpleNamespace(words_by_index={}, report={})

    def unload(self):
        self.log.append("unload")


def _audio():
    return types.SimpleNamespace(waveform=[0.0], sample_rate=24000)


def _args(**kw):
    return types.SimpleNamespace(keep_models=True, **kw)


def test_order_is_select_then_align_only_the_selection_then_write():
    log = []
    segs = [_seg(i) for i in range(5)]
    selected = segs[1:3]
    export, align = _Export(log, selected), _Align(log)
    pipe = _pipeline(conversation_export_svc=export, word_alignment_svc=align)
    ckpt, out = _Checkpoint(), _StageOut(log)

    pipe._export_conversation_exports(
        out, segs, _audio(), None, "/tmp/x", "/tmp/ep.mp3",
        align=lambda sel: pipe._align_selected(_args(), ckpt, out, sel, _audio(), None))

    assert log == ["select", "align", "write_word_alignment", "write",
                   "write_conversation_exports"]
    assert export.seen == segs
    assert align.aligned == selected                      # not the whole file
    assert ("word_alignment", "align-1,2") in ckpt.store


def test_a_cached_subset_alignment_is_reapplied_without_aligning():
    log = []
    selected = [_seg(1), _seg(2)]
    align = _Align(log)
    pipe = _pipeline(word_alignment_svc=align)
    ckpt = _Checkpoint()
    ckpt.store[("word_alignment", "align-1,2")] = {"words_by_index": {}, "report": {}}
    pipe._align_selected(_args(), ckpt, _StageOut(log), _Selection(selected), _audio(), None)
    assert "align" not in log


def test_nothing_selected_aligns_nothing():
    log = []
    align = _Align(log)
    pipe = _pipeline(word_alignment_svc=align)
    pipe._align_selected(_args(), _Checkpoint(), _StageOut(log), _Selection([]),
                         _audio(), None)
    assert log == []


def test_stop_before_write_returns_the_selection_after_alignment():
    log = []
    export = _Export(log, [_seg(0)])
    pipe = _pipeline(conversation_export_svc=export)
    sel = pipe._export_conversation_exports(
        _StageOut(log), [_seg(0)], _audio(), None, "/tmp/x", "/tmp/ep.mp3",
        align=lambda s: log.append("align"), stop_before_write=True)
    assert log == ["select", "align"] and isinstance(sel, _Selection)


# --- run() body and stop points ----------------------------------------------------------

def _source():
    with open(os.path.join(ROOT, "services", "pipeline_service.py"), encoding="utf-8") as fh:
        return fh.read()


def test_run_order_relabel_review_whole_file_alignment_then_exports():
    src = _source()
    body = src[src.index("    def run(self, args"):]
    relabel = body.index('opt_in_step_enabled(args, "speaker_relabel")')
    review = body.index('opt_in_step_enabled(args, "transcript_review")')
    alignment = body.index('opt_in_step_enabled(args, "word_alignment")')
    exports = body.index('opt_in_step_enabled(args, "conversation_exports")')
    assert relabel < review < alignment < exports
    # The whole-file alignment only runs when exports are not planned at all --
    # including the final post-process pass, whose exports flag is forced off.
    assert "alignment_on and not exports_planned" in body
    planned = body[body.index("exports_planned = ("):body.index("if alignment_on and not exports_planned")]
    assert "postprocess_only" not in planned
    assert body.index("self._review_transcript(") < body.index("self._export_conversation_exports(")


def test_stop_after_transcript_review_is_a_stop_point_between_relabel_and_alignment():
    order = PipelineService._STOP_ORDER
    assert order.index("speaker_relabel") < order.index("transcript_review") \
        < order.index("word_alignment") < order.index("conversation_exports")
    args = types.SimpleNamespace(stop_after="transcript_review")
    assert PipelineService._reaches(args, "transcript_review")
    assert not PipelineService._reaches(args, "word_alignment")
    body = _source()
    stop = body.index('getattr(args, "stop_after", None) == "transcript_review"')
    assert stop < body.index('opt_in_step_enabled(args, "word_alignment")', stop)


def test_main_accepts_the_stop_point_and_the_a100_profile_turns_review_on():
    import json
    with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as fh:
        assert '"transcript_review"' in fh.read()
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        a100 = json.load(fh)["environments"]["a100"]
    assert a100["steps"]["transcript_review"] is True
    assert a100["steps"]["speaker_relabel"] is False
    assert a100["models"]["transcript_review"]["window_tokens"] == 32000


# --- LLM residency -----------------------------------------------------------------------

def test_the_resident_llm_is_kept_from_refinement_into_review():
    pipe = _pipeline(review_svc=object())
    perf = {"enabled": True, "stages": {"refinement": {"keep_llm_across_relabel": True}}}
    args = types.SimpleNamespace(performance_config=perf, batch_continues=True,
                                 step_transcript_review=True)
    assert pipe._keep_llm_for_relabel(args)
    assert not _pipeline()._keep_llm_for_relabel(args)
    args.step_transcript_review = False
    assert not pipe._keep_llm_for_relabel(args)


def test_a_review_with_an_unreadable_window_is_not_remembered():
    result = _ReviewResult({"0": []}, windows=2, failed=0)
    result.unreadable_windows = 1
    review = _Review(result)
    ckpt = _Checkpoint()
    _pipeline(review_svc=review)._review_transcript(ckpt, _StageOut(), [_seg(0)])
    assert not ckpt.store


def test_the_resident_llm_is_kept_from_review_into_selection():
    pipe = _pipeline(review_svc=object(), conversation_export_svc=object())
    perf = {"enabled": True, "stages": {"refinement": {"keep_llm_across_relabel": True}}}
    args = types.SimpleNamespace(performance_config=perf, batch_continues=True,
                                 step_transcript_review=True, step_conversation_exports=True,
                                 keep_models=False)
    assert pipe._next_llm_step_after(args, "transcript_review") == "conversation_exports"
    unloaded = []
    pipe.refinement_svc = types.SimpleNamespace(unload=lambda: unloaded.append(1))
    pipe._defer_or_run = lambda fn: fn()
    pipe._release_llm_after(args, "transcript_review")
    assert not unloaded
    args.batch_continues = False
    pipe._release_llm_after(args, "transcript_review")
    assert unloaded
