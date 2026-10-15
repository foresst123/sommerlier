"""Relabel asks for several files at once; the state one file leaves behind stays its own."""

import types

from services.pipeline_service import PipelineService
from services.speaker_relabel_service import RELABEL_SYSTEM_PROMPT, SpeakerRelabelService
from utils.batch import _stage_parallelism
from utils.performance_config import resolve


def _args(files):
    perf = resolve({"performance": {"enabled": True, "stages": {"refinement": {
        "relabel_files_in_flight": files}}}})
    return types.SimpleNamespace(performance_config=perf)


def test_relabel_runs_the_configured_number_of_files_at_once():
    assert _stage_parallelism(_args(8), "speaker_relabel") == 8


def test_relabel_stays_one_file_at_a_time_by_default():
    assert _stage_parallelism(types.SimpleNamespace(
        performance_config=resolve({"performance": {"enabled": True}})), "speaker_relabel") == 1


def test_relabel_stage_gets_a_view_with_its_own_timeline():
    service = PipelineService.__new__(PipelineService)
    service.diarization_svc = service.separation_svc = object()
    service.music_svc = service.refinement_svc = object()
    service.timeline = object()
    service.noise_track = object()
    view = service.parallel_stage_view("speaker_relabel")
    assert view is not service and view.timeline is not service.timeline


class _Llm:
    model_name = "fake"
    batch_size = 1
    max_batch_tokens = 0

    def count_tokens(self, text):
        return len(text) // 4

    def ensure_loaded(self):
        return True

    def generate_texts(self, system_prompt, messages, max_new_tokens, **kw):
        return True, ["[]" for _ in messages]


def test_first_replies_are_returned_not_kept_on_the_service():
    svc = SpeakerRelabelService(_Llm())
    result = types.SimpleNamespace(failed_windows=0, unreadable_windows=0)
    replies, firsts = svc._ask(["a", "b"], result)
    assert replies == firsts == ["[]", "[]"]
    assert not hasattr(svc, "_first_replies")


def test_prompt_tells_the_model_to_stay_short():
    assert "KHÔNG chép lại" in RELABEL_SYSTEM_PROMPT
    assert "MỘT lần" in RELABEL_SYSTEM_PROMPT
