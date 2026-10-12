"""persona_profile wiring in PipelineService, with a fake service (no GPU, no models).

Run:  python -m pytest tests/test_persona_profile_pipeline.py -q     (from podcast-pipeline/)
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.pipeline_service import PipelineService
from services.stage_output_service import StageOutputService

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Checkpoint:
    def __init__(self):
        self.namespaces, self.store = {}, {}

    def _key(self, name):
        return (name, self.namespaces.get(name))

    def load_if_committed(self, name, **kw):
        return self.store.get(self._key(name))

    def save(self, name, payload, **kw):
        self.store[self._key(name)] = payload


class FakePersona:
    llm_enabled = True

    def __init__(self, result=None, boom=False):
        self.result, self.calls, self.boom = result, [], boom

    def prepare(self, segments, timeline, noise=None):
        return list(segments)

    def source_digest(self, prepared):
        return "d1"

    def checkpoint_namespace_for(self, digest):
        return f"persona-ns-{digest}"

    def profile(self, prepared, recording_id):
        self.calls.append(recording_id)
        if self.boom:
            raise ValueError("boom")
        return self.result


def _result(complete=True, skipped=None):
    return types.SimpleNamespace(profile={"recording_id": "ep", "speakers": {}},
                                 stats={"warnings": []}, complete=complete, skipped=skipped)


def _pipe(svc):
    pipe = PipelineService(*(None,) * 8, persona_svc=svc)
    pipe.timeline = []
    return pipe


def _segs():
    return [types.SimpleNamespace(index="0", speaker="A", text="x", start=0.0, end=1.0)]


def _profile_exists(tmp_path):
    return (tmp_path / "07c_persona_profile" / "profile.json").exists()


def test_compute_checkpoints_and_writes_both_files(tmp_path):
    svc, ckpt = FakePersona(_result()), _Checkpoint()
    out = StageOutputService(str(tmp_path))
    _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, out, _segs(), "/a/ep.mp3",
                                 allow_compute=True)
    assert svc.calls == ["ep"]
    assert ("persona_profile", "persona-ns-d1") in ckpt.store
    assert ckpt.store[("persona_profile", "persona-ns-d1")]["profile"]["recording_id"] == "ep"
    assert _profile_exists(tmp_path) and (tmp_path / "07c_persona_profile" / "stats.json").exists()


def test_an_incomplete_result_is_written_but_not_remembered(tmp_path):
    svc, ckpt = FakePersona(_result(complete=False)), _Checkpoint()
    out = StageOutputService(str(tmp_path))
    _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, out, _segs(), "/a/ep.mp3")
    assert not ckpt.store and _profile_exists(tmp_path)


def test_a_later_pass_loads_the_checkpoint_and_never_calls_the_service(tmp_path):
    svc, ckpt = FakePersona(_result()), _Checkpoint()
    ckpt.namespaces["persona_profile"] = "persona-ns-d1"
    ckpt.store[("persona_profile", "persona-ns-d1")] = {
        "profile": {"recording_id": "ep"}, "stats": {"warnings": []}}
    out = StageOutputService(str(tmp_path))
    _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, out, _segs(), "/a/ep.mp3",
                                 allow_compute=False)
    assert svc.calls == [] and _profile_exists(tmp_path)


def test_no_checkpoint_and_no_compute_writes_nothing_and_skips_the_llm(tmp_path):
    svc, ckpt = FakePersona(_result()), _Checkpoint()
    out = StageOutputService(str(tmp_path))
    _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, out, _segs(), "/a/ep.mp3",
                                 allow_compute=False)
    assert svc.calls == [] and not _profile_exists(tmp_path)
    assert ckpt.namespaces["persona_profile"] == "persona-ns-d1"      # still selected


def test_empty_or_none_transcripts_write_nothing(tmp_path):
    for transcripts in ([], None):
        svc, ckpt = FakePersona(_result()), _Checkpoint()
        out = StageOutputService(str(tmp_path))
        _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, out, transcripts, "/a/ep.mp3")
        assert svc.calls == [] and not ckpt.store and not _profile_exists(tmp_path)


def test_a_skipped_result_is_not_written_or_saved(tmp_path):
    svc, ckpt = FakePersona(_result(complete=False, skipped="no_segments")), _Checkpoint()
    out = StageOutputService(str(tmp_path))
    _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, out, _segs(), "/a/ep.mp3")
    assert not ckpt.store and not _profile_exists(tmp_path)


def test_an_exception_is_logged_and_swallowed(tmp_path):
    svc, ckpt = FakePersona(boom=True), _Checkpoint()
    out = StageOutputService(str(tmp_path))
    logged = []
    pipe = _pipe(svc)
    pipe.logger = types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None,
        error=lambda m, *a, **k: logged.append(m), exception=lambda m, *a, **k: logged.append(m))
    pipe._profile_personas(types.SimpleNamespace(), ckpt, out, _segs(), "/a/ep.mp3")
    assert logged and not ckpt.store and not _profile_exists(tmp_path)


def test_the_stage_does_not_touch_the_real_transcripts(tmp_path):
    svc, ckpt = FakePersona(_result()), _Checkpoint()
    segs = _segs()
    before = dict(vars(segs[0]))
    _pipe(svc)._profile_personas(types.SimpleNamespace(), ckpt, StageOutputService(str(tmp_path)),
                                 segs, "/a/ep.mp3")
    assert vars(segs[0]) == before


def _run_body():
    with open(os.path.join(ROOT, "services", "pipeline_service.py"), encoding="utf-8") as fh:
        src = fh.read()
    return src[src.index("    def run(self, args"):]


def test_run_order_review_then_persona_then_alignment():
    body = _run_body()
    review = body.index('opt_in_step_enabled(args, "transcript_review")')
    persona = body.index('opt_in_step_enabled(args, "persona_profile")')
    call = body.index("self._profile_personas(")
    stop = body.index('if getattr(args, "stop_after", None) == "persona_profile"')
    align = body.index("# 7c. Forced alignment")
    selection = body.index("Conversation judging is its own LLM stage")
    assert review < persona < call < stop < align < selection


def test_stop_point_writes_the_manifest_and_releases_the_llm():
    body = _run_body()
    stop = body.index('if getattr(args, "stop_after", None) == "persona_profile"')
    block = body[stop:stop + 700]
    assert '_release_llm_after(args, "persona_profile")' in block
    assert '"stopped_after": "persona_profile"' in block
    assert "return transcripts" in block


def test_only_the_persona_pass_or_a_single_file_run_computes():
    body = _run_body()
    call = body.index("self._profile_personas(")
    window = body[call - 700:call + 300]
    assert 'getattr(args, "postprocess_only", False)' in window
    assert 'getattr(args, "batch_continues", None) is None' in window
    assert 'getattr(args, "stop_after", None) == "persona_profile"' in window
    assert "allow_compute=allow_compute" in window


def test_the_final_llm_release_counts_the_persona_stage():
    body = _run_body()
    tail = body[body.rindex("or review_on"):]
    assert "persona_on" in tail[:120]


def test_main_accepts_persona_profile_as_a_stop_point():
    with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as fh:
        assert '"persona_profile"' in fh.read()


def _main_src():
    with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as fh:
        return fh.read()


def test_main_builds_the_persona_service_only_when_the_step_is_on():
    src = _main_src()
    gate = src.index('opt_in_step_enabled(args, "persona_profile")')
    assert "PersonaProfileService(" in src[gate:gate + 700]
    assert "refinement_svc, logger=logger" in src[gate:gate + 700]
    assert "persona_svc = None" in src[gate - 200:gate]
    assert "persona_svc=persona_svc" in src
    assert src.index("persona_svc = None") > src.index("review_svc = TranscriptReviewService(")


def test_persona_config_has_exactly_the_three_keys():
    import json
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    a100 = cfg["environments"]["a100"]
    assert a100["steps"]["persona_profile"] is False
    assert set(a100["models"]["persona_profile"]) == {"llm_enabled", "thinking", "max_new_tokens"}
    for env in cfg["environments"].values():
        persona = env.get("models", {}).get("persona_profile")
        if persona is not None:
            assert set(persona) <= {"llm_enabled", "thinking", "max_new_tokens"}
