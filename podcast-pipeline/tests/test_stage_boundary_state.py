"""State that has to survive from one run() to the next.

Stage-major execution re-enters run() once per stage, so anything that assumes
"once per file" is wrong. These pin the two places that assumption broke.

Run:  python -m pytest tests/test_stage_boundary_state.py -q   (from podcast-pipeline/)
"""
import ast
import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


# --- separation report ------------------------------------------------------

def test_the_separation_report_is_checkpointed_where_it_is_produced():
    """write_report() runs during export, which under stage-major execution is
    a later run() whose reset_stats() has already cleared the counters -- a real
    batch wrote stats={} while stats.json held spliced=3."""
    src = _source("services/pipeline_service.py")
    save = src.index('checkpoint.save("separation_report"')
    assert 'checkpoint.save("separation", speech_segments)' in src[:save], (
        "the report must be captured in the same branch that computed it")


def test_the_export_reads_the_checkpointed_report():
    src = _source("services/pipeline_service.py")
    call = src[src.index("self.separation_svc.write_report("):][:250]
    assert 'checkpoint.load("separation_report")' in call


def test_write_report_falls_back_to_live_counters():
    """The single-file path runs separation moments before export, so passing
    nothing must keep working."""
    tree = ast.parse(_source("services/separation_service.py"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "write_report")

    args = [a.arg for a in fn.args.args]
    assert "payload" in args
    assert fn.args.defaults, "payload must be optional"


# --- refinement checkpoint --------------------------------------------------

def test_refinement_is_checkpointed_like_every_other_stage():
    """It is the most expensive stage -- 12 minutes for 200 segments on a T4 --
    so a retry that recomputes it pays for half the run again."""
    src = _source("services/pipeline_service.py")
    assert 'checkpoint.load_if_committed("refinement")' in src
    assert 'checkpoint.save("refinement", transcripts)' in src


def test_a_restored_refinement_does_not_rewrite_its_stage_output():
    """changes.json needs the pre-refinement text, which a checkpoint restore
    does not have -- so the write belongs in the compute branch only."""
    src = _source("services/pipeline_service.py")
    load = src.index('checkpoint.load_if_committed("refinement")')
    save = src.index('checkpoint.save("refinement", transcripts)')
    write = src.index("stage_out.write_refinement(")

    assert load < save < write, (
        "write_refinement must sit after the save, inside the else branch")


def test_cached_diarization_stop_does_not_fall_through_to_later_stages(
        monkeypatch, tmp_path):
    """A diarization-stage retry must not start separation or the LLM.

    Fresh diarization has its own asynchronous return path.  This pins the
    checkpoint-hit path, which previously fell through all the way to
    refinement and started vLLM during the diarization pass.
    """
    import services.pipeline_service as pipeline_module
    from services.pipeline_service import PipelineService

    diarization = types.SimpleNamespace(
        segments=[types.SimpleNamespace(
            index="00000", start=0.0, end=0.1, speaker="1")],
        method="cached",
        raw_segments=None,
    )

    class _Checkpoint:
        def __init__(self, *_args, **_kwargs):
            self.namespaces = {}

        def exists(self, stage, **_kwargs):
            return stage == "diarization"

        def load_if_committed(self, stage, **_kwargs):
            return diarization if stage == "diarization" else None

        def load_array(self, stage):
            assert stage == "processed_audio"
            return np.zeros(2400, dtype=np.float32)

        def save(self, *_args, **_kwargs):
            raise AssertionError("a checkpointed diarization stop must not save later stages")

        def save_array(self, *_args, **_kwargs):
            raise AssertionError("the cached waveform must be reused")

    class _Service:
        checkpoint_version = None

        def reset_stats(self):
            pass

        def process_overlaps(self, *_args, **_kwargs):
            raise AssertionError("separation must not run")

        def passthrough(self, *_args, **_kwargs):
            raise AssertionError("separation passthrough must not run")

        def refine(self, *_args, **_kwargs):
            raise AssertionError("refinement/vLLM must not run")

    class _AudioService:
        def load_audio(self, *_args, **_kwargs):
            raise AssertionError("the cached waveform must avoid decoding")

    monkeypatch.setattr(pipeline_module, "CheckpointManager", _Checkpoint)
    later_stage = _Service()
    pipe = PipelineService(
        audio_svc=_AudioService(),
        diarization_svc=types.SimpleNamespace(timeline=None),
        separation_svc=later_stage,
        music_svc=types.SimpleNamespace(),
        asr_svc=types.SimpleNamespace(),
        caption_svc=types.SimpleNamespace(),
        refinement_svc=later_stage,
        export_svc=types.SimpleNamespace(),
    )
    audio_path = tmp_path / "episode.wav"
    audio_path.write_bytes(b"cached input")
    output_root = tmp_path / "output"
    args = types.SimpleNamespace(
        stop_after="diarization",
        save_path=str(output_root),
        cache_dir=str(tmp_path / "cache"),
        job_id="job",
        env="test",
        keep_models=False,
        no_stage_output=False,
        step_music_analysis=False,
        step_cut_music=False,
        step_diarization=True,
    )

    assert pipe.run(args, {}, str(audio_path)) is None

    manifest = json.loads(
        (output_root / "episode" / "manifest.json").read_text())
    assert manifest["metadata"]["audio_file"] == "episode.wav"
    assert manifest["metadata"]["stopped_after"] == "diarization"
