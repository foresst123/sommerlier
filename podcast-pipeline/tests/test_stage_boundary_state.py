"""State that has to survive from one run() to the next.

Stage-major execution re-enters run() once per stage, so anything that assumes
"once per file" is wrong. These pin the two places that assumption broke.

Run:  python -m pytest tests/test_stage_boundary_state.py -q   (from podcast-pipeline/)
"""
import ast
import os
import sys

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
    assert 'checkpoint.exists("refinement")' in src
    assert 'checkpoint.save("refinement", transcripts)' in src


def test_a_restored_refinement_does_not_rewrite_its_stage_output():
    """changes.json needs the pre-refinement text, which a checkpoint restore
    does not have -- so the write belongs in the compute branch only."""
    src = _source("services/pipeline_service.py")
    load = src.index('checkpoint.load("refinement")')
    save = src.index('checkpoint.save("refinement", transcripts)')
    write = src.index("stage_out.write_refinement(")

    assert load < save < write, (
        "write_refinement must sit after the save, inside the else branch")
