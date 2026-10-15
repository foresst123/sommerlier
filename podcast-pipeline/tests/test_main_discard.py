"""main.py parses sys.argv at import time, so it cannot be imported here --
these tests read its source text instead. See tests/test_stage_scheduling.py
for the same style applied to pipeline_service.py."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_discard_partial_drops_a_stale_prefetched_plan():
    """A discarded file's diarization checkpoint is gone, so its next
    attempt recomputes diarization from scratch. Any window plan already
    prefetched for the attempt that just failed must be dropped, or a retry
    could be handed a plan built before the checkpoint it depended on was
    wiped."""
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    body = src[src.index("def _discard_partial"):src.index("def main()")]
    assert "drop_prefetched_plan" in body
