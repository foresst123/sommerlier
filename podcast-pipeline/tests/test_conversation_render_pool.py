"""Conversation excerpts are cut and written on a bounded pool, in a fixed order."""

import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.conversation_export_service import (
    ConversationExportRun, ConversationExportService, _Plan, _Render)


def _service(**kwargs):
    return ConversationExportService(SimpleNamespace(), **kwargs)


def _cand(i, score):
    return SimpleNamespace(score=score, tier="S", noise=None, i=i)


def _drive(svc, tmp_path, count=6, fail=()):
    cands = [_cand(i, 90 - i) for i in range(count)]
    kept = {id(c): SimpleNamespace(topic=f"t{c.i}", semantic=5, trimmed=False) for c in cands}
    threads, active, peak = set(), [0], [0]
    lock = threading.Lock()

    def plan(finder, cand, item, *a, **k):
        p = _Plan(cand, item, None, [], 0.0, "S")
        p.render = _Render(0, 8, np.zeros(8, np.float32))
        svc._keep_render(p.render)
        return p

    def write(finder, plan_, directory, folder, export_id, *a, **k):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            threads.add(threading.current_thread().name)
        time.sleep(0.05)
        svc._release_render(plan_)
        with lock:
            active[0] -= 1
        if plan_.cand.i in fail:
            raise OSError("disk full")
        return {"verification": {"ok": True, "checks": {}}, "overlap_seconds": 0.0,
                "files": {"mixture": "mixture.wav"}, "duration": 1.0,
                "source_start": 0.0, "source_end": 1.0, "speaker_ids": {}}

    svc._plan, svc._write_conversation = plan, write
    result, report = ConversationExportRun(), {"verification_failed": 0, "tiers": {}}
    svc._plan_and_write(None, cands, kept, result, report, np.zeros(8, np.float32), 8000,
                        str(tmp_path), "ep 01", None, None)
    return result, report, threads, peak[0]


def test_pool_gives_the_same_rows_in_the_same_order_as_sequential(tmp_path):
    seq = _drive(_service(render_workers=1), tmp_path)
    par = _drive(_service(render_workers=4, reuse_render=True), tmp_path)
    assert seq[0].exports == par[0].exports
    assert seq[1] == par[1]
    assert [e["id"] for e in par[0].exports][:2] == ["ep_01_conversation_000001",
                                                    "ep_01_conversation_000002"]
    assert seq[3] == 1 and 1 < par[3] <= 4


def test_a_failed_write_is_skipped_without_shifting_other_numbers(tmp_path):
    result, report, _, _ = _drive(_service(render_workers=3), tmp_path, fail={2})
    ids = [e["id"] for e in result.exports]
    assert len(ids) == 5 and "ep_01_conversation_000003" not in ids
    assert report["exported"] == 5


def test_cut_cache_is_bounded_and_released(tmp_path):
    svc = _service(reuse_render=True)
    svc.RENDER_CACHE_BYTES = 40           # room for one 8-sample cut (32 bytes)
    first, second = _Render(0, 8, np.zeros(8, np.float32)), _Render(0, 8, np.zeros(8, np.float32))
    assert svc._keep_render(first) and not svc._keep_render(second)
    svc._release_render(_Plan(None, None, None, [], 0.0, "S", render=first))
    assert svc._render_bytes == 0 and svc._keep_render(second)


def test_reuse_is_off_by_default():
    svc = _service()
    assert not svc._keep_render(_Render(0, 8, np.zeros(8, np.float32)))
    assert svc.render_workers == 1
