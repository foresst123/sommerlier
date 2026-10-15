"""Re-judging the model's earlier answers under looser settings, without asking it again."""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.conversation_export_service import ConversationExportService as Svc  # noqa: E402
from tests.test_conversation_review_flags import FinderLLM, _kw, _talk  # noqa: E402
from utils.checkpoint import CheckpointManager  # noqa: E402


class NoCallLLM(FinderLLM):
    """Fails the test if the model is asked anything."""
    calls = 0

    def generate_texts(self, *a, **k):
        NoCallLLM.calls += 1
        raise AssertionError("the model must not be asked")


def _first_run(**settings):
    svc = Svc(FinderLLM(), **settings)
    return svc.select(_talk(), **_kw())


def test_a_window_answer_is_kept_in_the_report_so_it_can_be_replayed():
    sel = _first_run(max_seconds=100)                       # the 233 s region is too long for this one
    assert not sel.final and sel.report["judged_rejected"].get("duration")
    replay = Svc.replay_from_payloads([{"report": sel.report}])
    assert len(replay) == sel.report["windows"] and all(isinstance(v, str) for v in replay.values())


def test_looser_settings_rejudge_the_same_answers_without_asking_the_model_again():
    sel = _first_run(max_seconds=100)
    replay = Svc.replay_from_payloads([{"report": sel.report}])
    NoCallLLM.calls = 0
    loose = Svc(NoCallLLM(), max_seconds=300, replay_old_replies=True)
    again = loose.select(_talk(), replay=replay, **_kw())
    assert NoCallLLM.calls == 0
    assert again.final, again.report
    assert again.report["windows_replayed"] == again.report["windows"]


def test_replay_is_off_unless_asked_for():
    sel = _first_run()
    replay = Svc.replay_from_payloads([{"report": sel.report}])
    calls = []

    class Counting(FinderLLM):
        def generate_texts(self, *a, **k):
            calls.append(1)
            return super().generate_texts(*a, **k)

    Svc(Counting()).select(_talk(), replay=replay, **_kw())
    assert calls                                            # default: the model is asked as before


def test_a_window_without_an_earlier_answer_is_still_asked():
    sel = _first_run()
    replay = {("99999", "99999"): "[]"}                     # no window matches this key
    calls = []

    class Counting(FinderLLM):
        def generate_texts(self, *a, **k):
            calls.append(1)
            return super().generate_texts(*a, **k)

    out = Svc(Counting(), replay_old_replies=True).select(_talk(), replay=replay, **_kw())
    assert calls and out.report["windows_replayed"] == 0


def test_unreadable_or_cut_off_answers_are_never_replayed():
    rows = [{"first_index": "0", "last_index": "9", "raw": "x", "readable": False},
            {"first_index": "1", "last_index": "9", "raw": "y", "readable": True, "cut_in_thought": True},
            {"first_index": "2", "last_index": "9", "raw": "[]", "readable": True},
            {"first_index": "3", "last_index": "9", "raw": None, "readable": True}]
    assert Svc.replay_from_payloads([{"report": {"replies": rows}}]) == {("2", "9"): "[]"}


def test_the_newest_answer_for_a_window_wins():
    new = {"report": {"replies": [{"first_index": "0", "last_index": "9", "raw": "new", "readable": True}]}}
    old = {"report": {"replies": [{"first_index": "0", "last_index": "9", "raw": "old", "readable": True}]}}
    assert Svc.replay_from_payloads([new, old])[("0", "9")] == "new"


def test_load_siblings_returns_other_namespaces_newest_first(tmp_path):
    cp = CheckpointManager(str(tmp_path), "job")
    for name, payload in (("ns-old", {"v": 1}), ("ns-new", {"v": 2}), ("ns-now", {"v": 3})):
        cp.namespaces["conversation_selection"] = name
        cp.save("conversation_selection", payload)
        time.sleep(0.05)                                    # distinct modification times
    cp.namespaces["conversation_selection"] = "ns-now"
    assert [p["v"] for p in cp.load_siblings("conversation_selection")] == [2, 1]
