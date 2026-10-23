import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import audit_clean_pool as acp  # noqa: E402


def _rec(name):
    segs = [{"start": 0.0, "end": 30.0, "speaker": "A"}, {"start": 10.0, "end": 10.5, "speaker": "B"},
            {"start": 40.0, "end": 41.0, "speaker": "B"}, {"start": 50.0, "end": 52.5, "speaker": "B"},
            {"start": 60.0, "end": 80.0, "speaker": "A", "has_music": True},
            {"start": 100.0, "end": 110.0, "speaker": "A"}, {"start": 120.0, "end": 125.0, "speaker": "B"}]
    return {"path": f"cache/{name}", "segments": lambda: segs}


def test_audit_counts_clean_time_and_layout_outcomes(tmp_path):
    tp = {"default_x": [{"overlap": o, "pattern": p, "first_len": 5.0, "second_len": o, "delta_db": 0.0, "t": 10.0 * i}
                        for i, (o, p) in enumerate(((0.5, "contained"), (2.0, "partial"), (3.0, "full"),
                                                    (0.1, "contained"), (0.4, "contained")))]}
    tp["default_x"][-1]["t"] = 0.9       # within 1 s of the 0.5 s pair at t = 0 -> a two-pair event
    s = acp.audit([_rec("default_x")], tp, str(tmp_path), windows=[3.0, 4.0])
    # A: 0-9.7, 10.8-30 and 100-110 (B's backchannel +- 0.3 s cut out), music excluded; B: 1 s + 2.5 s
    assert s["unique_clean_hours"]["0.25"] * 3600 == pytest_approx(9.7 + 19.2 + 1.0 + 2.5 + 10 + 5)
    assert s["minor_speaker_clean_hours"]["0.25"] * 3600 == pytest_approx(8.5)
    f4 = s["feasibility"]["4.0"]
    assert f4["templates"] == 5 and f4["drop:out_of_range"] == 1 and f4["drop:multi_overlap"] == 2
    assert f4.get("nominal", 0) + f4.get("fallback", 0) + sum(v for k, v in f4.items()
                                                              if k.startswith("drop:")) == 5
    assert f4.get("enroll_ok:holder", 0) + f4.get("enroll_ok:interrupter", 0) >= 1
    assert (tmp_path / "speakers.csv").exists() and (tmp_path / "feasibility.csv").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["recordings"] == 1


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=0.01)
