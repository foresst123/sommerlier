import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("line_status", Path(__file__).resolve().parents[1] / "tools" / "line_status.py")
ls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ls)


def test_table_shows_handovers_failures_and_log_tails(tmp_path):
    out = tmp_path
    (out / "variants_manifest.json").write_text(json.dumps({"variants": [{"id": "none"}, {"id": "l_p2"}, {"id": "m_p1"}]}))
    for vid in ("none", "l_p2", "m_p1"):
        (out / vid).mkdir()
    for stage in ("denoise", "diarization", "denoise_pre_separation", "separation", "denoise_pre_asr"):
        (out / "none" / f".ready_{stage}").write_text("{}")
    (out / "none" / ".done").write_text("{}")
    (out / "l_p2" / ".ready_denoise").write_text("{}")
    (out / "l_p2" / ".ready_diarization").write_text("{}")
    (out / "l_p2" / ".ready_denoise_pre_separation").write_text("{}")
    (out / "l_p2" / ".failed_separation").write_text(json.dumps({"reason": "Sidon worker did not start"}))
    logs = out / "_resident" / "logs"
    logs.mkdir(parents=True)
    (logs / "4_p4_separation.log").write_text("noise\nwaiting for .ready_denoise_pre_separation of: m_p1\n\n")
    (out / "lanes").mkdir()                                   # not a variant

    text = ls.render(out)
    rows = {l.split()[0]: l for l in text.splitlines() if l.split() and l.split()[0] in ("none", "l_p2", "m_p1")}
    assert rows["none"].count("ok") == 6                      # every hand-over incl. asr
    assert "FAIL" in rows["l_p2"] and rows["l_p2"].count("ok") == 3
    assert rows["m_p1"].count("ok") == 0 and "lanes" not in text.split("failures")[0]
    assert "l_p2: separation: Sidon worker did not start" in text
    assert "waiting for .ready_denoise_pre_separation of: m_p1" in text
