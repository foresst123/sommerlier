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


def test_per_file_cells_partial_failed_and_noop(tmp_path):
    out = tmp_path
    (out / "variants_manifest.json").write_text(json.dumps({"variants": [
        {"id": "none", "chain": [], "chain_pre_separation": [], "chain_pre_asr": [], "set": {}},
        {"id": "p3", "chain": [], "chain_pre_separation": [], "chain_pre_asr": ["dfn3"], "set": {}}]}))
    for vid in ("none", "p3"):
        (out / vid).mkdir()
    inp = out / "_resident" / "input_20260101_000000_1"
    inp.mkdir(parents=True)
    for n in ("a", "b", "c"):
        (inp / f"{n}.wav").write_bytes(b"x")
    for n in ("a", "b", "c"):
        (out / "none" / f".ready_diarization.{n}").write_text("{}")
    for n in ("a", "b"):
        (out / "none" / f".ready_separation.{n}").write_text("{}")
    (out / "p3" / ".ready_separation.a").write_text("{}")
    (out / "p3" / ".failed_separation.b").write_text(json.dumps({"reason": "Sidon OOM"}))
    text = ls.render(out)
    lines = {l.split()[0]: l for l in text.splitlines() if l.split() and l.split()[0] in ("none", "p3")}
    cols = lambda line: [line[9 + i * 16: 9 + (i + 1) * 16].split(" ")[0] for i in range(6)]
    none = cols(lines["none"])
    assert none[0] == "-"                                   # music off, no chain: denoise is a no-op
    assert none[1] == "ok" and none[2] == "-" and none[3] == "2/3" and none[4] == "-"
    rows = lines
    assert "FAIL" in rows["p3"] and "p3: separation b: Sidon OOM" in text


def test_hidden_folders_are_not_variants(tmp_path):
    for d in ("none", ".ipynb_checkpoints", ".manifest_abc123", "lanes", "_resident"):
        (tmp_path / d).mkdir()
    assert ls.variant_ids(tmp_path) == ["none"]
