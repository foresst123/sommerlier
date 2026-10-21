"""Integrity of the shipped tools/variants.example.json.
Run: PYTHONPATH=. python -m pytest tests/test_variants_integrity.py -q"""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / "tools" / "variants.example.json"

spec = importlib.util.spec_from_file_location("run_variants_integrity", ROOT / "tools" / "run_variants.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)

from models.denoise import BACKENDS, parse_chain  # noqa: E402
from models.denoise.pseudo import is_pseudo, parse_pseudo  # noqa: E402

FIRST_12 = [("none", "baseline"), ("dfn3", "masking"), ("mf2se48", "masking"), ("mpsenet", "masking"),
            ("mpsenet_dns", "masking"), ("bsrnn26", "masking"), ("dfn3>mf2se48", "chains"),
            ("mf2se48>dfn3", "chains"), ("bsrnn26>dfn3", "chains"), ("dbs_tiger", "analysis_only"),
            ("mambattention", "analysis_only"), ("dbs_tiger>dfn3", "analysis_only")]
GROUPS = {"baseline", "masking", "chains", "analysis_only", "ctrl_music", "pipe_music", "pipe_reverb",
          "pipe_crowd", "pipe_order", "pipe_stack", "pipe_blend", "analysis_pipes"}
ANALYSIS = "ANALYSIS ONLY: do not export this output as training audio before a listening test"


@pytest.fixture(scope="module")
def entries():
    return json.loads(PLAN.read_text(encoding="utf-8"))["variants"]


@pytest.fixture(scope="module")
def loaded():
    return rv.load_variants(str(PLAN))


def test_first_twelve_entries_are_unchanged(entries):
    assert [(e["chain"], e["group"]) for e in entries[:12]] == FIRST_12
    assert all("set" not in e and "tag" not in e for e in entries[:12])


def test_total_and_groups(entries):
    assert len(entries) >= 55
    assert {e["group"] for e in entries} <= GROUPS
    assert all(e.get("label") and e.get("group") for e in entries)
    assert all(e.get("note") for e in entries[1:])


def test_chains_parse_and_tokens_are_known(entries):
    for e in entries:
        toks = parse_chain(e["chain"])
        for i, t in enumerate(toks):
            if is_pseudo(t):
                assert i > 0 and parse_pseudo(t) is not None, e["chain"]
            else:
                assert t in BACKENDS, (e["chain"], t)
        if toks:
            assert any(not is_pseudo(t) for t in toks), e["chain"]


def test_ids_unique_and_filename_safe(entries, loaded):
    ids = [v["id"] for v in loaded]
    assert len(ids) == len(set(ids))
    assert len(ids) == len(entries)  # baseline entry is the file's own first entry
    assert all(rv.ID_RE.match(i) for i in ids)


def test_baseline_exactly_once(loaded):
    base = [v for v in loaded if not v["chain"] and not v["set"] and not v["music_separator"] and not v["extra_args"]]
    assert [v["id"] for v in base] == ["none"]


def test_rf_vocals_chains_turn_the_music_stage_off(entries):
    n = 0
    for e in entries:
        if any(t.startswith("rf_vocals") for t in parse_chain(e["chain"])):
            n += 1
            if "double-music" in (e.get("tag") or ""):
                continue
            assert e.get("set", {}).get("steps.music_removal") is False, e["chain"]
            assert e.get("tag") == "music-stage-off", e["chain"]
    assert n >= 20


def test_analysis_entries_carry_the_note(entries):
    for e in entries:
        if e["group"] in ("analysis_only", "analysis_pipes"):
            assert e["note"].startswith(ANALYSIS), e["chain"]
            assert "analysis only" in e["label"].lower()


def test_crowd_entries_warn_unverified(entries):
    for e in entries:
        if "rf_crowd" in parse_chain(e["chain"]) and e["group"] == "pipe_crowd":
            assert "UNVERIFIED" in e["note"], e["chain"]


def test_list_matches_the_file(tmp_path, entries):
    audio = tmp_path / "audio"
    audio.mkdir()
    for n in ("a.wav", "b.wav"):
        (audio / n).write_bytes(b"x")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "run_variants.py"), "--audio-dir", str(audio),
                        "--variants", str(PLAN), "--out-root", str(tmp_path / "o"), "--list"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    n = len(entries)
    assert f"variants: {n} (0 skipped, {n} to run)" in r.stdout
    assert "files: 2" in r.stdout
    assert f"variant-file runs: {n * 2}" in r.stdout
    assert "ASSUMING 20 min" in r.stdout
    for e in entries:
        assert re.search(re.escape(e["chain"]) if e["chain"] != "none" else "none", r.stdout)
