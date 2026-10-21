"""End to end, no GPU and no real model: run_variants -> build_compare_data -> make_compare_page.

The stub main stands in for main.py: it writes the stage layout the pipeline writes
(01_music, 01b_denoise via the REAL DenoiseService and the REAL identity/fake_hp
backends, 02_diarization, 05_asr) and nothing else.
Run: PYTHONPATH=. python -m pytest tests/test_denoise_compare_e2e.py -q
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


NO_EXPORT = 'outputs.clean_two_channel_dir=""'   # run_variants keeps comparison runs out of the shared export folder


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rv = _load("run_variants_e2e", "tools/run_variants.py")
page = _load("make_compare_page_e2e", "tools/make_compare_page.py")

SR = 24000
STUB = '''
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, %(root)r)
import soundfile as sf
from services.denoise_service import DenoiseService
p = argparse.ArgumentParser()
for f in ("--audio_dir", "--save_path", "--job_id", "--env", "--stop_after", "--denoise", "--cache_dir"):
    p.add_argument(f)
p.add_argument("--music_separator")
p.add_argument("--set", action="append")
a = p.parse_args()
Path(a.save_path).mkdir(parents=True, exist_ok=True)
with open(Path(a.save_path) / "calls.json", "w") as _f:
    json.dump({"argv": sys.argv[1:], "set": a.set or [], "music_separator": a.music_separator}, _f)


class Audio:  # what DenoiseService needs from AudioData
    pass


for src in sorted(Path(a.audio_dir).iterdir()):
    d = Path(a.save_path) / src.stem
    wav, sr = sf.read(str(src), dtype="float32")
    (d / "01_music").mkdir(parents=True, exist_ok=True)
    sf.write(str(d / "01_music" / "after_music.wav"), wav, sr, subtype="PCM_16")
    if a.denoise:
        au = Audio(); au.waveform = wav.copy(); au.sample_rate = sr
        info = DenoiseService().apply(au, a.denoise)
        (d / "01b_denoise").mkdir(exist_ok=True)
        sf.write(str(d / "01b_denoise" / "after_denoise.wav"), au.waveform, sr, subtype="PCM_16")
        (d / "01b_denoise" / "denoise.json").write_text(json.dumps(info))
    spk = 1 if a.denoise == "fake_hp" else 0
    (d / "02_diarization").mkdir(exist_ok=True)
    (d / "02_diarization" / "segments.json").write_text(json.dumps([
        {"index": "0", "start": 0.0, "end": 2.0, "speaker": "S0"},
        {"index": "1", "start": 2.0, "end": 4.0, "speaker": "S%%d" %% spk}]))
    (d / "05_asr").mkdir(exist_ok=True)
    word = "ca phe" if a.denoise else "cà phê"
    (d / "05_asr" / "transcripts.json").write_text(json.dumps([
        {"index": "0", "start": 0.0, "end": 2.0, "speaker": "S0", "text": "xin chào các bạn"},
        {"index": "1", "start": 2.0, "end": 4.0, "speaker": "S1", "text": "uống " + word + " nhé"}],
        ensure_ascii=False))
'''


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    audio = tmp / "audio"
    audio.mkdir()
    rng = np.random.RandomState(0)
    t = np.arange(SR * 4) / SR
    for name in ("a", "b"):
        x = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.02 * rng.randn(len(t)) + 0.1 * np.sin(2 * np.pi * 50 * t)
        sf.write(str(audio / f"{name}.wav"), x.astype("float32"), SR, subtype="PCM_16")
    stub = tmp / "stub_main.py"
    stub.write_text(STUB % {"root": str(ROOT)})
    var = tmp / "variants.json"
    var.write_text(json.dumps({"variants": [{"chain": "fake_hp"}, {"chain": "identity>fake_hp"}]}))
    out = tmp / "out"
    code = rv.main(["--audio-dir", str(audio), "--variants", str(var), "--out-root", str(out),
                    "--main", str(stub), "--env", "a100", "--stop-after", "asr",
                    "--python", sys.executable])
    assert code == 0
    return tmp, out


def test_runner_produced_the_manifest(run):
    _, out = run
    m = json.loads((out / "variants_manifest.json").read_text())
    ids = [v["id"] for v in m["variants"]]
    assert ids == ["none", "fake_hp", "identity+fake_hp"]
    assert all(v["status"] == "ok" and set(v["files"]) == {"a", "b"} for v in m["variants"])
    assert not (Path(m["variants"][0]["files"]["a"]) / "01b_denoise").exists()
    den = json.loads((Path(m["variants"][2]["files"]["a"]) / "01b_denoise" / "denoise.json").read_text())
    assert den["chain"] == ["identity", "fake_hp"] and den["chain_id"] == "identity+fake_hp"
    assert den["errors"] == [] and den["samples"] == SR * 4


def test_compare_data_and_page(run):
    tmp, out = run
    from tools.compare_lib.build import build_compare_data, write_compare_data
    data = build_compare_data(str(out / "variants_manifest.json"), embed=False)
    cd = tmp / "compare_data.json"
    write_compare_data(data, str(cd))
    assert [v["id"] for v in data["variants"]] == ["none", "fake_hp", "identity+fake_hp"]
    f = next(x for x in data["files"] if x["name"] == "a")
    assert f["text_diff"]["fake_hp"]["wer_vs_baseline"] > 0
    assert f["text_diff"]["none"]["wer_vs_baseline"] == 0
    assert f["speakers"]["fake_hp"]["disagree_seconds"] > 0
    assert f["metrics"]["fake_hp"]["audio"]["length_equal"] is True
    assert f["metrics"]["fake_hp"]["audio"]["corr_with_raw"] < 1.0
    assert "cà phê" in f["text"]["none"]["full_text"]
    html_path = tmp / "compare.html"
    page.main([str(cd), "-o", str(html_path), "--embed", "clips"])
    text = html_path.read_text(encoding="utf-8")
    for tab in ("btn-text", "btn-speakers", "btn-metrics"):
        assert tab in text
    assert "cà phê" in text or "c\\u00e0 ph\\u00ea" in text
    assert "http://" not in re.sub(r"http://www\.w3\.org/\d+/svg", "", text)


# ---- contracts between the pieces, with the real code on both sides ------------
class _Audio:
    def __init__(self, wav, sr):
        self.waveform, self.sample_rate = wav, sr


def _tone(sr=24000, sec=3.0):
    t = np.arange(int(sr * sec)) / sr
    return (0.3 * np.sin(2 * np.pi * 400 * t) + 0.1 * np.sin(2 * np.pi * 40 * t)).astype(np.float32)


def test_denoise_service_with_the_real_backends():
    from services.denoise_service import DenoiseService
    wav = _tone()
    au = _Audio(wav.copy(), 24000)
    info = DenoiseService().apply(au, "identity")
    assert info["errors"] == [] and info["chain_id"] == "identity" and info["lag_samples"] == 0
    assert np.array_equal(au.waveform, wav)
    au = _Audio(wav.copy(), 24000)
    info = DenoiseService().apply(au, " identity > fake_hp ", {"fake_hp": {"cutoff_hz": 200}})
    assert info["errors"] == [] and info["chain"] == ["identity", "fake_hp"]
    assert len(au.waveform) == len(wav) and not np.array_equal(au.waveform, wav)
    assert info["noise_floor_db_after"] is not None and info["lag_samples"] == 0
    assert info["corr_with_input"] > 0.9


def test_denoise_service_survives_a_string_config_and_unknown_backend():
    from services.denoise_service import DenoiseService
    au = _Audio(_tone(), 24000)
    assert DenoiseService().apply(au, "identity", "none")["errors"] == []
    au = _Audio(_tone(), 24000)
    info = DenoiseService().apply(au, "no_such_backend")
    assert info["errors"] and len(au.waveform) == len(_tone())


@pytest.mark.parametrize("spec", ["none", "", None, "dfn3", "dfn3>mf2se48", " dfn3 > mf2se48 ",
                                  "DFN3>MF2SE48", ["dfn3", "mpsenet"]])
def test_run_variants_chain_ids_agree_with_the_models_package(spec):
    from models.denoise import chain_id, parse_chain
    assert rv._denoise_helpers()[1](parse_chain(spec)) == chain_id(spec)
    # the fallback used when models.denoise cannot be imported must agree too
    assert rv._fallback_chain_id(spec) == chain_id(spec)
    assert rv._fallback_parse_chain(spec) == parse_chain(spec)
    from services.denoise_service import chain_id_of
    assert chain_id_of(spec) == chain_id(spec)


def test_dry_run_commands_survive_a_shell(capsys, tmp_path):
    import shlex
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    rv.main(["--files", str(audio), "--out-root", str(tmp_path / "o"), "--dry-run",
             "--variants", str(_variants(tmp_path))])
    for line in capsys.readouterr().out.splitlines():
        shlex.split(line.split("] ", 1)[1])
    

def _variants(tmp):
    p = tmp / "v.json"
    p.write_text(json.dumps({"variants": [{"chain": "dfn3>mf2se48"}]}))
    return p


# ---- the shipped plan: tools/variants.example.json with --only-available -------
PLAN = ROOT / "tools" / "variants.example.json"


def _run_plan(tmp, extra=()):
    audio = tmp / "audio"
    audio.mkdir()
    t = np.arange(SR * 4) / SR
    sf.write(str(audio / "a.wav"), (0.3 * np.sin(2 * np.pi * 300 * t)).astype("float32"), SR, subtype="PCM_16")
    stub = tmp / "stub_main.py"
    stub.write_text(STUB % {"root": str(ROOT)})
    out = tmp / "out"
    code = rv.main(["--audio-dir", str(audio), "--variants", str(PLAN), "--out-root", str(out),
                    "--main", str(stub), "--env", "a100", "--stop-after", "asr",
                    "--python", sys.executable, "--only-available", *extra])
    return code, out


def _page_from(tmp, out):
    from tools.compare_lib.build import build_compare_data, write_compare_data
    data = build_compare_data(str(out / "variants_manifest.json"), embed=False)
    cd = tmp / "compare_data.json"
    write_compare_data(data, str(cd))
    html_path = tmp / "compare.html"
    page.main([str(cd), "-o", str(html_path), "--embed", "clips"])
    return data, html_path.read_text(encoding="utf-8")


def test_shipped_plan_skips_unavailable_backends_and_still_builds_a_page(tmp_path, monkeypatch, capsys):
    import models.denoise as md
    reasons = {"bsrnn26": "URGENT26_DIR is not set", "dbs_tiger": "DBS_TIGER_DIR is not set",
               "mambattention": "MAMBATTENTION_PYTHON is not set"}
    monkeypatch.setattr(md, "available_backends", lambda cfg=None: {
        k: ((False, reasons[k]) if k in reasons else (True, "")) for k in md.BACKENDS})
    code, out = _run_plan(tmp_path, ("--groups", "masking,chains,analysis_only"))
    assert code == 0
    console = capsys.readouterr().out
    for vid, why in (("bsrnn26", "URGENT26_DIR"), ("dbs_tiger", "DBS_TIGER_DIR"), ("mambattention", "MAMBATTENTION_PYTHON")):
        assert f"[{vid}] SKIPPED" in console and why in console
    m = json.loads((out / "variants_manifest.json").read_text())
    st = {v["id"]: v for v in m["variants"]}
    assert {k for k, v in st.items() if v["status"] == "skipped"} == {
        "bsrnn26", "bsrnn26+dfn3", "dbs_tiger", "dbs_tiger+dfn3", "mambattention"}
    assert all(st[k]["reason"] for k in st if st[k]["status"] == "skipped")
    # a runnable variant is ok, or degraded when its real backend cannot load in this environment
    assert {k for k, v in st.items() if v["status"] in ("ok", "degraded")} == {
        "none", "dfn3", "mf2se48", "mpsenet", "mpsenet_dns", "dfn3+mf2se48", "mf2se48+dfn3"}
    assert st["none"]["status"] == "ok"
    assert st["dfn3"]["group"] == "masking" and st["dfn3+mf2se48"]["group"] == "chains"
    assert "analysis" in st["dbs_tiger"]["note"].lower() and not (out / "dbs_tiger").exists()
    data, text = _page_from(tmp_path, out)
    row = {v["id"]: v for v in data["variants"]}
    assert row["dbs_tiger"]["status"] == "skipped" and "DBS_TIGER_DIR" in row["dbs_tiger"]["note"]
    assert row["dbs_tiger"]["group"] == "analysis_only"
    assert 'data-vid="dfn3+mf2se48"' in text and '<span class="grp">chains</span>' in text


def test_shipped_plan_with_the_real_availability_check(tmp_path):
    import models.denoise as md
    avail = md.available_backends()
    code, out = _run_plan(tmp_path, ("--groups", "analysis_only,masking"))
    assert code == 0
    m = json.loads((out / "variants_manifest.json").read_text())
    assert {v["group"] for v in m["variants"]} <= {None, "baseline", "masking", "analysis_only"}
    for v in m["variants"]:
        bad = [b for b in v["chain"] if not avail[b][0]]
        assert (v["status"] == "skipped") == bool(bad), v["id"]
        if bad:
            assert avail[bad[0]][1] in v["reason"]
    assert next(v for v in m["variants"] if v["id"] == "none")["status"] == "ok"
    assert not any(v["group"] == "chains" for v in m["variants"])
    _page_from(tmp_path, out)


# ---- the multi-stage plan: ctrl_music + pipe_blend with per-variant overrides ----
def _calls(out, vid):
    return json.loads((out / vid / "calls.json").read_text())


def test_ctrl_music_and_pipe_blend_with_the_real_availability_check(tmp_path):
    import models.denoise as md
    avail = md.available_backends()
    code, out = _run_plan(tmp_path, ("--groups", "ctrl_music,pipe_blend"))
    assert code == 0
    m = json.loads((out / "variants_manifest.json").read_text())
    st = {v["id"]: v for v in m["variants"]}
    assert {v["group"] for v in m["variants"]} == {None, "ctrl_music", "pipe_blend"}
    assert len(st) == 1 + 5 + 7
    from models.denoise.pseudo import backend_tokens
    for v in m["variants"]:
        bad = [b for b in backend_tokens(v["chain"]) if not avail[b][0]]
        assert (v["status"] == "skipped") == bool(bad), v["id"]
        for key in ("set", "tag", "group"):
            assert key in v, (v["id"], key)
    raw = st["none.music-removal-off"]
    assert raw["tag"] == "music-stage-off" and raw["set"] == {"steps.music_removal": False}
    if raw["status"] == "ok":
        assert "steps.music_removal=false" in _calls(out, raw["id"])["set"]
    _page_from(tmp_path, out)


def test_ctrl_music_and_pipe_blend_with_a_mocked_availability(tmp_path, monkeypatch):
    import models.denoise as md
    monkeypatch.setattr(md, "available_backends", lambda cfg=None: {
        k: ((False, "BSRNN_NOT_SET") if k in ("bsrnn26", "rf_vocals_3005") else (True, "")) for k in md.BACKENDS})
    code, out = _run_plan(tmp_path, ("--groups", "ctrl_music,pipe_blend"))
    assert code == 0
    m = json.loads((out / "variants_manifest.json").read_text())
    st = {v["id"]: v for v in m["variants"]}
    skipped = {k for k, v in st.items() if v["status"] == "skipped"}
    assert skipped == {"bsrnn26+blend50", "rf_vocals_3005+dfn3+blend50.music-removal-off",
                       "rf_vocals_3005+rf_dereverb+dfn3+blend30.music-removal-off"}
    assert all(st[k]["status"] in ("ok", "degraded") for k in st if k not in skipped)
    assert st["none"]["status"] == "ok"
    # blend steps are pseudo: dfn3>blend50 is runnable although no backend is called blend50
    assert st["dfn3+blend50"]["status"] in ("ok", "degraded") and not st["dfn3+blend50"].get("reason")
    c = _calls(out, "none.music-scope-spans")
    assert c["set"] == [NO_EXPORT, 'pipeline.music_scope="spans"'] and "--denoise" not in c["argv"]
    c = _calls(out, "none.music-removal-off")
    assert c["set"] == [NO_EXPORT, "steps.music_removal=false"]
    sep = next(v for v in m["variants"] if v["tag"] == "sep-kim")
    c = _calls(out, sep["id"])
    assert c["music_separator"] == "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt" and "--denoise" in c["argv"]
    c = _calls(out, "dfn3+mf2se48+blend50")
    assert c["set"] == [NO_EXPORT] and c["argv"][c["argv"].index("--denoise") + 1] == "dfn3>mf2se48>blend50"
    # the blend actually ran in the real DenoiseService (backends may fall back, errors recorded)
    den = json.loads((Path(st["dfn3+blend50"]["files"]["a"]) / "01b_denoise" / "denoise.json").read_text())
    assert den["chain"] == ["dfn3", "blend50"]
    data, text = _page_from(tmp_path, out)
    row = {v["id"]: v for v in data["variants"]}
    assert row["none.music-removal-off"]["tag"] == "music-stage-off"
    assert row["none.music-removal-off"]["set"] == {"steps.music_removal": False}
    assert "music-stage-off" in text and "scope-spans" in text


# ---- silent fallbacks must surface: degraded status, badges, banner ------------
FAULTY_STUB = STUB.replace(
    "info = DenoiseService().apply(au, a.denoise)",
    "info = ({'chain': ['broken'], 'chain_id': 'broken', 'errors': ['broken: model weights not found\\ntrace'], "
    "'corr_with_input': 1.0, 'lag_samples': 0, 'steps': [], 'blend': None} if a.denoise == 'broken' "
    "else DenoiseService().apply(au, a.denoise))").replace(
    '(d / "01b_denoise" / "denoise.json").write_text(json.dumps(info))',
    'None if a.denoise == "ghost" else (d / "01b_denoise" / "denoise.json").write_text(json.dumps(info))')


def test_degraded_variants_surface_in_manifest_data_and_page(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    t = np.arange(SR * 4) / SR
    for name in ("a", "b"):
        sf.write(str(audio / f"{name}.wav"), (0.3 * np.sin(2 * np.pi * 300 * t)).astype("float32"), SR, subtype="PCM_16")
    stub = tmp_path / "stub_main.py"
    stub.write_text(FAULTY_STUB % {"root": str(ROOT)})
    var = tmp_path / "variants.json"
    var.write_text(json.dumps({"variants": [{"chain": "fake_hp"}, {"chain": "broken"}, {"chain": "ghost"},
                                            {"chain": "identity"}]}))
    out = tmp_path / "out"
    args = ["--audio-dir", str(audio), "--variants", str(var), "--out-root", str(out),
            "--main", str(stub), "--env", "a100", "--stop-after", "asr", "--python", sys.executable]
    assert rv.main(args) == 0
    assert rv.main(args + ["--strict"]) == 3
    m = {v["id"]: v for v in json.loads((out / "variants_manifest.json").read_text())["variants"]}
    assert {k: v["status"] for k, v in m.items()} == {
        "none": "ok", "fake_hp": "ok", "broken": "degraded", "ghost": "degraded", "identity": "ok"}
    assert m["broken"]["denoise_report"]["a"]["verdict"] == "fallback"
    assert m["ghost"]["denoise_report"]["a"]["verdict"] == "missing"
    assert m["identity"]["denoise_report"]["a"]["verdict"] == "ok"
    assert not (out / "broken" / ".done").exists() and (out / "fake_hp" / ".done").exists()

    data, text = _page_from(tmp_path, out)
    row = {v["id"]: v for v in data["variants"]}
    assert row["broken"]["degraded"] and row["broken"]["degraded_files"] == 2 and row["broken"]["total_files"] == 2
    assert "model weights not found" in row["broken"]["degraded_reason"]
    assert row["ghost"]["degraded"] and not row["fake_hp"]["degraded"] and not row["none"]["degraded"]
    assert data["summary_excluded_degraded"]["broken"] == 2 and data["summary_n"]["broken"] == 2
    assert data["summary_excluded_degraded"]["fake_hp"] == 0
    f = next(x for x in data["files"] if x["name"] == "a")
    assert f["denoise"]["broken"]["verdict"] == "fallback" and f["denoise"]["fake_hp"]["verdict"] == "ok"
    assert text.count('class="badge bad"') == 2 and "fell back to unfiltered audio in 2/2 files" in text
    assert 'role="alert"' in text and "Degraded variants" in text
