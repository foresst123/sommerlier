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
a = p.parse_args()


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
