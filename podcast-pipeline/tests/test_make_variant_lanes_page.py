import base64
import json
import os
import re

import numpy as np
import soundfile as sf

from tools import make_variant_lanes_page as lanes

SR = 16000


def _wav(path, seconds=3.0, amp=0.5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(int(SR * seconds)) / SR
    sf.write(path, (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR)


def _transcripts(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def _plan(tmp_path):
    out = tmp_path / "out"
    rows = [{"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00", "text": "xin chào"},
            {"start": 1.0, "end": 2.5, "speaker": "SPEAKER_01", "text": "chào bạn </script>"}]
    base = out / "none" / "ep1"
    _wav(str(base / "01_music" / "after_music.wav"))
    _transcripts(str(base / "05_asr" / "transcripts.json"), rows)
    p3 = out / "m_p3" / "ep1"
    _wav(str(p3 / "01_music" / "after_music.wav"))
    _wav(str(p3 / "04b_denoise_pre_asr" / "after_denoise.wav"), amp=0.1)
    _transcripts(str(p3 / "05_asr" / "transcripts.json"), rows[:1])
    raw = out / "raw" / "ep1"                      # music stage off: no after_music.wav
    _transcripts(str(raw / "05_asr" / "transcripts.json"), rows)
    _wav(str(out / "raw" / "input" / "ep1.wav"))
    manifest = {"baseline": "none", "variants": [
        {"id": "m_p3", "chain": [], "chain_pre_asr": ["mpsenet_dns"], "status": "ok",
         "files": {"ep1": str(p3)}},
        {"id": "none", "chain": [], "status": "ok", "files": {"ep1": str(base)}},
        {"id": "raw", "chain": [], "set": {"steps.music_removal": False}, "status": "ok",
         "files": {"ep1": str(raw)}},
        {"id": "l_p1", "chain": ["rf_crowd"], "status": "failed", "files": {}},
    ]}
    path = out / "variants_manifest.json"
    path.write_text(json.dumps(manifest))
    return str(path), out


def test_page_data_per_variant(tmp_path):
    manifest, out = _plan(tmp_path)
    page_dir = str(out / "lanes")
    data = lanes.build_page_data(manifest, "ep1", page_dir)
    ids = [v["id"] for v in data["variants"]]
    assert ids[0] == "none" and set(ids) == {"none", "m_p3", "raw", "l_p1"}
    by = {v["id"]: v for v in data["variants"]}
    # the latest filter point is what is drawn and played
    assert by["m_p3"]["audio"] == "../m_p3/ep1/04b_denoise_pre_asr/after_denoise.wav"
    assert by["raw"]["audio"].endswith("raw/input/ep1.wav")
    assert "missing" in by["l_p1"]
    pk = np.frombuffer(base64.b64decode(by["none"]["peaks"]), dtype=np.uint8)
    assert len(pk) == 150 and 120 < pk.max() <= 128            # 3 s at 50/s, amp 0.5
    pk3 = np.frombuffer(base64.b64decode(by["m_p3"]["peaks"]), dtype=np.uint8)
    assert pk3.max() < 30                                       # quieter filtered wav, same scale
    assert {s["spk"] for s in by["none"]["segments"]} == {"SPEAKER_00", "SPEAKER_01"}
    assert by["m_p3"]["flow"].endswith("separation → mpsenet_dns → asr")
    assert "tiền xử lý" not in by["raw"]["flow"]


def test_main_writes_one_self_contained_page_per_file(tmp_path):
    manifest, out = _plan(tmp_path)
    assert lanes.main(["--manifest", manifest]) == 0
    page = (out / "lanes" / "ep1_lanes.html").read_text(encoding="utf-8")
    assert "</script>\"" not in page and "<\\/script>" in page   # text cannot close the script
    assert not re.search(r"(src|href)=\"https?://", page)
    data = json.loads(re.search(r"const DATA = (.*?);\n", page).group(1))
    assert len(data["variants"]) == 4


def test_audio_none_keeps_waveform_and_text(tmp_path):
    manifest, out = _plan(tmp_path)
    data = lanes.build_page_data(manifest, "ep1", str(out / "lanes"), audio_mode="none")
    assert all("audio" not in v for v in data["variants"])
    assert data["variants"][0]["peaks"]


def test_original_recording_outside_out_is_linked_through_the_audio_mount(tmp_path):
    out = tmp_path / "out"
    real = tmp_path / "recordings"; real.mkdir()
    sf.write(str(real / "ep1.wav"), np.zeros(SR, dtype=np.float32), SR)
    inp = out / "_resident" / "input_20260101_000000_1"; inp.mkdir(parents=True)
    (inp / "ep1.wav").symlink_to(real / "ep1.wav")
    d = out / "none" / "ep1"                                      # no after_music.wav: music steps are off
    _transcripts(str(d / "05_asr" / "transcripts.json"), [{"start": 0, "end": 1, "speaker": "S0", "text": "x"}])
    (out / "variants_manifest.json").write_text(json.dumps({"baseline": "none", "variants": [
        {"id": "none", "chain": [], "status": "ok", "files": {"ep1": str(d)}}]}))
    data = lanes.build_page_data(str(out / "variants_manifest.json"), "ep1", str(out / "lanes"))
    row = data["variants"][0]
    assert row["audio"] == "../_audio/ep1.wav" and row["audio_source"] == "input" and row["peaks"]


def test_input_inside_out_keeps_the_relative_link(tmp_path):
    manifest, out = _plan(tmp_path)                               # raw has OUT/raw/input/ep1.wav, a real file in OUT
    data = lanes.build_page_data(manifest, "ep1", str(out / "lanes"))
    raw = next(v for v in data["variants"] if v["id"] == "raw")
    assert raw["audio"].endswith("raw/input/ep1.wav") and "audio_source" not in raw
