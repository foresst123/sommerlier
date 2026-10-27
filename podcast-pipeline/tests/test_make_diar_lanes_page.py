import json
import os

import numpy as np
import soundfile as sf

from tools import make_diar_lanes_page as dl

SR = 16000


def test_page_has_wave_rows_per_speaker_and_no_text(tmp_path):
    audio = tmp_path / "c.wav"
    sf.write(str(audio), (0.3 * np.sin(np.arange(SR * 4) / 9)).astype(np.float32), SR)
    diar = tmp_path / "c.json"
    diar.write_text(json.dumps({"segments": [
        {"start": 0.5, "end": 1.5, "speaker": "SPEAKER_00"}, {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_05"},
        {"start": 2.5, "end": 2.4, "speaker": "SPEAKER_01"}], "num_speakers": 2}))
    out = tmp_path / "lanes" / "c_diar.html"
    assert dl.main(["--audio", str(audio), "--diar", f"nemo={diar}", "--diar", str(diar), "--out", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    data = json.loads(page.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert data["no_text"] is True and [v["id"] for v in data["variants"]] == ["nemo", "c"]
    v = data["variants"][0]
    assert {s["spk"] for s in v["segments"]} == {"SPEAKER_00", "SPEAKER_05"}       # the reversed segment is dropped
    assert v["audio"] == "../c.wav" and v["peaks"]
    assert 'if (!DATA.no_text) addLane("text"' in page
