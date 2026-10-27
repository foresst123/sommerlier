#!/usr/bin/env python3
"""Waveform lanes of one recording under one or more diarization results: one row per speaker, no text.

    python tools/make_diar_lanes_page.py --audio c.wav --diar nemotron=c_nemotron.json --diar diarizen=c_diarizen.json \
        --out OUT/lanes/c_diar.html

Each --diar is NAME=PATH (or just PATH, named after the file). The JSON holds {"segments": [{start, end, speaker}]}
(also a bare list, or a 02_diarization/segments.json). Every result becomes a block of rows: the waveform of each speaker
drawn only where that speaker is labelled, plus a row for the waveform no speaker owns. It is the lanes page of
make_variant_lanes_page.py with the text rows off, so playing, zoom (ctrl + wheel), drag-to-select and the variant
ticks work the same. Serve the folder that holds both the page and the audio with tools/serve_results.py; when the
audio lies outside it, pass --audio-url (for example ../_audio/c.wav with serve_results.py --audio-dir).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.make_variant_lanes_page import peaks, render  # noqa: E402


def load_segments(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("segments", data) if isinstance(data, dict) else data
    out = []
    for r in rows or []:
        try:
            s, e = float(r["start"]), float(r["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > s:
            out.append({"s": round(s, 3), "e": round(e, 3), "spk": str(r.get("speaker", "?")), "t": ""})
    return sorted(out, key=lambda x: (x["s"], x["e"]))


def build(audio, diars, out_path, audio_url=None, per_sec=50):
    pk, dur = peaks(audio, per_sec)
    url = audio_url or os.path.relpath(os.path.abspath(audio), os.path.dirname(os.path.abspath(out_path))).replace(os.sep, "/")
    rows = []
    for i, (name, path) in enumerate(diars):
        segs = load_segments(path)
        n_spk = len({x["spk"] for x in segs})
        talk = sum(x["e"] - x["s"] for x in segs)
        rows.append({"id": name, "label": name, "flow": f"{os.path.basename(path)} · {talk:.0f}s có nhãn",
                     "status": "ok", "baseline": i == 0, "peaks": pk, "duration": dur, "segments": segs,
                     "audio": url, "words": None, "n_speakers": n_spk})
    stem = os.path.splitext(os.path.basename(audio))[0]
    return {"file": stem, "files": [], "peaks_per_sec": per_sec, "transcript": "diarization",
            "no_text": True, "variants": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--diar", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio-url", help="how the page reaches the audio (default: path relative to the page)")
    ap.add_argument("--peaks-per-sec", type=int, default=50)
    a = ap.parse_args(argv)
    diars = []
    for item in a.diar:
        name, sep, path = item.partition("=")
        if not sep:
            name, path = os.path.splitext(os.path.basename(item))[0], item
        diars.append((name, path))
    data = build(a.audio, diars, a.out, a.audio_url, a.peaks_per_sec)
    if data["variants"][0]["peaks"] is None:
        ap.error(f"cannot read audio: {a.audio}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(render(data))
    for v in data["variants"]:
        print(f"{v['id']}: {v['n_speakers']} speaker, {len(v['segments'])} đoạn")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
