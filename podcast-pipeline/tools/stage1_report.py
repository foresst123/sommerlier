"""Step three: turn the matrix into one page you can listen to.

Audio is embedded as base64 mp3 rather than linked, because the report is meant
to survive being moved off this machine -- a link to results/ would go dead the
moment it left.
"""
import base64
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

OUT = sys.argv[1]
CLIP_SECONDS = 12.0
SEPS = ["bsroformer_ep368", "melband_denoise", "melband_denoise_aggr"]
SEP_LABEL = {
    "bsroformer_ep368": "BS-RoFormer ep_368",
    "melband_denoise": "Mel-Band Denoise",
    "melband_denoise_aggr": "Mel-Band Denoise Aggressive",
}
REC_LABEL = {"thu_that_thach_10m": "Thử thách (10 phút)",
             "lm8": "LM8 — Vòng tay nắng (22 phút)",
             "vimeanh": "Vì mẹ anh — chia tay (28 phút)"}


def mp3_uri(wav_path, start=None, length=None):
    """A short excerpt as a data: URI, or None if it could not be made."""
    if not wav_path or not os.path.exists(wav_path):
        return None
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        dest = tmp.name
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.2f}"]
    cmd += ["-i", wav_path]
    if length is not None:
        cmd += ["-t", f"{length:.2f}"]
    cmd += ["-ac", "1", "-ar", "22050", "-b:a", "48k", dest]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        with open(dest, "rb") as f:
            blob = base64.b64encode(f.read()).decode()
        return f"data:audio/mpeg;base64,{blob}"
    except Exception:
        return None
    finally:
        os.path.exists(dest) and os.unlink(dest)


def energy_split(out, rep):
    """How much of a span's energy the separator decided was not voice.

    Reported as a share rather than dB because that is the question being
    asked: a bed under speech that turns out to be one percent of the energy
    was never the reason a segment was unusable.
    """
    import soundfile as sf
    keeps, drops = [], []
    for d in rep.get("spans_done", []):
        kept = os.path.join(out, "sep", rep["separator"], d.get("kept") or "")
        other = next((os.path.join(out, "sep", rep["separator"], x)
                      for x in d.get("stems", []) if x != d.get("kept")), None)
        if not other or not os.path.exists(kept) or not os.path.exists(other):
            continue
        db = lambda v: 20 * np.log10(
            np.sqrt((np.asarray(v, dtype=float) ** 2).mean()) + 1e-12)
        keeps.append(db(sf.read(kept)[0]))
        drops.append(db(sf.read(other)[0]))
    if not keeps:
        return None
    k, o = float(np.mean(keeps)), float(np.mean(drops))
    return {"kept_db": k, "dropped_db": o,
            "dropped_share": 100 / (1 + 10 ** ((k - o) / 10))}


def main():
    with open(os.path.join(OUT, "index.json")) as f:
        index = json.load(f)
    sep_path = os.path.join(OUT, "separate.json")
    seps = json.load(open(sep_path)) if os.path.exists(sep_path) else {}

    data = {"recordings": {}, "separators": SEP_LABEL, "sep_order": SEPS,
            "dist": {}}
    for key in index:
        npz = os.path.join(OUT, f"{key}_scores.npz")
        if not os.path.exists(npz):
            continue
        z = np.load(npz)
        data["dist"][key] = {
            k: {"p50": float(np.percentile(z[k], 50)),
                "p90": float(np.percentile(z[k], 90)),
                "p99": float(np.percentile(z[k], 99)),
                "max": float(z[k].max()),
                "over": float((z[k] >= 0.10).mean() * 100)}
            for k in ("speech", "music", "singing") if k in z and len(z[k])}
    for key, entry in index.items():
        rec, det = entry["recording"], entry["detector"]
        node = data["recordings"].setdefault(
            rec, {"label": REC_LABEL.get(rec, rec),
                  "duration": entry["duration"], "detectors": {}})

        # The span to listen to: the longest one that actually got separated,
        # so the clip shows the model working on real material rather than on
        # a fragment too short to judge.
        cand = sorted(entry["separate"],
                      key=lambda s: s["end"] - s["start"], reverse=True)
        pick = cand[0] if cand else None

        clips = {}
        if pick:
            src = os.path.join(OUT, "spans", key, pick["file"])
            clips["before"] = mp3_uri(src, 0, CLIP_SECONDS)
            for sep in SEPS:
                rep = seps.get(f"{key}__{sep}", {})
                match = next((d for d in rep.get("spans_done", [])
                              if d["i"] == pick["i"]), None)
                if match and match.get("kept"):
                    # audio-separator fixes its output directory when the
                    # Separator is built, so every key's stems land together in
                    # sep/<separator>/ rather than under the key. Span
                    # filenames carry index and timestamps and were checked to
                    # be unique across all six combinations, so the flat layout
                    # loses nothing -- but the path has to be read flat too.
                    clips[sep] = mp3_uri(
                        os.path.join(OUT, "sep", sep, match["kept"]),
                        0, CLIP_SECONDS)

        # How fragmented the map is. A detector that finds the same total
        # seconds in twice as many pieces is a worse detector: every extra
        # piece is another join, and pieces under a second get dropped whole.
        lens = sorted(b - a for a, b, _k in entry["spans"])
        cuts = sorted(b - a for a, b, k in entry["spans"] if k != "music")

        node["detectors"][det] = {
            "span_len": {
                "n": len(lens),
                "median": float(np.median(lens)) if lens else 0.0,
                "max": float(max(lens)) if lens else 0.0,
                "under_1s": sum(1 for x in lens if x < 1.0),
                "cut_median": float(np.median(cuts)) if cuts else 0.0,
            },
            "totals": entry["totals"], "excised": entry["excised"],
            "n_spans": len(entry["spans"]), "n_separate": len(entry["separate"]),
            "detect_seconds": entry["detect_seconds"], "fps": entry["fps"],
            "curves": entry["curves"],
            "pick": ({"start": pick["start"], "end": pick["end"]} if pick else None),
            "clips": clips,
            "sep_stats": {s: {
                "ok": len(seps.get(f"{key}__{s}", {}).get("spans_done", [])),
                "fail": len(seps.get(f"{key}__{s}", {}).get("spans_failed", [])),
                "wall": seps.get(f"{key}__{s}", {}).get("wall_seconds", 0),
                "audio": seps.get(f"{key}__{s}", {}).get("audio_seconds", 0),
                "energy": (energy_split(OUT, seps[f"{key}__{s}"])
                           if f"{key}__{s}" in seps else None),
            } for s in SEPS},
        }

    with open(os.path.join(OUT, "report_data.json"), "w") as f:
        json.dump(data, f)
    size = os.path.getsize(os.path.join(OUT, "report_data.json"))
    print(f"report_data.json: {size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
