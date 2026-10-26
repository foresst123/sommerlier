"""Load one per-file pipeline output dir into a normalized dict.

Layout read (written by services/stage_output_service.py and the denoise stage):
  01_music/after_music.wav            raw (pre-denoise) audio
  01b_denoise/after_denoise.wav       filtered audio, absent for the no-filter run
  01b_denoise/denoise.json
  02_diarization/segments.json        [{index,start,end,speaker}]
  03_separation/segments.json         SpeechSegment rows (bss_spans, bss_failed_spans)
  03_separation/stats.json
  05_asr/transcripts.json             TranscriptSegment rows; "words" is
                                      [{word,start,end}] (absolute seconds) or null
Every stage may be missing; its entry is then None.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np


def read_json(path: str) -> Any:
    """Parsed JSON, or None when the file is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_wav(path: Optional[str]):
    """(mono float32 samples, sample_rate) or (None, None)."""
    if not path or not os.path.isfile(path):
        return None, None
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return None, None
    return data.mean(axis=1).astype(np.float32), int(sr)


def wav_duration(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.isfile(path):
        return None
    try:
        import soundfile as sf
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def _rows(data) -> Optional[List[dict]]:
    if isinstance(data, dict):
        data = data.get("segments")
    if not isinstance(data, list):
        return None
    return [r for r in data if isinstance(r, dict)]


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _norm_words(words) -> Optional[List[dict]]:
    if not words:
        return None
    out = []
    for w in words:
        if not isinstance(w, dict):
            continue
        s, e = _num(w.get("start")), _num(w.get("end"))
        if s is None or e is None:
            continue
        out.append({"word": str(w.get("word", w.get("text", ""))).strip(), "start": s, "end": e})
    return out or None


def _norm_segments(rows, with_text=False) -> Optional[List[dict]]:
    if rows is None:
        return None
    out = []
    for r in rows:
        s, e = _num(r.get("start")), _num(r.get("end"))
        if s is None or e is None:
            continue
        seg = {"start": s, "end": e, "speaker": str(r.get("speaker"))}
        if with_text:
            seg["text"] = str(r.get("text") or "")
            seg["words"] = _norm_words(r.get("words"))
        else:
            seg["bss_spans"] = [list(sp) for sp in (r.get("bss_spans") or [])]
            seg["bss_failed_spans"] = [list(sp) for sp in (r.get("bss_failed_spans") or [])]
            seg["bss"] = bool(r.get("bss"))
        out.append(seg)
    return out


def load_file_output(path: str, name: Optional[str] = None) -> Dict[str, Any]:
    """Normalized view of one file's output dir; never raises on missing stages."""
    raw = os.path.join(path, "01_music", "after_music.wav")
    # The latest filter point is what the later stages worked on: pre_asr, pre_separation, then 01b.
    points = [os.path.join(path, d) for d in ("04b_denoise_pre_asr", "02b_denoise_pre_separation", "01b_denoise")]
    den = next((os.path.join(d, "after_denoise.wav") for d in points
                if os.path.isfile(os.path.join(d, "after_denoise.wav"))), None)
    den_json = next((os.path.join(d, "denoise.json") for d in points
                     if os.path.isfile(os.path.join(d, "denoise.json"))), points[-1] + "/denoise.json")
    raw = raw if os.path.isfile(raw) else None

    diar = _norm_segments(_rows(read_json(os.path.join(path, "02_diarization", "segments.json"))))
    sep_rows = _rows(read_json(os.path.join(path, "03_separation", "segments.json")))
    sep_stats = read_json(os.path.join(path, "03_separation", "stats.json"))
    asr = _norm_segments(_rows(read_json(os.path.join(path, "05_asr", "transcripts.json"))),
                         with_text=True)

    duration = wav_duration(raw) or wav_duration(den)
    if duration is None:
        ends = [s["end"] for s in (diar or []) + (asr or [])]
        duration = max(ends) if ends else None

    return {
        "name": name or os.path.basename(os.path.normpath(path)),
        "dir": path,
        "duration": duration,
        "audio": {"raw": raw, "denoised": den},
        "denoise": read_json(den_json),
        "diarization": diar,
        "separation": ({"segments": _norm_segments(sep_rows), "stats": sep_stats}
                       if sep_rows is not None or sep_stats is not None else None),
        "asr": asr,
    }


NOOP_CORR = 0.99999  # same threshold as run_variants.py


def denoise_entry(chain, info, from_manifest=None) -> Optional[Dict[str, Any]]:
    """files[].denoise[variant]: verdict/errors/corr/lag from the manifest's denoise_report row
    (preferred) or derived from the stage's denoise.json; steps and blend come from the json.
    None when neither source says anything."""
    info = info if isinstance(info, dict) else None
    row = dict(from_manifest) if isinstance(from_manifest, dict) else None
    if row is None:
        if info is None:
            return None
        errors = [str(e) for e in info.get("errors") or []]
        corr = info.get("corr_with_input")
        real = any(t != "identity" and not re.fullmatch(r"blend\d{1,3}", str(t)) for t in chain or [])
        verdict = ("fallback" if errors else
                   "suspect_noop" if isinstance(corr, (int, float)) and corr >= NOOP_CORR and real else "ok")
        row = {"verdict": verdict, "errors": errors, "corr_with_input": corr,
               "lag_samples": info.get("lag_samples")}
    row.setdefault("errors", [])
    row["steps"] = (info or {}).get("steps") or []
    row["blend"] = (info or {}).get("blend")
    return row


def load_reference(path: Optional[str]) -> Optional[List[dict]]:
    """Reference transcript rows [{start,end,text}] (tools/hoahau_edited.json style).

    `text_edited` (the human-corrected text) wins over `text` when present.
    """
    if not path:
        return None
    rows = _rows(read_json(path))
    if rows is None:
        return None
    out = []
    for r in rows:
        s, e = _num(r.get("start")), _num(r.get("end"))
        if s is None or e is None:
            continue
        text = r.get("text_edited") or r.get("text") or ""
        out.append({"start": s, "end": e, "text": str(text)})
    return out
