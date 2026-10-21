"""Assemble compare_data.json (see SCHEMA.md) from a variants manifest."""

import json
import os
from typing import Dict, List, Optional

from . import metrics as M
from .io_stage import denoise_entry, load_file_output, load_reference
from .speaker_align import compare_speakers
from .text_align import compare_transcripts

VERSION = 1


def _full_text(segs) -> str:
    return " ".join(s["text"].strip() for s in sorted(segs, key=lambda s: (s["start"], s["end"]))
                    if s["text"].strip())


def _flat(vals: dict, prefix: str = "") -> dict:
    """Nested dicts flattened to 'a.b' keys (audio.learned_mos.<key>)."""
    out = {}
    for k, x in (vals or {}).items():
        if isinstance(x, dict):
            out.update(_flat(x, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = x
    return out


def _summary(variants: List[dict], metrics_by_file: Dict[str, Dict[str, dict]]) -> dict:
    out: Dict[str, dict] = {}
    for v in variants:
        acc: Dict[str, List[float]] = {}
        for per_variant in metrics_by_file.values():
            for sec, vals in (per_variant.get(v["id"]) or {}).items():
                for k, x in _flat(vals).items():
                    if isinstance(x, (int, float)) and x == x:  # bools count, NaN does not
                        acc.setdefault(f"{sec}.{k}", []).append(float(x))
        out[v["id"]] = {k: sum(xs) / len(xs) for k, xs in sorted(acc.items())}
    return out


BAD_VERDICTS = ("fallback", "missing")


def _first_error(entries) -> str:
    for e in entries:
        if e.get("verdict") in BAD_VERDICTS and e.get("errors"):
            return str(e["errors"][0]).strip().splitlines()[0][:200]
    return ""


def build_compare_data(manifest_path: str, files: Optional[List[str]] = None,
                       references: Optional[Dict[str, str]] = None, embed: bool = True) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    base_id = manifest.get("baseline", "none")
    refs = dict(manifest.get("reference") or {})
    refs.update(references or {})

    variants_out, usable = [], []
    for v in manifest.get("variants", []):
        row = {"id": v["id"], "chain": v.get("chain", []), "label": v.get("label", v["id"])}
        status = v.get("status", "ok")
        row["status"] = status
        if status == "degraded":
            row["degraded"] = True
        if v.get("group"):
            row["group"] = v["group"]
        if v.get("tag"):
            row["tag"] = str(v["tag"])
        if v.get("set"):
            row["set"] = dict(v["set"])
        if v.get("note"):
            row["note"] = str(v["note"])
        if status in ("failed", "missing", "skipped"):
            why = v.get("reason") or f"variant status is {status}"
            row["note"] = f"skipped: {why}" + (f" | {v['note']}" if v.get("note") else "")
        else:
            usable.append(v)
        variants_out.append(row)

    stems: List[str] = []
    for v in usable:
        for stem in (v.get("files") or {}):
            if stem not in stems and (not files or stem in files):
                stems.append(stem)

    files_out, metrics_by_file = [], {}
    manifest_by_id = {v["id"]: v for v in usable}
    den_all: Dict[str, List[dict]] = {}
    for stem in stems:
        loaded = {v["id"]: load_file_output(v["files"][stem], stem)
                  for v in usable if stem in (v.get("files") or {})}
        base = loaded.get(base_id)
        raw = next((o["audio"]["raw"] for o in ([base] if base else []) + list(loaded.values())
                    if o["audio"]["raw"]), None)
        duration = (base or {}).get("duration") or next(
            (o["duration"] for o in loaded.values() if o["duration"]), None)
        ref_rows = load_reference(refs.get(stem))

        audio = {"raw": raw}
        text, text_diff, speakers, mets = {}, {}, {}, {}
        for vid, o in loaded.items():
            is_base = vid == base_id
            asr = o["asr"]
            audio[vid] = raw if is_base else o["audio"]["denoised"]
            var_wav = audio[vid]
            if asr is not None:
                text[vid] = {"segments": [{k: s[k] for k in ("start", "end", "speaker", "text")}
                                          for s in asr], "full_text": _full_text(asr)}
            vs_base = None
            if base is not None and asr is not None and base["asr"] is not None:
                vs_base = compare_transcripts(base["asr"], asr, detail=not is_base)
            vs_ref = compare_transcripts(ref_rows, asr, detail=False) if ref_rows is not None else None
            if vs_base is not None or vs_ref is not None:
                text_diff[vid] = {
                    "wer_vs_baseline": vs_base["wer"] if vs_base else None,
                    "ops": vs_base["ops"] if vs_base else None,
                    "wer_vs_reference": vs_ref["wer"] if vs_ref else None,
                    "segments": vs_base["segments"] if vs_base else [],
                }
            if base is not None:
                sp = compare_speakers(base["diarization"], o["diarization"], duration)
                if sp is not None:
                    speakers[vid] = sp
            td = text_diff.get(vid, {})
            mets[vid] = {
                "audio": M.audio_metrics(raw, var_wav, same_as_raw=is_base, embed=embed),
                "diarization": M.diarization_metrics(o["diarization"]),
                "separation": M.separation_metrics(o["separation"]),
                "asr": M.asr_metrics(asr, var_wav, td.get("wer_vs_baseline"),
                                     td.get("wer_vs_reference")),
            }
        denoise = {}
        for vid, o in loaded.items():
            mv = manifest_by_id[vid]
            ent = denoise_entry(mv.get("chain"), o.get("denoise"), (mv.get("denoise_report") or {}).get(stem))
            if ent is not None:
                denoise[vid] = ent
                den_all.setdefault(vid, []).append(ent)
        files_out.append({"name": stem, "duration": duration, "audio": audio, "text": text,
                          "text_diff": text_diff, "speakers": speakers, "metrics": mets,
                          "denoise": denoise})
        metrics_by_file[stem] = mets

    summary_n = {v["id"]: sum(1 for m in metrics_by_file.values() if v["id"] in m) for v in variants_out}
    excluded = {}
    for row in variants_out:
        ents = den_all.get(row["id"], [])
        bad = sum(1 for e in ents if e.get("verdict") in BAD_VERDICTS)
        mv = manifest_by_id.get(row["id"])
        if mv is not None:
            # the manifest counts files that never produced output too (total_files)
            bad = max(bad, int(mv.get("degraded_files") or 0))
            noop = sum(1 for e in ents if e.get("verdict") == "suspect_noop")
            total = int(mv.get("total_files") or 0) or len(ents)
            row["degraded"] = bool(bad or row.get("degraded"))
            row["degraded_files"], row["total_files"] = bad, total
            if row["degraded"]:
                row["degraded_reason"] = (f"fell back to unfiltered audio in {bad}/{total} files"
                                          + (f": {_first_error(ents)}" if _first_error(ents) else ""))
            if noop:
                row["suspect_noop_files"] = noop
        excluded[row["id"]] = bad

    return {
        "version": VERSION, "baseline": base_id, "variants": variants_out, "files": files_out,
        "summary": _summary(variants_out, metrics_by_file),
        "summary_n": summary_n, "summary_excluded_degraded": excluded,
    }


def write_compare_data(data: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
