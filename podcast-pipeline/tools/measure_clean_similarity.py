#!/usr/bin/env python3
"""How much do the clean channels differ from the channels before the filter?  Speaker similarity and level numbers.

    python tools/measure_clean_similarity.py --final OUT/final                     # all conversations
    python tools/measure_clean_similarity.py --final OUT/final --limit 50 --device cuda --csv sim.csv
    python tools/measure_clean_similarity.py --final OUT/final --embedder none     # level numbers only, no model

For every conversation folder (the ones with a conversation.json) it takes the two channels of stereo_2ch.wav (before)
and of stereo_2ch_clean.wav (after), says which is the assistant and which the user from meta["channels"]["left"], and
measures, per channel, on the frames where the channel is not silent:
  spk_dist         cosine DISTANCE (1 - cosine) between the WeSpeaker embeddings of the speech before and after the
                   filter: 0 = the same voice, larger = the filter changed the voice (spk_sim = the cosine itself)
  turn_dist_max/mean  the same for single turns of 1 s or more (up to 12 turns): the worst turn shows a damaged sentence
  si_sdr_db        how far the clean waveform is from the original (low = changed a lot)
  lsd_db           log-spectral distance (dB) between the two
  level_change_db  RMS after vs before (negative = the filter took energy out)
  floor_raw_dbfs / floor_clean_dbfs / floor_drop_db   the 10th percentile frame level (the room) before and after
and across the speakers: cross_dist_raw / cross_dist_clean, the distance between the assistant's and the user's voice
(and cross_sim_*, the cosine). A cross distance close to the distance a person has to themself (spk_dist of the same
recording) hints that "two speakers" are one person; a clean channel that lost its voice shows as a large spk_dist.
A channel the filter did not touch (an old export: the user channel was never filtered) is marked `filtered=0` and its
numbers are the identity (spk_dist 0, si_sdr_db 99). Standard library + numpy + soundfile; the embedder needs the
pipeline's WeSpeaker (models/wespeaker_embedding.py) and is skipped with --embedder none.
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FRAME = 0.020
ACTIVE_RMS = 1e-4            # a frame quieter than this is the digital silence between sentences
MAX_EMBED_S = 30.0
MAX_TURNS = 12
MIN_TURN_S = 1.0


def frames_rms(x, sr, frame=FRAME):
    n = max(1, int(sr * frame))
    k = len(x) // n
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    f = np.asarray(x[:k * n], dtype=np.float64).reshape(k, n)
    return np.sqrt(np.mean(f * f, axis=1))


def active_mask(x, sr):
    return frames_rms(x, sr) > ACTIVE_RMS


def _active_samples(x, mask, sr, limit_s=None):
    n = max(1, int(sr * FRAME))
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.concatenate([x[i * n:(i + 1) * n] for i in idx]).astype(np.float32)
    return out[: int(limit_s * sr)] if limit_s else out


def si_sdr(reference, estimate):
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if not np.any(ref):
        return float("nan")
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-12))
    target = alpha * ref
    noise = est - target
    return float(10 * math.log10((np.sum(target ** 2) + 1e-12) / (np.sum(noise ** 2) + 1e-12)))


def lsd_db(a, b, sr, n_fft=512):
    n = min(len(a), len(b))
    if n < n_fft:
        return float("nan")
    hop = n_fft // 2
    win = np.hanning(n_fft)
    idx = range(0, n - n_fft, hop)
    sa = np.array([np.abs(np.fft.rfft(a[i:i + n_fft] * win)) for i in idx]) + 1e-8
    sb = np.array([np.abs(np.fft.rfft(b[i:i + n_fft] * win)) for i in idx]) + 1e-8
    d = 20 * np.log10(sa) - 20 * np.log10(sb)
    return float(np.mean(np.sqrt(np.mean(d * d, axis=1))))


def _db(x):
    return 20 * math.log10(max(float(x), 1e-9))


def signal_numbers(raw, clean, sr):
    """Level numbers over the frames where the original is not silent."""
    n = min(len(raw), len(clean))
    raw, clean = raw[:n], clean[:n]
    mask = active_mask(raw, sr)
    if not mask.any():
        return {"speech_s": 0.0}
    r, c = _active_samples(raw, mask, sr), _active_samples(clean, mask, sr)
    rms_r, rms_c = float(np.sqrt(np.mean(r ** 2))), float(np.sqrt(np.mean(c ** 2)))
    fr_r = frames_rms(r, sr)
    fr_c = frames_rms(c, sr)
    floor_r = _db(np.percentile(fr_r, 10)) if fr_r.size else float("nan")
    floor_c = _db(np.percentile(fr_c, 10)) if fr_c.size else float("nan")
    return {"speech_s": round(len(r) / sr, 2),
            "si_sdr_db": round(si_sdr(r, c), 2), "lsd_db": round(lsd_db(r, c, sr), 2),
            "level_change_db": round(_db(rms_c) - _db(rms_r), 2),
            "floor_raw_dbfs": round(floor_r, 1), "floor_clean_dbfs": round(floor_c, 1),
            "floor_drop_db": round(floor_r - floor_c, 1)}


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _embed(embedder, wave, sr):
    out = embedder.embed(np.asarray(wave, dtype=np.float32), sr)
    return out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)


def speaker_numbers(embedder, raw, clean, sr, turns):
    """spk_sim over the speech and over single turns (cosine of the voice before and after the filter)."""
    n = min(len(raw), len(clean))
    raw, clean = raw[:n], clean[:n]
    mask = active_mask(raw, sr)
    if not mask.any():
        return {}, None, None
    r, c = _active_samples(raw, mask, sr, MAX_EMBED_S), _active_samples(clean, mask, sr, MAX_EMBED_S)
    er, ec = _embed(embedder, r, sr), _embed(embedder, c, sr)
    sim = cosine(er, ec)
    out = {"spk_dist": round(1.0 - sim, 4), "spk_sim": round(sim, 4)}
    sims = []
    for a, b in turns[:MAX_TURNS]:
        i, j = int(a * sr), int(b * sr)
        if j - i < MIN_TURN_S * sr or j > n:
            continue
        seg_r, seg_c = raw[i:j], clean[i:j]
        if not np.any(np.abs(seg_r) > 1e-4) or not np.any(np.abs(seg_c) > 1e-4):
            continue
        try:
            sims.append(cosine(_embed(embedder, seg_r, sr), _embed(embedder, seg_c, sr)))
        except Exception:
            continue
    if sims:
        out["turn_dist_max"] = round(1.0 - min(sims), 4)
        out["turn_dist_mean"] = round(1.0 - float(np.mean(sims)), 4)
        out["turn_sim_min"] = round(min(sims), 4)
        out["turn_sim_mean"] = round(float(np.mean(sims)), 4)
    return out, er, ec


def find_conversations(final):
    found = []
    for cur, dirs, files in os.walk(final):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in ("lhotse",))
        if "conversation.json" in files:
            found.append(Path(cur))
            dirs[:] = []
    return found


def measure_folder(folder, embedder=None):
    """One row (dict) for a conversation folder, or None when it has no clean pair."""
    import soundfile as sf
    raw_path, clean_path = folder / "stereo_2ch.wav", folder / "stereo_2ch_clean.wav"
    if not (raw_path.is_file() and clean_path.is_file()):
        return None
    meta = json.loads((folder / "conversation.json").read_text(encoding="utf-8"))
    raw, sr = sf.read(str(raw_path), dtype="float32", always_2d=True)
    clean, sr2 = sf.read(str(clean_path), dtype="float32", always_2d=True)
    if raw.shape[1] != 2 or clean.shape[1] != 2 or sr != sr2:
        return None
    left = ((meta.get("channels") or {}).get("left")) or "assistant"
    chan = {"assistant": 0, "user": 1} if left == "assistant" else {"assistant": 1, "user": 0}
    row = {"id": str(folder), "category": folder.parent.name, "duration_s": round(raw.shape[0] / sr, 1)}
    embeds = {}
    for role, ch in chan.items():
        r, c = raw[:, ch], clean[:, ch]
        n = min(len(r), len(c))
        r, c = r[:n], c[:n]
        filtered = not np.array_equal(r, c)
        row[f"{role}_filtered"] = int(filtered)
        sn = signal_numbers(r, c, sr) if filtered else {"speech_s": signal_numbers(r, r, sr).get("speech_s", 0.0),
                                                       "si_sdr_db": 99.0, "lsd_db": 0.0, "level_change_db": 0.0}
        for k, v in sn.items():
            row[f"{role}_{k}"] = v
        if embedder is not None:
            turns = [(float(t["start"]), float(t["end"])) for t in meta.get("conversation", [])
                     if t.get("role") == role and "start" in t and "end" in t]
            try:
                if filtered:
                    sp, er, ec = speaker_numbers(embedder, r, c, sr, turns)
                else:
                    sp, er, ec = speaker_numbers(embedder, r, r, sr, turns)
            except Exception as exc:                 # a bad clip must not stop the survey
                sp, er, ec = {}, None, None
                row[f"{role}_embed_error"] = f"{type(exc).__name__}: {exc}"[:80]
            for k, v in sp.items():
                row[f"{role}_{k}"] = v
            embeds[role] = (er, ec)
    if embedder is not None and embeds.get("assistant", (None,))[0] is not None \
            and embeds.get("user", (None,))[0] is not None:
        sim_raw = cosine(embeds["assistant"][0], embeds["user"][0])
        sim_clean = cosine(embeds["assistant"][1], embeds["user"][1])
        row["cross_dist_raw"] = round(1.0 - sim_raw, 4)
        row["cross_dist_clean"] = round(1.0 - sim_clean, 4)
        row["cross_sim_raw"] = round(sim_raw, 4)
        row["cross_sim_clean"] = round(sim_clean, 4)
    return row


def _make_wespeaker(device):
    from models.wespeaker_embedding import WeSpeakerONNXEmbedder
    return WeSpeakerONNXEmbedder(device)


def _stats(values):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if v.size == 0:
        return "n=0"
    return (f"n={v.size} mean={v.mean():.3f} min={v.min():.3f} p10={np.percentile(v, 10):.3f} "
            f"p50={np.percentile(v, 50):.3f} p90={np.percentile(v, 90):.3f}")


def summarize(rows, out=print):
    out(f"\n{len(rows)} conversations measured")
    for role in ("assistant", "user"):
        done = [r for r in rows if r.get(f"{role}_filtered")]
        out(f"\n[{role}] filtered in {len(done)} of {len(rows)}")
        for key in ("spk_dist", "turn_dist_max", "si_sdr_db", "lsd_db", "level_change_db", "floor_drop_db"):
            vals = [r.get(f"{role}_{key}") for r in done]
            if any(v is not None for v in vals):
                out(f"  {key:16s} {_stats(vals)}")
    cross = [r for r in rows if "cross_sim_clean" in r]
    if cross:
        out(f"\n[between the two speakers] distance = 1 - cosine\n"
            f"  cross_dist_raw    {_stats([r['cross_dist_raw'] for r in cross])}"
            f"\n  cross_dist_clean  {_stats([r['cross_dist_clean'] for r in cross])}")
        out("\nclosest speakers (smallest distance, maybe one person):")
        for r in sorted(cross, key=lambda r: r["cross_dist_raw"])[:8]:
            out(f"  {r['cross_dist_raw']:.3f}  {r['id']}")
    for role in ("assistant", "user"):
        far = sorted((r for r in rows if r.get(f"{role}_spk_dist") is not None and r.get(f"{role}_filtered")),
                     key=lambda r: -r[f"{role}_spk_dist"])[:8]
        if far:
            out(f"\nlargest {role} voice distance before/after the filter (voice changed most):")
            for r in far:
                out(f"  {r[f'{role}_spk_dist']:.3f}  {r['id']}")


def main(argv=None, embedder_factory=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", type=Path, required=True, help="the final folder (OUT/final)")
    ap.add_argument("--limit", type=int, default=0, help="measure only the first N conversations (0 = all)")
    ap.add_argument("--csv", type=Path, default=None, help="default: <final>/clean_similarity.csv")
    ap.add_argument("--device", default="cpu", help="cpu or cuda, for the WeSpeaker embedder")
    ap.add_argument("--embedder", choices=("wespeaker", "none"), default="wespeaker")
    args = ap.parse_args(argv)
    if not args.final.is_dir():
        sys.exit(f"not a directory: {args.final}")
    embedder = None
    if args.embedder == "wespeaker":
        embedder = (embedder_factory or _make_wespeaker)(args.device)
    folders = find_conversations(args.final)
    if args.limit:
        folders = folders[:args.limit]
    rows = []
    for i, folder in enumerate(folders, 1):
        try:
            row = measure_folder(folder, embedder)
        except Exception as exc:
            print(f"skip {folder}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if row is not None:
            rows.append(row)
        if i % 20 == 0:
            print(f"  {i}/{len(folders)}", flush=True)
    summarize(rows)
    if rows:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        path = args.csv or args.final / "clean_similarity.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
