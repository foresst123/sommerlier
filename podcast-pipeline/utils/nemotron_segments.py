"""Turn Nemotron-3-Diarization output into the pipeline's diarization segments.

No NeMo or torch import here, so it is importable (and testable) anywhere. The
worker returns the same {"start", "end", "speaker"} dicts as the DiariZen one.
"""

import re

# Model card: up to eight speakers, channels ordered by first arrival.
MAX_SPEAKERS = 8


def _parse_one(raw):
    """Return (start, end, speaker_id) for one raw segment.

    Accepted shapes: "0.00 3.20 speaker_0" (assumed NeMo format; the card only
    says 'begin_seconds, end_seconds, speaker_index'), (start, end, speaker),
    and the Transformers processor's {"Start", "End", "Speaker"} dict.
    """
    if isinstance(raw, dict):
        return float(raw["Start"]), float(raw["End"]), raw["Speaker"]
    if isinstance(raw, str):
        parts = raw.replace(",", " ").split()
    else:
        parts = list(raw)
    if len(parts) != 3:
        raise ValueError(f"Unrecognised Nemotron segment: {raw!r}")
    return float(parts[0]), float(parts[1]), parts[2]


def raw_to_segments(raw_segments):
    """Normalize one model output list without renaming its local speakers."""
    result = []
    for raw in raw_segments or []:
        start, end, speaker = _parse_one(raw)
        if end > start:
            result.append({"start": start, "end": end, "speaker": str(speaker)})
    return sorted(result, key=lambda item: (item["start"], item["end"], item["speaker"]))


def _speaker_index(label):
    """'speaker_3' / 3 / '3' -> 3."""
    match = re.search(r"\d+", str(label))
    if match is None:
        raise ValueError(f"Speaker label without an index: {label!r}")
    return int(match.group())


def to_pipeline_segments(raw_segments, merge_gap=0.0, min_duration=0.0):
    """Convert raw model segments to sorted pipeline segments.

    Speakers are renamed SPEAKER_00, SPEAKER_01, ... in order of first
    appearance (DiariZen's naming). Same-speaker turns separated by at most
    ``merge_gap`` seconds are joined, then turns shorter than ``min_duration``
    are dropped. Overlapping turns of different speakers are kept as-is.
    """
    turns = []
    for raw in raw_segments or []:
        start, end, speaker = _parse_one(raw)
        if end > start:
            turns.append((start, end, _speaker_index(speaker)))
    turns.sort()

    names = {}
    for _, _, speaker in turns:
        names.setdefault(speaker, f"SPEAKER_{len(names):02d}")

    merged = []
    open_turn = {}  # speaker -> index into merged of its latest turn
    for start, end, speaker in turns:
        at = open_turn.get(speaker)
        if at is not None and start - merged[at]["end"] <= merge_gap:
            merged[at]["end"] = max(merged[at]["end"], end)
            continue
        open_turn[speaker] = len(merged)
        merged.append({"start": start, "end": end, "speaker": names[speaker]})

    return [seg for seg in merged if seg["end"] - seg["start"] >= min_duration]


# NeMo's post-processing YAML for Sortformer (a `parameters:` block). Values are
# those NeMo ships for CallHome part 1 (sortformer_diar_4spk-v1_callhome-part1.yaml);
# they were tuned for other Sortformer models, not for Nemotron-3, so treat them as
# a starting point. Keys not given in the profile take these values.
POSTPROCESSING_KEYS = ("onset", "offset", "pad_onset", "pad_offset",
                       "min_duration_on", "min_duration_off")
POSTPROCESSING_BASE = {"onset": 0.53, "offset": 0.49, "pad_onset": 0.23, "pad_offset": 0.01,
                       "min_duration_on": 0.42, "min_duration_off": 0.34}


def postprocessing_params(cfg):
    """The six thresholds from ``models.nemotron.postprocessing``, or None when the key is absent.

    An empty ``{}`` is "set": every threshold then takes its base value.

    Raises ValueError on an unknown key so a typo is not a silently ignored setting.
    """
    cfg = cfg or {}
    if cfg.get("postprocessing") is None:
        return None
    section = cfg["postprocessing"]          # {} means "the base values as they are"
    unknown = sorted(set(section) - set(POSTPROCESSING_KEYS))
    if unknown:
        raise ValueError(f"unknown models.nemotron.postprocessing key(s): {', '.join(unknown)}")
    return {key: float(section.get(key, POSTPROCESSING_BASE[key])) for key in POSTPROCESSING_KEYS}


def write_postprocessing_yaml(params, directory):
    """Write ``params`` as the YAML NeMo's ``postprocessing_yaml`` expects; returns its path."""
    import os
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "nemotron_postprocessing.yaml")
    lines = ["parameters:"] + [f"  {key}: {params[key]}" for key in POSTPROCESSING_KEYS]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def omi_pause_bridging_segments(raw_segments, unconditional_gap=0.18, conditional_gap=0.30, min_duration=0.18):
    """2-Stage Uninterrupted Pause Bridging Decoder optimized for Vietnamese Nemotron-3."""
    turns = []
    for raw in raw_segments or []:
        s, e, spk = _parse_one(raw)
        if e > s:
            turns.append([float(s), float(e), _speaker_index(spk)])
    turns.sort(key=lambda x: (x[0], x[1]))
    if not turns:
        return []

    names = {}
    for _, _, spk in turns:
        names.setdefault(spk, f"SPEAKER_{len(names):02d}")

    by_spk = {}
    for s, e, spk in turns:
        by_spk.setdefault(spk, []).append([s, e])

    for spk, segs in by_spk.items():
        merged = [segs[0]]
        for s, e in segs[1:]:
            if s - merged[-1][1] <= unconditional_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        by_spk[spk] = merged

    for spk, segs in list(by_spk.items()):
        other_segs = [iv for other_spk, ivs in by_spk.items() if other_spk != spk for iv in ivs]
        merged = [segs[0]]
        for s, e in segs[1:]:
            gap = s - merged[-1][1]
            if gap <= conditional_gap:
                p_start, p_end = merged[-1][1], s
                if not any(os_ < p_end and oe_ > p_start for os_, oe_ in other_segs):
                    merged[-1][1] = max(merged[-1][1], e)
                    continue
            merged.append([s, e])
        by_spk[spk] = merged

    out = []
    for spk, segs in by_spk.items():
        for s, e in segs:
            if e - s >= min_duration:
                out.append({"start": s, "end": e, "speaker": names[spk]})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out

# === VIETNAMESE OPTIMIZED POST-PROCESSING ===
import numpy as np

def omi_pause_bridging_segments(raw_segments, unconditional_gap=0.18, conditional_gap=0.30, min_duration=0.18):
    turns = []
    for raw in raw_segments or []:
        s, e, spk = _parse_one(raw)
        if e > s:
            turns.append([float(s), float(e), _speaker_index(spk)])
    turns.sort(key=lambda x: (x[0], x[1]))
    if not turns: return []
    names = {}
    for _, _, spk in turns:
        names.setdefault(spk, f"SPEAKER_{len(names):02d}")
    by_spk = {}
    for s, e, spk in turns:
        by_spk.setdefault(spk, []).append([s, e])
    for spk, segs in by_spk.items():
        merged = [segs[0]]
        for s, e in segs[1:]:
            if s - merged[-1][1] <= unconditional_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        by_spk[spk] = merged
    for spk, segs in list(by_spk.items()):
        other_segs = [iv for other_spk, ivs in by_spk.items() if other_spk != spk for iv in ivs]
        merged = [segs[0]]
        for s, e in segs[1:]:
            gap = s - merged[-1][1]
            if gap <= conditional_gap:
                p_start, p_end = merged[-1][1], s
                if not any(os_ < p_end and oe_ > p_start for os_, oe_ in other_segs):
                    merged[-1][1] = max(merged[-1][1], e)
                    continue
            merged.append([s, e])
        by_spk[spk] = merged
    out = []
    for spk, segs in by_spk.items():
        for s, e in segs:
            if e - s >= min_duration:
                out.append({"start": s, "end": e, "speaker": names[spk]})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out

def precision_lock_segments(raw_segments, wav=None, sr=16000, duration=60.0, unconditional_gap=0.22, conditional_gap=0.38):
    bridged = omi_pause_bridging_segments(raw_segments, unconditional_gap=0.18, conditional_gap=0.30, min_duration=0.16)
    if not bridged: return []
    step = 0.01
    n_frames = max(1, int(round(duration / step)))
    spk_names = []
    for s in bridged:
        if s["speaker"] not in spk_names: spk_names.append(s["speaker"])
    K = len(spk_names)
    grid = np.zeros((n_frames, K), dtype=bool)
    for s in bridged:
        k = spk_names.index(s["speaker"])
        i0, i1 = max(0, int(round(s["start"] / step))), min(n_frames, int(round(s["end"] / step)))
        if i1 > i0: grid[i0:i1, k] = True
    tot_frames = grid.sum(axis=0)
    order_by_dur = sorted(range(K), key=lambda k: int(tot_frames[k]), reverse=True)
    kept_ks = []
    for k in order_by_dur:
        if tot_frames[k] == 0: continue
        if len(kept_ks) == 0:
            kept_ks.append(k)
            continue
        union_kept = np.any(grid[:, kept_ks], axis=1)
        ov_ratio = int(np.count_nonzero(grid[:, k] & union_kept)) / float(max(1, int(tot_frames[k])))
        if len(kept_ks) == 1:
            if ov_ratio > 0.70 and int(np.count_nonzero(grid[:, k] & (~union_kept))) * step < 0.80:
                grid[grid[:, k] & (~union_kept), kept_ks[0]] = True
                grid[:, k] = False
            else: kept_ks.append(k)
        else:
            if ov_ratio > 0.35 or int(np.count_nonzero(grid[:, k] & (~union_kept))) * step < 2.50:
                excl = grid[:, k] & (~union_kept)
                if np.any(excl):
                    best_p = max(kept_ks, key=lambda p: int(grid[:, p].sum()))
                    grid[excl, best_p] = True
                grid[:, k] = False
            else: kept_ks.append(k)
    raw_turns = []
    for k in sorted(kept_ks):
        col = grid[:, k]
        in_seg, s0 = False, 0
        for t in range(n_frames + 1):
            val = bool(col[t]) if t < n_frames else False
            if val and not in_seg:
                in_seg, s0 = True, t
            elif not val and in_seg:
                in_seg = False
                if (t - s0) >= 14:
                    raw_turns.append((s0 * step, t * step, k))
    raw_turns.sort(key=lambda x: (x[0], x[1]))
    names = {}
    for _, _, k in raw_turns:
        names.setdefault(k, f"SPEAKER_{len(names):02d}")
    return [{"start": s, "end": e, "speaker": names[k]} for s, e, k in raw_turns]

def split_collapsed_with_titanet(segs, wav, sr=16000):
    import torch
    try:
        from nemo.collections.asr.models import EncDecSpeakerLabelModel
        titanet = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large").to("cuda").eval()
    except Exception: return segs
    spks = {s["speaker"] for s in segs}
    if len(spks) != 1 or len(segs) < 4: return segs
    embs, durs, valid_ids = [], [], []
    with torch.no_grad():
        for idx, s in enumerate(segs):
            i0, i1 = max(0, int(round(s["start"] * sr))), min(len(wav), int(round(s["end"] * sr)))
            chunk = wav[i0:i1]
            if len(chunk) < int(0.35 * sr): continue
            sig, sig_len = torch.from_numpy(chunk).unsqueeze(0).to("cuda"), torch.tensor([len(chunk)], dtype=torch.long, device="cuda")
            _, emb = titanet.forward(input_signal=sig, input_signal_length=sig_len)
            v = emb.squeeze(0).cpu().numpy()
            embs.append(v / (np.linalg.norm(v) + 1e-8))
            durs.append(s["end"] - s["start"])
            valid_ids.append(idx)
    if len(embs) < 4: return segs
    E = np.stack(embs, axis=0)
    sim = E @ E.T
    min_pos = np.unravel_index(np.argmin(sim), sim.shape)
    c0, c1 = E[min_pos[0]].copy(), E[min_pos[1]].copy()
    labels = np.zeros(len(E), dtype=np.int32)
    for _ in range(10):
        labels = (E @ c1 > E @ c0).astype(np.int32)
        if np.all(labels == 0) or np.all(labels == 1): return segs
        w0, w1 = np.array(durs) * (labels == 0), np.array(durs) * (labels == 1)
        c0, c1 = (E.T @ w0) / (w0.sum() + 1e-8), (E.T @ w1) / (w1.sum() + 1e-8)
        c0 /= (np.linalg.norm(c0) + 1e-8); c1 /= (np.linalg.norm(c1) + 1e-8)
    import copy
    out = copy.deepcopy(segs)
    l_map = {valid_ids[i]: int(labels[i]) for i in range(len(valid_ids))}
    last_cl = 0
    for idx, s in enumerate(out):
        cl = l_map.get(idx, last_cl)
        last_cl = cl
        s["speaker"] = f"SPEAKER_{cl:02d}"
    names = {}
    for s in out:
        names.setdefault(s["speaker"], f"SPEAKER_{len(names):02d}")
        s["speaker"] = names[s["speaker"]]
    return out
