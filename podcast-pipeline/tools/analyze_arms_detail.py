#!/usr/bin/env python3
"""Phân tích chi tiết các arm so với nhãn tay: vùng nào sai, sai vì sao, và âm học tại chỗ sai ra sao.

    python tools/analyze_arms_detail.py OUT/diar_arms --arms nemotron,diarizen_auto,nemotron_dz \
        [--postprocess] [--exclude 01_Src_15] [--gap-min 0.1 --gap-max 0.9 --gap-seg 2.0] \
        [--out OUT/diar_arms/analysis_detail] [--top 25] [--boot 2000] [--ctx 1.5]

Đơn vị: giây-người (speaker-second) như compare_arms.py. DER = (miss + FA + conf) / giây nhãn, chỉ tính khung được
chấm (bỏ quanh mép nhãn theo tolerance và các vùng "không chắc" của nhãn). Các số đều tính trên cùng khung lưới 10 ms.

Phần 1  kiểm tra chéo DER với compare_arms.py (phải khớp).
Phần 2  thống kê từng arm: trung bình, trung vị, phân vị, khoảng tin cậy bootstrap; số người nói; quy tắc khoảng nối.
Phần 3  nguyên nhân lỗi, phân loại từng vùng sai (một chuỗi khung liên tiếp cùng loại lỗi, gộp nếu cách ≤ 50 ms):
        miss  (nhãn có, arm im)
              đoạn đệm bị bỏ      nhãn của người đó dài < 1 s (backchannel)
              ranh giới           vùng ≤ 1 s, cách mép nhãn ≤ 0,5 s
              arm im giữa lời     arm có tiếng cả trước lẫn sau vùng (trong ±ctx s)
              arm bỏ mép lời      arm có tiếng một phía
              arm bỏ cả đoạn      arm không có tiếng trong ±ctx s
        FA    (arm có, nhãn im)
              ranh giới
              arm nói khi nhãn im nhãn không có tiếng trong ±ctx s (ảo giác, nhạc, tạp âm)
              arm lấn mép nhãn    nhãn có tiếng một phía
        conf  (cả hai có tiếng nhưng gán khác người)
              đoạn đệm gán nhầm, ranh giới, chồng tiếng: gán nhầm (nhãn ≥ 2 người), gán nhầm người
Phần 4  lỗi số người: nhãn thiếu / tách, arm thừa / gộp (chỉ người ≥ 1 s).
Phần 5  khoảng nối: arm im 0,1–0,9 s trong đoạn nhãn cùng người ≥ 2 s, chỉ một người nói; DER trước/sau.
Phần 6  âm học:
        – mức tín hiệu, mức nền (10 % phân vị khung im nhãn), mức tiếng nói, SNR ước lượng cho mỗi mẫu;
        – tại mỗi vùng sai: mức dB, chênh với ngữ cảnh ±ctx s, độ phẳng phổ, trọng tâm phổ, ZCR, tỉ lệ tần thấp
          (< 300 Hz, dấu hiệu hum/rumble), tỉ lệ khung có tiếng, F0 nhãn và F0 arm (cho lỗi gán nhầm: lệch
          bao nhiêu semitone);
        – tỉ lệ lỗi theo từng khoảng đặc trưng (mức dB, độ phẳng, trọng tâm, tỉ lệ tần thấp, F0, số người cùng lúc),
          tách theo miss / FA / conf: đây là chỗ thấy âm học nào làm arm hay sai;
        – trung vị đặc trưng theo từng nguyên nhân lỗi.
Phần 7  vị trí trong clip (mười đoạn bằng nhau) và giây sai theo vị trí.
Phần 8  so cặp arm theo mẫu (bootstrap, dấu p hai phía).
Phần 9  mẫu DER cao nhất; vùng sai lớn nhất.

Ghi ra thư mục --out: report.txt, summary_by_arm.csv, pairs.csv, per_sample.csv, error_regions.csv, causes.csv,
speaker_count.csv, gap_rule.csv, position.csv, acoustic_bins.csv, acoustic_by_cause.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
import statistics as st
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))      # utils/ (cho --postprocess)
sys.path.insert(0, HERE)
import compare_arms as ca                     # noqa: E402

STEP = ca.STEP
SHORT_S = ca.SHORT_S
MIN_SPK = ca.MIN_SPEAKER_S
EDGE_NEAR = 0.5                                # giây: vùng sai cách mép nhãn ≤ này là "ranh giới"
MERGE_SLACK = 5                                # khung (50 ms) gộp các vùng liền nhau
MIN_F0_FRAMES = 20                             # khung có tiếng tối thiểu để lấy F0 của một người
METRICS = ["der", "miss", "fa", "conf", "der_gap", "short_any", "short_spk",
           "speaker_recall", "ovl_ratio", "spk_err", "spk_err_abs"]
MISS_CAUSES = ["đoạn đệm bị bỏ", "ranh giới", "arm im giữa lời", "arm bỏ mép lời", "arm bỏ cả đoạn"]
FA_CAUSES = ["ranh giới", "arm lấn mép nhãn", "arm nói khi nhãn im"]
CONF_CAUSES = ["đoạn đệm gán nhầm", "ranh giới", "chồng tiếng: gán nhầm", "gán nhầm người"]
ALL_CAUSES = [("miss", c) for c in MISS_CAUSES] + [("fa", c) for c in FA_CAUSES] + [("conf", c) for c in CONF_CAUSES]
AC_FIELDS = ["rms_db", "ctx_db", "d_db", "flat", "cent", "low", "zcr", "voiced_frac", "f0"]
# khoảng để chia đặc trưng thành nhóm; giá trị ngoài khoảng bị bỏ
BIN_SPEC = {
    "rms_db": [-80, -50, -40, -35, -30, -25, -20, -15, -10, 0.1],
    "flat": [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01],
    "centroid_hz": [0, 500, 800, 1200, 1800, 2500, 4000, 8001],
    "low_ratio": [0, 0.05, 0.1, 0.2, 0.4, 1.01],
    "f0_hz": [0, 100, 130, 160, 190, 220, 260, 320, 400, 1000],
    "ref_people": [-0.5, 0.5, 1.5, 2.5, 99],
}
BIN_NAMES = {"ref_people": ["0 người", "1 người", "2 người", "3+ người"]}


# ----------------------------------------------------------------------------- nạp dữ liệu

def load_samples(root: str, arms: list[str] | None, postprocess: bool, exclude: list[str]) -> list[dict]:
    out = []
    for sdir in sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)):
        name = os.path.basename(sdir.rstrip("/"))
        if any(x and x in name for x in exclude):
            continue
        doc = ca.load_labels(sdir)
        if doc is None:
            continue
        ref = [{"start": float(s["start"]), "end": float(s["end"]), "speaker": str(s["speaker"])}
               for s in doc["segments"]]
        ignore = [(float(i["start"]), float(i["end"])) for i in doc.get("ignore", [])]
        tol = float(doc.get("tolerance_s", 0.25))
        files = {ca.arm_of(f): f for f in glob.glob(os.path.join(sdir, "segments_*.json"))}
        names = sorted(a for a in files if not arms or a in arms)
        segs = {a: ca.load_segments(files[a]) for a in names}
        if postprocess:
            segs = {a: ca.pipeline_postprocess(v) if ca.is_raw_arm(a) else v for a, v in segs.items()}
        if not segs:
            continue
        duration = ca.clip_duration(sdir, [ref] + list(segs.values()))
        out.append(dict(name=name, dir=sdir, ref=ref, ignore=ignore, tol=tol, segs=segs, duration=duration))
    return out


def load_wav(path: str) -> tuple[np.ndarray, int] | None:
    if not os.path.isfile(path):
        return None
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float64", always_2d=True)
    except ImportError:
        from scipy.io import wavfile
        sr, raw = wavfile.read(path)
        x = raw.astype(np.float64)
        if np.issubdtype(raw.dtype, np.integer):
            x = x / float(np.iinfo(raw.dtype).max)
        x = x.reshape(len(x), -1)
    return x.mean(axis=1), int(sr)


# ----------------------------------------------------------------------------- đặc trưng âm học

def frame_acoustics(x: np.ndarray, sr: int, n: int) -> dict:
    """Đặc trưng cho từng khung 10 ms, khung k có tâm tại k·0,01 s (cùng lưới với nhãn)."""
    win = int(round(0.025 * sr))
    hop = int(round(STEP * sr))
    half = win // 2
    need = n * hop + win
    buf = np.zeros(need, dtype=np.float64)
    m = min(len(x), need - half)
    buf[half:half + m] = x[:m]
    fr = np.lib.stride_tricks.sliding_window_view(buf, win)[::hop][:n]
    fw = fr * np.hanning(win)
    rms = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    P = np.abs(np.fft.rfft(fw, axis=1)) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    band = (freqs >= 60) & (freqs <= 4000)
    flat = np.exp(np.mean(np.log(P[:, band]), axis=1)) / np.mean(P[:, band], axis=1)
    total = P.sum(axis=1)
    cent = (P * freqs).sum(axis=1) / total
    low = P[:, freqs < 300].sum(axis=1) / total
    zcr = np.mean(np.signbit(fr[:, 1:]) != np.signbit(fr[:, :-1]), axis=1).astype(np.float64)
    ac = np.fft.irfft(P, n=2 * win, axis=1)[:, :win]              # autocorrelation của từng khung
    lo, hi = int(sr / 400), int(sr / 60) + 1                       # trễ tương ứng 60–400 Hz
    seg = ac[:, lo:hi]
    best = np.argmax(seg, axis=1)
    r = seg[np.arange(n), best] / (ac[:, 0] + 1e-12)
    voiced = (r > 0.5) & (rms > -50)
    f0 = np.where(voiced, sr / (best + lo), np.nan)
    return dict(rms=rms, flat=flat, cent=cent, low=low, zcr=zcr, f0=f0, voiced=voiced)


def _med(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    return float(np.median(x)) if x.size else float("nan")


def region_feats(AF: dict, a: int, b: int, ctx: int) -> dict:
    rms = AF["rms"]
    near = np.r_[rms[max(0, a - ctx):a], rms[b:b + ctx]]
    sl = slice(a, b)
    v = AF["voiced"][sl]
    return dict(rms_db=round(_med(rms[sl]), 2), ctx_db=round(_med(near), 2),
                d_db=round(_med(rms[sl]) - _med(near), 2) if near.size and b > a else float("nan"),
                flat=round(_med(AF["flat"][sl]), 3), cent=round(_med(AF["cent"][sl]), 1),
                low=round(_med(AF["low"][sl]), 3), zcr=round(_med(AF["zcr"][sl]), 3),
                voiced_frac=round(float(v.mean()), 3) if v.size else float("nan"),
                f0=round(_med(AF["f0"][sl][v]), 1))


def speaker_f0(M: np.ndarray, labels: list, AF: dict) -> dict:
    out = {}
    for i, name in enumerate(labels):
        m = M[:, i] & AF["voiced"]
        out[str(name)] = round(_med(AF["f0"][m]), 1) if m.sum() >= MIN_F0_FRAMES else float("nan")
    return out


def semitones(f_a: float, f_b: float) -> float:
    if not (f_a > 0 and f_b > 0):
        return float("nan")
    return round(12 * math.log2(f_a / f_b), 2)


# ----------------------------------------------------------------------------- khung và so khớp

def frame_state(S: dict, arm: str) -> dict:
    n = int(math.ceil(S["duration"] / STEP))
    ref, rl = ca.to_frames(S["ref"], n)
    hyp, hl = ca.to_frames(S["segs"][arm], n)
    sc = ca.collar_mask(S["ref"], n, S["tol"])
    for a, b in S["ignore"]:
        sc[max(0, int(math.floor(a / STEP))):min(n, int(math.ceil(b / STEP)))] = False
    mp = ca.best_mapping(ref, hyp, sc)
    n_ref, n_hyp = ref.sum(1), hyp.sum(1)
    correct = np.zeros(n, dtype=np.int64)
    for h, r in mp.items():
        correct += (ref[:, r] & hyp[:, h]).astype(np.int64)
    miss = np.maximum(0, n_ref - n_hyp).astype(np.int64)
    fa = np.maximum(0, n_hyp - n_ref).astype(np.int64)
    conf = (np.minimum(n_ref, n_hyp) - correct).astype(np.int64)
    return dict(n=n, ref=ref, rl=rl, hyp=hyp, hl=hl, sc=sc, mp=mp, n_ref=n_ref, n_hyp=n_hyp,
                miss=miss, fa=fa, conf=conf)


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def merge_runs(runs: list[tuple[int, int]], slack: int) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in runs:
        if out and a - out[-1][1] <= slack:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def cover_len(ref: list[dict], t0: float, t1: float, speaker: str) -> float:
    best = 0.0
    for s in ref:
        if s["speaker"] == speaker and s["start"] <= t0 + 1e-9 and s["end"] >= t1 - 1e-9:
            best = max(best, s["end"] - s["start"])
    return best


def describe(vals: list[float], rng: random.Random, boot: int) -> dict:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {}
    n = len(v)
    if n >= 2:
        q = st.quantiles(v, n=4)
        q1, q3, sd = q[0], q[2], st.stdev(v)
    else:
        q1 = q3 = v[0]
        sd = 0.0
    means = sorted(st.mean(rng.choice(v) for _ in range(n)) for _ in range(boot))
    return dict(n=n, mean=st.mean(v), median=st.median(v), sd=sd, q1=q1, q3=q3, min=min(v), max=max(v),
                ci_lo=means[int(0.025 * boot)], ci_hi=means[int(0.975 * boot)])


# ----------------------------------------------------------------------------- phân tích một (mẫu, arm)

def analyze(S: dict, AF: dict | None, arm: str, args) -> dict:
    F = frame_state(S, arm)
    n, ref, hyp, sc, mp = F["n"], F["ref"], F["hyp"], F["sc"], F["mp"]
    rl, hl = F["rl"], F["hl"]
    ref_any, hyp_any = ref.any(1), hyp.any(1)
    cf = int(round(args.ctx / STEP))
    edges = np.array(sorted({t for s in S["ref"] for t in (s["start"], s["end"])}))
    total = float(F["n_ref"][sc].sum()) * STEP
    inv = {r: h for h, r in mp.items()}
    ref_f0 = speaker_f0(ref, rl, AF) if AF is not None else {}
    hyp_f0 = speaker_f0(hyp, hl, AF) if AF is not None else {}

    regions, gap_rows = [], []
    sec_tot = {"miss": 0.0, "fa": 0.0, "conf": 0.0}
    gap_rule_n = gap_rule_sec = gap_any_n = gap_any_sec = 0.0
    for kind in ("miss", "fa", "conf"):
        arr = F[kind]
        mask = (arr > 0) & sc
        for a, b in merge_runs(runs_of(mask), MERGE_SLACK):
            sec = float((arr[a:b] * sc[a:b]).sum()) * STEP
            if sec <= 0:
                continue
            t0, t1 = a * STEP, b * STEP
            dur = t1 - t0
            sec_tot[kind] += sec
            ref_names = [rl[i] for i in np.flatnonzero(ref[a:b].any(0))]
            hyp_names = [hl[i] for i in np.flatnonzero(hyp[a:b].any(0))]
            overlap_ref = bool(b > a and ref[a:b].sum(1).max() >= 2)
            pre_h = bool(hyp_any[max(0, a - cf):a].any())
            post_h = bool(hyp_any[b:min(n, b + cf)].any())
            pre_r = bool(ref_any[max(0, a - cf):a].any())
            post_r = bool(ref_any[b:min(n, b + cf)].any())
            ref_seg_lens = [cover_len(S["ref"], t0, t1, nm) for nm in ref_names]
            short_ref = any(0 < L < SHORT_S for L in ref_seg_lens)
            edge_d = float(np.min(np.maximum(0, np.maximum(t0 - edges, edges - t1)))) if edges.size else 99.0
            near_edge = edge_d <= EDGE_NEAR and dur <= 1.0

            if kind == "miss":
                if short_ref:
                    cause = "đoạn đệm bị bỏ"
                elif near_edge:
                    cause = "ranh giới"
                elif pre_h and post_h:
                    cause = "arm im giữa lời"
                elif not pre_h and not post_h:
                    cause = "arm bỏ cả đoạn"
                else:
                    cause = "arm bỏ mép lời"
            elif kind == "fa":
                if near_edge:
                    cause = "ranh giới"
                elif not pre_r and not post_r:
                    cause = "arm nói khi nhãn im"
                else:
                    cause = "arm lấn mép nhãn"
            else:
                if short_ref:
                    cause = "đoạn đệm gán nhầm"
                elif near_edge:
                    cause = "ranh giới"
                elif overlap_ref:
                    cause = "chồng tiếng: gán nhầm"
                else:
                    cause = "gán nhầm người"

            gap_hit = False
            if kind == "miss" and args.gap_min - 1e-9 <= dur <= args.gap_max + 1e-9:
                gap_any_n += 1
                gap_any_sec += sec
                if (len(ref_names) == 1 and not hyp_any[a:b].any()
                        and cover_len(S["ref"], t0, t1, ref_names[0]) >= args.gap_seg):
                    gap_hit = True
                    gap_rule_n += 1
                    gap_rule_sec += sec
                    gap_rows.append(dict(sample=S["name"], arm=arm, start=round(t0, 2), end=round(t1, 2),
                                         length_s=round(dur, 2), speaker=str(ref_names[0]), sec=round(sec, 2)))

            rec = dict(sample=S["name"], arm=arm, type=kind, start=round(t0, 2), end=round(t1, 2),
                       dur=round(dur, 2), sec=round(sec, 2), cause=cause, short_ref=short_ref,
                       overlap_ref=overlap_ref, edge_d=round(edge_d, 2),
                       ref_spk=" ".join(map(str, ref_names)), arm_spk=" ".join(map(str, hyp_names)),
                       ref_seg_len=round(max(ref_seg_lens), 2) if ref_seg_lens else "",
                       n_ref_people=len(ref_names), n_arm_people=len(hyp_names),
                       gap_rule=gap_hit, _mid=(t0 + t1) / 2)
            if AF is not None:
                rec.update(region_feats(AF, a, b, cf))
                # lỗi gán nhầm: F0 của người nhãn so với F0 của người arm gán (một người mỗi bên)
                if kind == "conf" and len(ref_names) >= 1 and len(hyp_names) >= 1:
                    f_ref = ref_f0.get(str(ref_names[0]), float("nan"))
                    f_arm = hyp_f0.get(str(hyp_names[0]), float("nan"))
                    rec["f0_ref_spk"], rec["f0_arm_spk"] = f_ref, f_arm
                    rec["dF0_st"] = semitones(f_arm, f_ref)
            regions.append(rec)

    err_total = sum(sec_tot.values())
    der = 100.0 * err_total / total if total > 0 else float("nan")
    der_gap = 100.0 * (err_total - gap_rule_sec) / total if total > 0 else float("nan")

    # kiểu lệch số người
    ref_full, hyp_full = ref.sum(0) * STEP, hyp.sum(0) * STEP
    O = ref.T.astype(np.int64) @ hyp.astype(np.int64)            # khung trùng (nhãn, arm)
    ref_big = [i for i in range(len(rl)) if ref_full[i] >= MIN_SPK]
    hyp_big = [j for j in range(len(hl)) if hyp_full[j] >= MIN_SPK]
    spk_rows = []
    for i in ref_big:
        cover = [j for j in hyp_big if O[i, j] * STEP >= MIN_SPK]
        kind = "thiếu" if not cover else ("tách" if len(cover) >= 2 else "khớp")
        spk_rows.append(dict(sample=S["name"], arm=arm, side="nhãn", speaker=str(rl[i]),
                             dur=round(float(ref_full[i]), 1), mapped=str(hl[inv[i]]) if i in inv else "-",
                             kind=kind, with_=" ".join(str(hl[j]) for j in cover),
                             f0=ref_f0.get(str(rl[i]), "")))
    for j in hyp_big:
        cover = [i for i in ref_big if O[i, j] * STEP >= MIN_SPK]
        kind = "thừa" if not cover else ("gộp" if len(cover) >= 2 else "khớp")
        spk_rows.append(dict(sample=S["name"], arm=arm, side="arm", speaker=str(hl[j]),
                             dur=round(float(hyp_full[j]), 1), mapped=str(rl[mp[j]]) if j in mp else "-",
                             kind=kind, with_=" ".join(str(rl[i]) for i in cover),
                             f0=hyp_f0.get(str(hl[j]), "")))

    # kiểm tra chéo với compare_arms
    summ = ca.compare(S["ref"], S["segs"][arm], S["duration"], S["tol"], ignore=S["ignore"])
    rec_vals = list(summ["speaker_recall"].values())
    kinds = [r["kind"] for r in spk_rows]
    pct = (lambda s: 100.0 * s / total) if total > 0 else (lambda s: float("nan"))
    row = dict(sample=S["name"], arm=arm,
               ref_speakers=summ["ref_speakers"], arm_speakers=summ["arm_speakers"],
               spk_err=summ["arm_speakers"] - summ["ref_speakers"],
               spk_err_abs=abs(summ["arm_speakers"] - summ["ref_speakers"]),
               der=round(der, 2), miss=round(pct(sec_tot["miss"]), 2), fa=round(pct(sec_tot["fa"]), 2),
               conf=round(pct(sec_tot["conf"]), 2), der_check=summ["der"], der_gap=round(der_gap, 2),
               ref_s=summ["ref_s"], arm_s=summ["arm_s"], ref_total_spk_s=round(total, 1),
               short_any=summ["short_any"], short_spk=summ["short_spk"],
               speaker_recall=round(st.mean(rec_vals), 3) if rec_vals else None,
               ovl_ratio=summ["overlap_ratio"], ovl_s=summ["overlap_s"], ref_ovl_s=summ["ref_overlap_s"],
               split_n=kinds.count("tách"), merge_n=kinds.count("gộp"),
               missing_n=kinds.count("thiếu"), extra_n=kinds.count("thừa"),
               gap_rule_n=int(gap_rule_n), gap_rule_sec=round(gap_rule_sec, 2),
               gap_any_n=int(gap_any_n), gap_any_sec=round(gap_any_sec, 2))

    # âm học theo mẫu: mức nền (khung nhãn im), mức tiếng nói (khung có nhãn), SNR
    if AF is not None:
        quiet = AF["rms"][(F["n_ref"] == 0) & sc]
        speech = AF["rms"][(F["n_ref"] > 0) & sc]
        noise = float(np.percentile(quiet, 10)) if quiet.size else float("nan")
        level = _med(speech)
        row.update(noise_db=round(noise, 2), speech_db=round(level, 2), snr_db=round(level - noise, 2),
                   quiet_s=round(float(quiet.size) * STEP, 1))

    # tỉ lệ lỗi theo đặc trưng khung (chỉ khung được chấm)
    bins = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])       # ref_s, miss_s, fa_s, conf_s, frame_s
    if AF is not None:
        feats = {"rms_db": AF["rms"], "flat": AF["flat"], "centroid_hz": AF["cent"], "low_ratio": AF["low"],
                 "f0_hz": np.where(AF["voiced"], AF["f0"], np.nan), "ref_people": F["n_ref"].astype(float)}
        for fname, vals in feats.items():
            edges_f = BIN_SPEC[fname]
            for k in range(len(edges_f) - 1):
                sel = sc & (vals >= edges_f[k]) & (vals < edges_f[k + 1])
                if not sel.any():
                    continue
                acc = bins[(fname, k)]
                acc[0] += float(F["n_ref"][sel].sum()) * STEP
                acc[1] += float(F["miss"][sel].sum()) * STEP
                acc[2] += float(F["fa"][sel].sum()) * STEP
                acc[3] += float(F["conf"][sel].sum()) * STEP
                acc[4] += float(sel.sum()) * STEP

    return dict(row=row, regions=regions, spk=spk_rows, gap=gap_rows, bins=bins)


# ----------------------------------------------------------------------------- tiện ích báo cáo

def fmt_t(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def fnum(x, width=7, nd=2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return " " * (width - 1) + "-"
    return f"{x:{width}.{nd}f}"


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def pair_table(rows: list[dict], arms: list[str], rng: random.Random, boot: int, out: list[str]) -> list[dict]:
    """Hiệu từng cặp arm trên từng chỉ số, theo mẫu; ghi dòng văn bản vào `out`, trả về bản ghi CSV."""
    recs = []
    by = defaultdict(dict)
    for r in rows:
        by[r["sample"]][r["arm"]] = r
    for a, b in combinations(arms, 2):
        for m in METRICS:
            diffs = []
            for d in by.values():
                if a in d and b in d and d[a].get(m) is not None and d[b].get(m) is not None:
                    x, y = float(d[a][m]), float(d[b][m])
                    if math.isnan(x) or math.isnan(y):
                        continue
                    diffs.append(x - y)
            if len(diffs) < 2:
                continue
            ds = describe(diffs, rng, boot)
            wa = sum(x < 0 for x in diffs)
            wb = sum(x > 0 for x in diffs)
            p = sign_test_p(wa, wb)
            recs.append(dict(a=a, b=b, metric=m, n=len(diffs), mean_diff=round(ds["mean"], 3),
                             median_diff=round(ds["median"], 3), ci95_lo=round(ds["ci_lo"], 3),
                             ci95_hi=round(ds["ci_hi"], 3), a_wins=wa, b_wins=wb, ties=len(diffs) - wa - wb,
                             sign_p=round(p, 4)))
            out.append(f"  {m:14s} {a:>16s} − {b:<16s} n={len(diffs):2d}  TB={ds['mean']:+7.2f}  "
                       f"TV={ds['median']:+7.2f}  KTC95=[{ds['ci_lo']:+7.2f}; {ds['ci_hi']:+7.2f}]  "
                       f"{a} thắng {wa}, {b} thắng {wb}  p(dấu)={p:.3f}")
    return recs


def sign_test_p(wins_a: int, wins_b: int) -> float:
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


# ----------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--arms", required=True, help="danh sách arm, cách nhau bằng dấu phẩy")
    ap.add_argument("--postprocess", action="store_true")
    ap.add_argument("--exclude", default="", help="mẫu có chuỗi này trong tên bị loại (cách nhau bằng dấu phẩy)")
    ap.add_argument("--gap-min", type=float, default=0.1)
    ap.add_argument("--gap-max", type=float, default=0.9)
    ap.add_argument("--gap-seg", type=float, default=2.0)
    ap.add_argument("--ctx", type=float, default=1.5, help="giây nhìn quanh vùng sai để xét bên trái/phải")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)

    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    excl = [x.strip() for x in args.exclude.split(",") if x.strip()]
    out_dir = args.out or os.path.join(args.root, "analysis_detail")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(0)

    samples = load_samples(args.root, arms, args.postprocess, excl)
    if not samples:
        print(f"không có mẫu nào có labels.json và segments_*.json dưới {args.root}", file=sys.stderr)
        return 1

    rows, regions, spk_rows, gap_rows = [], [], [], []
    bins_all = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0]))
    sample_audio = {}
    for S in samples:
        wav = load_wav(os.path.join(S["dir"], "clip.wav"))
        AF = None
        if wav is not None:
            x, sr = wav
            n = int(math.ceil(S["duration"] / STEP))
            AF = frame_acoustics(x, sr, n)
        for arm in arms:
            if arm not in S["segs"]:
                continue
            res = analyze(S, AF, arm, args)
            rows.append(res["row"])
            regions += res["regions"]
            spk_rows += res["spk"]
            gap_rows += res["gap"]
            for key, v in res["bins"].items():
                acc = bins_all[arm][key]
                for i in range(5):
                    acc[i] += v[i]
            if res["row"].get("noise_db") is not None:
                sample_audio[(S["name"], arm)] = res["row"]

    lines: list[str] = []

    def P(s: str = ""):
        lines.append(s)
        print(s)

    P(f"Mẫu: {len(samples)}  |  arm: {', '.join(arms)}  |  postprocess: {args.postprocess}  |  loại trừ: {excl or 'không'}")
    P(f"Khoảng nối: arm im {args.gap_min}–{args.gap_max} s, đoạn nhãn cùng người ≥ {args.gap_seg} s, một người nói")
    P(f"Âm học: {'có' if any(r.get('noise_db') is not None for r in rows) else 'KHÔNG (không đọc được clip.wav)'}")

    # 1. kiểm tra chéo
    P("\n=== 1. Kiểm tra chéo DER với compare_arms (lệch > 0,05 là có vấn đề) ===")
    bad = [r for r in rows if not math.isnan(r["der_check"]) and abs(r["der"] - r["der_check"]) > 0.05]
    P(f"  số dòng lệch: {len(bad)} / {len(rows)}")
    for r in bad[:10]:
        P(f"  {r['sample']} {r['arm']}: của tôi {r['der']}  compare {r['der_check']}")

    # 2. thống kê từng arm
    summary_rows = []
    fields = METRICS + ["miss", "fa", "conf", "ref_s", "arm_s", "ref_total_spk_s", "snr_db", "noise_db", "speech_db"]
    scopes = [("TẤT CẢ MẪU", rows)]
    if excl:
        scopes.append(("MẪU SAU KHI LOẠI TRỪ", rows))
    for label, subset in scopes:
        P(f"\n=== 2. Thống kê theo arm — {label} ===")
        for arm in arms:
            rs = [r for r in subset if r["arm"] == arm]
            if not rs:
                continue
            P(f"\n-- {arm} (n={len(rs)})")
            for m in fields:
                d = describe([float(r[m]) for r in rs if r.get(m) is not None], rng, args.boot)
                if not d:
                    continue
                P(f"  {m:16s} TB={d['mean']:8.2f} TV={d['median']:8.2f} Q1={d['q1']:7.2f} Q3={d['q3']:7.2f} "
                  f"sd={d['sd']:7.2f} min={d['min']:7.2f} max={d['max']:7.2f} KTC95=[{d['ci_lo']:.2f};{d['ci_hi']:.2f}]")
                summary_rows.append(dict(scope=label, arm=arm, metric=m, **{k: round(v, 3) for k, v in d.items()}))
            n_sp = [r["arm_speakers"] - r["ref_speakers"] for r in rs]
            P(f"  số người: đúng={sum(x == 0 for x in n_sp)} thừa={sum(x > 0 for x in n_sp)} "
              f"thiếu={sum(x < 0 for x in n_sp)}  | tách={sum(r['split_n'] for r in rs)} "
              f"gộp={sum(r['merge_n'] for r in rs)} thiếu_người={sum(r['missing_n'] for r in rs)} "
              f"thừa_người={sum(r['extra_n'] for r in rs)}")
            P(f"  DER có bỏ khoảng nối: TB={st.mean(r['der_gap'] for r in rs):.2f}  "
              f"(giảm {st.mean(r['der'] - r['der_gap'] for r in rs):.2f} điểm)")

    # 3. nguyên nhân lỗi
    causes = defaultdict(float)
    cause_count = defaultdict(int)
    for g in regions:
        causes[(g["arm"], g["type"], g["cause"])] += g["sec"]
        cause_count[(g["arm"], g["type"], g["cause"])] += 1
    cause_rows = []
    P("\n=== 3. Nguyên nhân lỗi: giây-người và % giây nhãn ===")
    for arm in arms:
        tot_ref = sum(r["ref_total_spk_s"] for r in rows if r["arm"] == arm)
        P(f"\n-- {arm}  (giây nhãn {tot_ref:.1f})")
        for kind, lst in (("miss", MISS_CAUSES), ("fa", FA_CAUSES), ("conf", CONF_CAUSES)):
            tot_k = sum(causes[(arm, kind, c)] for c in lst)
            P(f"  {kind:4s} tổng {tot_k:8.1f} s ({100 * tot_k / tot_ref if tot_ref else 0:5.2f}%)")
            for c in lst:
                s = causes[(arm, kind, c)]
                cnt = cause_count[(arm, kind, c)]
                pct = 100 * s / tot_ref if tot_ref else 0
                P(f"     {c:24s} {s:8.1f} s  {pct:5.2f}%  ({cnt} vùng)")
                cause_rows.append(dict(arm=arm, type=kind, cause=c, sec=round(s, 2), pct_of_ref=round(pct, 3),
                                       regions=cnt))

    # 4. theo mẫu: nguyên nhân chính
    sample_cause = defaultdict(lambda: defaultdict(float))
    for g in regions:
        sample_cause[(g["sample"], g["arm"])][g["cause"]] += g["sec"]
    per_sample = []
    for r in rows:
        d = dict(r)
        sc_ = sample_cause.get((r["sample"], r["arm"]), {})
        c, s = max(sc_.items(), key=lambda kv: kv[1]) if sc_ else ("", 0.0)
        d["top_cause"], d["top_cause_sec"] = c, round(s, 2)
        per_sample.append(d)

    # 5. âm học tại chỗ sai
    P("\n=== 4. Âm học tại vùng sai: trung vị theo nguyên nhân (dB; F0 tính Hz; dF0 tính semitone) ===")
    P("  cột: n vùng | mức vùng dB | chênh ngữ cảnh dB | độ phẳng | trọng tâm Hz | tỉ lệ tần thấp | F0 vùng | tỉ lệ khung có tiếng")
    ac_rows = []
    for arm in arms:
        P(f"\n-- {arm}")
        for kind, c in ALL_CAUSES:
            rs = [g for g in regions if g["arm"] == arm and g["type"] == kind and g["cause"] == c]
            if not rs:
                continue
            med = lambda k: _med([g.get(k, float("nan")) if g.get(k, "") != "" else float("nan") for g in rs])
            P(f"  {kind:4s} {c:24s} n={len(rs):3d} | {med('rms_db'):7.2f} | {med('d_db'):7.2f} | {med('flat'):6.3f} | "
              f"{med('cent'):7.1f} | {med('low'):6.3f} | {med('f0'):6.1f} | {med('voiced_frac'):5.2f}")
            ac_rows.append(dict(arm=arm, type=kind, cause=c, n=len(rs),
                                rms_db=round(med("rms_db"), 2), d_db=round(med("d_db"), 2),
                                flat=round(med("flat"), 3), cent=round(med("cent"), 1), low=round(med("low"), 3),
                                zcr=round(med("zcr"), 3), f0=round(med("f0"), 1),
                                voiced_frac=round(med("voiced_frac"), 3)))
    conf_f0 = [g for g in regions if g["type"] == "conf" and g.get("dF0_st") not in (None, "")
               and not (isinstance(g.get("dF0_st"), float) and math.isnan(g["dF0_st"]))]
    if conf_f0:
        P("\n-- Lỗi gán nhầm người: chênh F0 giữa người nhãn và người arm gán (semitone, trị tuyệt đối)")
        for arm in arms:
            vals = [abs(float(g["dF0_st"])) for g in conf_f0 if g["arm"] == arm]
            if vals:
                small = sum(v < 1.0 for v in vals)
                P(f"  {arm:16s} n={len(vals):3d} trung vị={st.median(vals):5.2f}  <1 semitone: {small}/{len(vals)} "
                  f"(khó tách bằng cao độ)  ≥3 semitone: {sum(v >= 3 for v in vals)}/{len(vals)} (dễ tách, arm vẫn sai)")

    P("\n-- Âm học theo mẫu: mức nền, mức tiếng, SNR (dB)")
    for (name, arm), r in sorted(sample_audio.items()):
        if arm != arms[0]:
            continue
        P(f"  {name:28s} nền={r['noise_db']:7.2f} tiếng={r['speech_db']:7.2f} SNR={r['snr_db']:6.2f}  "
          f"khung im={r['quiet_s']} s")

    # 6. tỉ lệ lỗi theo đặc trưng âm học
    P("\n=== 5. Tỉ lệ lỗi theo đặc trưng âm học ===")
    P("  DER nhóm = (miss+FA+conf) / giây nhãn của nhóm (chỉ khi nhóm có ≥ 1 s nhãn);"
      " lỗi/s = giây lỗi chia cho giây khung của nhóm (đo mật độ lỗi, dùng được cả khi nhóm không có nhãn)")
    bin_rows = []
    for arm in arms:
        P(f"\n-- {arm}")
        for fname in BIN_SPEC:
            names = BIN_NAMES.get(fname)
            edges_f = BIN_SPEC[fname]
            P(f"  [{fname}]")
            for k in range(len(edges_f) - 1):
                ref_s, miss_s, fa_s, conf_s, frame_s = bins_all[arm].get((fname, k), [0.0] * 5)
                if frame_s <= 0:
                    continue
                label = names[k] if names else f"[{edges_f[k]:g}; {edges_f[k + 1]:g})"
                err = miss_s + fa_s + conf_s
                der_txt = f"{100 * err / ref_s:7.2f}%" if ref_s >= 1.0 else "      -"
                miss_txt = f"{100 * miss_s / ref_s:6.2f}%" if ref_s >= 1.0 else "     -"
                conf_txt = f"{100 * conf_s / ref_s:6.2f}%" if ref_s >= 1.0 else "     -"
                P(f"    {label:22s} khung={frame_s:8.1f}s giây nhãn={ref_s:8.1f}  miss={miss_txt}  FA={fa_s:7.2f}s  "
                  f"conf={conf_txt}  DER nhóm={der_txt}  lỗi/s={err / frame_s:6.3f}")
                bin_rows.append(dict(arm=arm, feature=fname, bin=label, frame_s=round(frame_s, 2),
                                     ref_s=round(ref_s, 2), miss_s=round(miss_s, 2), fa_s=round(fa_s, 2),
                                     conf_s=round(conf_s, 2),
                                     der_pct=round(100 * err / ref_s, 2) if ref_s >= 1.0 else "",
                                     err_per_s=round(err / frame_s, 4)))

    # 7. lỗi số người
    P("\n=== 6. Lỗi số người nói (chỉ người ≥ 1 s) ===")
    for arm in arms:
        sp = [s for s in spk_rows if s["arm"] == arm]
        kinds_c = defaultdict(int)
        for s in sp:
            kinds_c[(s["side"], s["kind"])] += 1
        P(f"-- {arm}: nhãn thiếu={kinds_c[('nhãn', 'thiếu')]} tách={kinds_c[('nhãn', 'tách')]}  |  "
          f"arm thừa={kinds_c[('arm', 'thừa')]} gộp={kinds_c[('arm', 'gộp')]}")
    bad_sp = [s for s in spk_rows if s["kind"] != "khớp"]
    P(f"  {len(bad_sp)} người lệch (chi tiết trong speaker_count.csv):")
    for s in bad_sp[:args.top]:
        P(f"  {s['sample']:28s} {s['arm']:16s} {s['side']:4s} {s['kind']:6s} người={s['speaker']} dur={s['dur']}s "
          f"F0={s['f0']} ghép={s['mapped']} trùng=[{s['with_']}]")

    # 8. khoảng nối
    P("\n=== 7. Quy tắc khoảng nối (bỏ khỏi miss) ===")
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        P(f"-- {arm}: vùng im {args.gap_min}–{args.gap_max} s (mọi loại) = {sum(r['gap_any_n'] for r in rs)} vùng, "
          f"{sum(r['gap_any_sec'] for r in rs):.1f} s; bị loại theo quy tắc = {sum(r['gap_rule_n'] for r in rs)} vùng, "
          f"{sum(r['gap_rule_sec'] for r in rs):.1f} s; DER TB {st.mean(r['der'] for r in rs):.2f} → "
          f"{st.mean(r['der_gap'] for r in rs):.2f}")

    # 9. vị trí trong clip
    pos = defaultdict(float)
    dur_of = {S["name"]: S["duration"] for S in samples}
    for g in regions:
        dur = dur_of.get(g["sample"], 0.0)
        b = min(9, max(0, int(g["_mid"] / dur * 10))) if dur > 0 else 0
        pos[(g["arm"], g["type"], b)] += g["sec"]
    P("\n=== 8. Giây sai theo vị trí trong clip (mười đoạn bằng nhau, 0 = đầu) ===")
    pos_rows = []
    for arm in arms:
        for kind in ("miss", "fa", "conf"):
            vals = [pos[(arm, kind, b)] for b in range(10)]
            P(f"  {arm:16s} {kind:4s} " + " ".join(f"{v:6.1f}" for v in vals))
            for b, v in enumerate(vals):
                pos_rows.append(dict(arm=arm, type=kind, decile=b, sec=round(v, 2)))

    # 10. so cặp
    P("\n=== 9. So cặp theo mẫu (bootstrap KTC 95%, dấu p hai phía) ===")
    pair_lines: list[str] = []
    pairs = pair_table(rows, arms, rng, args.boot, pair_lines)
    for line in pair_lines:
        P(line)

    # 11. mẫu tệ nhất và vùng sai lớn nhất
    P("\n=== 10. Mẫu DER cao nhất của từng arm ===")
    for arm in arms:
        worst = sorted((r for r in per_sample if r["arm"] == arm), key=lambda r: -r["der"])[:5]
        P(f"-- {arm}")
        for r in worst:
            P(f"  {r['sample']:28s} DER={r['der']:6.2f} miss={r['miss']:6.2f} FA={r['fa']:6.2f} conf={r['conf']:6.2f} "
              f"người {r['arm_speakers']}/{r['ref_speakers']}  nguyên nhân chính: {r['top_cause'] or '-'}")
    P(f"\n=== 11. {args.top} vùng sai lớn nhất (theo giây) của từng arm ===")
    for arm in arms:
        top = sorted((g for g in regions if g["arm"] == arm), key=lambda g: -g["sec"])[:args.top]
        P(f"\n-- {arm}")
        for g in top:
            flags = ("  đệm" if g["short_ref"] else "") + ("  chồng" if g["overlap_ref"] else "") + \
                    ("  khoảng-nối" if g["gap_rule"] else "")
            ac = (f"  dB={g.get('rms_db', '-')} Δ={g.get('d_db', '-')} phẳng={g.get('flat', '-')} "
                  f"F0={g.get('f0', '-')}" + (f" ΔF0={g.get('dF0_st')}st" if g.get("dF0_st") not in (None, "") else ""))
            P(f"  {g['sample']:28s} {fmt_t(g['start'])}–{fmt_t(g['end'])} ({g['dur']:5.2f}s) {g['type']:4s} "
              f"{g['sec']:6.2f}s  {g['cause']:22s} nhãn=[{g['ref_spk']}] arm=[{g['arm_spk']}]{flags}{ac}")

    # ghi file
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    write_csv(os.path.join(out_dir, "summary_by_arm.csv"), summary_rows)
    write_csv(os.path.join(out_dir, "pairs.csv"), pairs)
    write_csv(os.path.join(out_dir, "per_sample.csv"), per_sample)
    write_csv(os.path.join(out_dir, "error_regions.csv"),
              [{k: v for k, v in g.items() if not k.startswith("_")} for g in regions])
    write_csv(os.path.join(out_dir, "causes.csv"), cause_rows)
    write_csv(os.path.join(out_dir, "speaker_count.csv"), spk_rows)
    write_csv(os.path.join(out_dir, "gap_rule.csv"), gap_rows)
    write_csv(os.path.join(out_dir, "position.csv"), pos_rows)
    write_csv(os.path.join(out_dir, "acoustic_bins.csv"), bin_rows)
    write_csv(os.path.join(out_dir, "acoustic_by_cause.csv"), ac_rows)
    print(f"\n-- đã ghi vào {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
