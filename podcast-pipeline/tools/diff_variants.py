#!/usr/bin/env python3
"""Where do some variants of ONE file differ, in diarization and in text? A report to read, no audio, no page.

    python tools/diff_variants.py --out-root OUT --file a --variants l_p2,l_p3,m_p3,l_p2_m_p3
    python tools/diff_variants.py --out-root OUT --file a --variants l_p2,l_p3 --baseline none --min-sec 1 --limit 30

Reads OUT/<variant>/<file>/02_diarization/segments.json and 05_asr/transcripts.json, so it works while other variants
are still running. Writes OUT/diff_<file>.txt (and prints the same):

  1. a table: speakers, segments, words, overlap, and how far each variant is from the baseline
  2. the same distances between every pair of the chosen variants
  3. per variant, against the baseline: the stretches where the speaker labels disagree (missed / extra / swapped /
     overlap), and the baseline segments whose words changed, with the changed words marked
        [a→b] replaced   [-w] missing in the variant   [+w] added by the variant   … unchanged words left out

Speaker labels are per variant (each ran its own diarization), so they are matched to the baseline's by the largest
overlap before they are compared. Without a reference transcript "different" does not mean "worse".
"""
import argparse
import os
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.compare_lib.io_stage import load_file_output  # noqa: E402
from tools.compare_lib.speaker_align import compare_speakers  # noqa: E402
from tools.compare_lib.text_align import compare_transcripts, tokens  # noqa: E402

KIND_TEXT = {"missing": "thiếu (baseline có người nói, variant không)",
             "extra": "thừa (variant có người nói, baseline không)",
             "swap": "đổi người nói", "overlap": "khác về chồng tiếng"}


def mmss(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m)}:{s:04.1f}"


def clip(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[:max(1, width - 1)] + "…"


def marked(ops: List[list], keep: int = 3) -> str:
    """The words of a segment with the changed ones marked; long unchanged runs shortened to …"""
    out: List[str] = []
    run: List[str] = []

    def flush(last: bool):
        if len(run) > 2 * keep + 1:
            out.extend(run[:keep] + ["…"] + run[-keep:])
        else:
            out.extend(run)
        run.clear()

    for op in ops:
        if op[0] == "equal":
            run.append(op[1])
            continue
        if run:
            flush(False)
        out.append(f"[{op[1]}→{op[2]}]" if op[0] == "sub" else f"[-{op[1]}]" if op[0] == "del" else f"[+{op[1]}]")
    if run:
        flush(True)
    return " ".join(out)


def load(out_root: str, vid: str, stem: str) -> Optional[dict]:
    path = os.path.join(out_root, vid, stem)
    if not os.path.isdir(path):
        return None
    d = load_file_output(path, vid)
    return d if d["diarization"] is not None or d["asr"] is not None else None


def overlap_seconds(segs: Optional[List[dict]]) -> Optional[float]:
    if not segs:
        return None
    ev = sorted([(s["start"], 1) for s in segs] + [(s["end"], -1) for s in segs])
    depth, last, total = 0, 0.0, 0.0
    for t, d in ev:
        if depth >= 2:
            total += t - last
        depth, last = depth + d, t
    return total


def words_of(asr: Optional[List[dict]]) -> Optional[int]:
    return None if asr is None else sum(len(tokens(s["text"])) for s in asr)


def text_diff(base: Optional[List[dict]], var: Optional[List[dict]], detail: bool = False) -> Optional[dict]:
    if base is None or var is None:
        return None
    return compare_transcripts(base, var, detail=detail)


def pct(x: Optional[float]) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}%"


def secs(x: Optional[float]) -> str:
    return "   -  " if x is None else f"{x:6.1f}s"


def num(x) -> str:
    return "  -" if x is None else f"{x:3d}" if isinstance(x, int) else f"{x:5.1f}"


def summary_table(rows: List[dict], base_id: str) -> List[str]:
    out = [f"{'variant':<14}{'người nói':>9}{'đoạn':>7}{'từ':>8}{'chồng':>8}   {'khác text':>9}{'khác diar':>11}{'DER':>8}"]
    for r in rows:
        mark = "  (baseline)" if r["id"] == base_id else ""
        out.append(f"{r['id']:<14}{num(r['speakers']):>9}{num(r['segments']):>7}{num(r['words']):>8}{secs(r['overlap']):>8}"
                   f"   {pct(r['text']):>9}{secs(r['diar_s']):>11}{pct(r['der']):>8}{mark}")
    return out


def pair_matrix(ids: List[str], data: Dict[str, dict], pick) -> List[str]:
    width = max(7, max(len(i) for i in ids) + 1)
    out = [" " * 14 + "".join(f"{i:>{width}}" for i in ids)]
    for a in ids:
        cells = []
        for b in ids:
            cells.append(f"{'':>{width}}" if a == b else f"{pick(data[a], data[b]):>{width}}")
        out.append(f"{a:<14}" + "".join(cells))
    return out


def diar_lines(rep: dict, min_sec: float, limit: int, width: int) -> List[str]:
    ranges = [r for r in rep["disagree_ranges"] if r["end"] - r["start"] >= min_sec]
    ranges.sort(key=lambda r: -(r["end"] - r["start"]))
    shown = sorted(ranges[:limit], key=lambda r: r["start"])
    out = [f"  diarization: {len(rep['disagree_ranges'])} đoạn khác nhau, tổng {rep['disagree_seconds']:.1f}s "
           f"(hiện {len(shown)} đoạn ≥ {min_sec:g}s dài nhất, theo thời gian)"]
    for r in shown:
        out.append(f"    {mmss(r['start'])} – {mmss(r['end'])}  ({r['end'] - r['start']:5.1f}s)  {r['kind']:<8} "
                   f"baseline={r['baseline'] or '-'}  variant={r['variant'] or '-'}")
    if not shown:
        out.append("    (không có đoạn nào đủ dài)")
    return out


def text_lines(rep: dict, min_wer: float, limit: int, width: int) -> List[str]:
    rows = []
    for seg in rep["segments"]:
        errs = sum(1 for o in seg["ops"] if o[0] != "equal")
        n_ref = sum(1 for o in seg["ops"] if o[0] != "ins")
        if errs and (seg["wer"] >= min_wer or n_ref == 0):
            rows.append((errs, seg))
    rows.sort(key=lambda p: -p[0])
    shown = sorted(rows[:limit], key=lambda p: p[1]["start"])
    c = rep["ops"]
    out = [f"  text: thay {c['sub']} · thiếu {c['del']} · thêm {c['ins']} · giữ {c['equal']} từ; "
           f"{len(rows)} đoạn baseline đổi ≥ {int(min_wer * 100)}% (hiện {len(shown)} đoạn nhiều lỗi nhất, theo thời gian)"]
    for errs, seg in shown:
        out.append(f"    {mmss(seg['start'])} – {mmss(seg['end'])}  {errs} lỗi")
        out.append("      " + clip(marked(seg["ops"]), width))
    if not shown:
        out.append("    (không có đoạn nào đổi đủ nhiều)")
    return out


def report(out_root: str, stem: str, ids: List[str], base_id: str, min_sec=0.5, min_wer=0.2, limit=40, width=160) -> str:
    wanted = [base_id] + [i for i in ids if i != base_id]
    data, missing = {}, []
    for vid in wanted:
        d = load(out_root, vid, stem)
        (data.__setitem__(vid, d) if d else missing.append(vid))
    if base_id not in data:
        return f"không có kết quả của baseline '{base_id}' cho file '{stem}' trong {out_root}"
    base = data[base_id]
    dur = base["duration"]

    def diar(a, b):
        return compare_speakers(a["diarization"], b["diarization"], dur)

    rows = []
    for vid, d in data.items():
        dr = None if vid == base_id else diar(base, d)
        tx = None if vid == base_id else text_diff(base["asr"], d["asr"])
        rows.append({"id": vid, "speakers": len({s["speaker"] for s in d["diarization"] or []}) or None,
                     "segments": len(d["diarization"]) if d["diarization"] is not None else None,
                     "words": words_of(d["asr"]), "overlap": overlap_seconds(d["diarization"]),
                     "text": tx["wer"] if tx else None, "diar_s": dr["disagree_seconds"] if dr else None,
                     "der": dr["der_vs_baseline"] if dr else None})
    lines = [f"File {stem}  ·  baseline {base_id}  ·  {mmss(dur) if dur else '?'} dài", ""]
    if missing:
        lines += [f"CHƯA CÓ kết quả (bỏ qua): {', '.join(missing)}", ""]
    lines += ["1. Tổng quan (khác text = tỉ lệ từ khác baseline; khác diar = số giây nhãn người nói khác; DER = diarization lệch)", ""]
    lines += summary_table(rows, base_id) + [""]
    chosen = [i for i in data if i != base_id]
    if len(chosen) >= 2:
        lines += ["2. Khoảng cách giữa các variant với nhau", "", "   khác text (từ khác nhau, hàng = chuẩn):"]
        lines += pair_matrix(chosen, data, lambda a, b: pct((text_diff(a["asr"], b["asr"]) or {}).get("wer")).strip())
        lines += ["", "   khác diarization (số giây nhãn khác nhau):"]
        lines += pair_matrix(chosen, data, lambda a, b: secs((diar(a, b) or {}).get("disagree_seconds")).strip())
        lines += [""]
    for vid in chosen:
        d = data[vid]
        lines += [f"3. {vid} so với {base_id}", ""]
        sp = diar(base, d)
        lines += diar_lines(sp, min_sec, limit, width) if sp else ["  diarization: thiếu dữ liệu"]
        tx = text_diff(base["asr"], d["asr"], detail=True)
        lines += text_lines(tx, min_wer, limit, width) if tx else ["  text: thiếu dữ liệu"]
        lines += [""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--file", required=True, help="file stem, e.g. a")
    ap.add_argument("--variants", required=True, help="comma-separated variant ids")
    ap.add_argument("--baseline", default="none")
    ap.add_argument("--min-sec", type=float, default=0.5, help="smallest diarization disagreement to list (s)")
    ap.add_argument("--min-wer", type=float, default=0.2, help="smallest share of changed words to list a segment")
    ap.add_argument("--limit", type=int, default=40, help="most items listed per variant and kind")
    ap.add_argument("--width", type=int, default=160, help="characters of text per item")
    ap.add_argument("--out", help="where to write the report (default: OUT/diff_<file>.txt)")
    a = ap.parse_args(argv)
    ids = [v for v in a.variants.split(",") if v]
    text = report(os.path.abspath(a.out_root), a.file, ids, a.baseline, a.min_sec, a.min_wer, a.limit, a.width)
    path = a.out or os.path.join(os.path.abspath(a.out_root), f"diff_{a.file}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
