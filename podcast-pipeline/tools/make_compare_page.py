"""Build a single-file HTML page comparing noise-filter variants with the baseline.

Input is the compare_data.json written by build_compare_data.py. The page shows,
for each variant against the baseline (no filter): the text differences, the
speaker/diarization differences and where they occur, and the metrics that show
what the filter did to the later stages (diarization, separation, ASR).

Without ground truth "different" does not mean "better"; the page says so and
shows wer_vs_reference whenever the data carries it.

Usage:
    python tools/make_compare_page.py compare_data.json -o compare.html \
        [--embed clips|link|none] [--max-mb 300] [--top-segments 80] [--files a,b]

--embed clips  cut short WAV clips (top-N segments by WER, every disagreement
               range) from the raw and filtered audio and embed them as data: URIs
--embed link   refer to the wav files by relative path (page must stay next to them)
--embed none   no audio players

The page is one file with inline CSS and JS: no CDN, no network.
"""

import argparse
import base64
import html
import io
import json
import os
import pathlib
import re
import sys
import urllib.parse

DEFAULT_MAX_MB = 300
DEFAULT_TOP_SEGMENTS = 80
MAX_CLIP_SECONDS = 30.0


def _num(x, default=0.0):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if v == v and v not in (float("inf"), float("-inf")) else default


def _span_key(fi: int, start, end) -> str:
    return f"{fi}/{_num(start):.3f}-{_num(end):.3f}"


def _js_json(obj) -> str:
    """JSON that is safe inside a <script> element."""
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return (s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def load_data(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return data


def _variants(data: dict) -> list:
    """Variant list with the baseline guaranteed to be present and first."""
    base = str(data.get("baseline") or "none")
    out = []
    for v in data.get("variants") or []:
        if isinstance(v, dict) and v.get("id") is not None:
            out.append({**v, "id": str(v["id"])})
    if not any(v["id"] == base for v in out):
        out.insert(0, {"id": base, "chain": [], "label": base})
    return out


def select_files(data: dict, stems) -> list:
    files = [f for f in (data.get("files") or []) if isinstance(f, dict)]
    if not stems:
        return files
    wanted = {s.strip() for s in stems if s.strip()}
    return [f for f in files
            if f.get("name") in wanted or os.path.splitext(str(f.get("name", "")))[0] in wanted]


class _Cutter:
    """Cuts WAV clips with soundfile (imported lazily); anything wrong -> None."""

    def __init__(self):
        try:
            import soundfile as sf
        except Exception:
            sf = None
        self.sf = sf
        self._info = {}

    @property
    def available(self) -> bool:
        return self.sf is not None

    def cut(self, path, start, end):
        sf = self.sf
        if sf is None or not path or not os.path.isfile(path):
            return None
        try:
            if path not in self._info:
                info = sf.info(path)
                self._info[path] = (info.samplerate, info.frames)
            sr, frames = self._info[path]
            a = max(0, int(round(start * sr)))
            b = min(frames, int(round(min(end, start + MAX_CLIP_SECONDS) * sr)))
            if b <= a:
                return None
            data, _ = sf.read(path, start=a, stop=b, dtype="float32", always_2d=False)
            buf = io.BytesIO()
            sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
            return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return None


def _wanted_spans(fi: int, f: dict, base: str, top_n: int):
    """Ordered {key: (start, end, [variant ids])}: worst WER first, then ranges."""
    spans = {}

    def add(start, end, vid):
        key = _span_key(fi, start, end)
        if key not in spans:
            spans[key] = (_num(start), _num(end), [])
        if vid not in spans[key][2]:
            spans[key][2].append(vid)

    td = f.get("text_diff") or {}
    for vid in sorted(k for k in td if k != base):
        segs = [s for s in ((td[vid] or {}).get("segments") or []) if isinstance(s, dict)]
        segs.sort(key=lambda s: (-_num(s.get("wer")), _num(s.get("start"))))
        for s in segs[:max(0, top_n)]:
            add(s.get("start"), s.get("end"), vid)
    sp = f.get("speakers") or {}
    for vid in sorted(k for k in sp if k != base):
        for r in ((sp[vid] or {}).get("disagree_ranges") or []):
            if isinstance(r, dict):
                add(r.get("start"), r.get("end"), vid)
    return spans


def _annotate_keys(fi: int, f: dict):
    """Give every span the page may play a `_k` key (matches _span_key)."""
    for td in (f.get("text_diff") or {}).values():
        for s in (td or {}).get("segments") or []:
            if isinstance(s, dict):
                s["_k"] = _span_key(fi, s.get("start"), s.get("end"))
    for sp in (f.get("speakers") or {}).values():
        for r in (sp or {}).get("disagree_ranges") or []:
            if isinstance(r, dict):
                r["_k"] = _span_key(fi, r.get("start"), r.get("end"))


def prepare(data: dict, files: list, mode: str, max_mb: float, top_n: int, out_path: str):
    """The data object embedded in the page, and the clips map (key -> {src: uri})."""
    base = str(data.get("baseline") or "none")
    variants = _variants(data)
    files = json.loads(json.dumps(files))  # own copy, we add keys
    clips, urls = {}, {}
    stats = {"mode": mode, "clips": 0, "left_out": 0, "bytes": 0, "note": ""}
    out_dir = os.path.dirname(os.path.abspath(out_path)) if out_path else os.getcwd()
    budget = int(max_mb * 1024 * 1024)
    cutter = _Cutter() if mode == "clips" else None
    if cutter is not None and not cutter.available:
        stats["note"] = "soundfile is not installed: no audio clips were embedded"

    for fi, f in enumerate(files):
        _annotate_keys(fi, f)
        audio = f.get("audio") if isinstance(f.get("audio"), dict) else {}
        if mode == "link":
            urls[str(fi)] = {}
            for src, p in sorted(audio.items()):
                if not isinstance(p, str) or not os.path.isfile(p):
                    continue
                try:
                    rel = os.path.relpath(p, out_dir).replace(os.sep, "/")
                    urls[str(fi)][src] = urllib.parse.quote(rel)
                except ValueError:
                    urls[str(fi)][src] = pathlib.Path(os.path.abspath(p)).as_uri()
        elif mode == "clips":
            raw = audio.get("raw")
            for key, (s, e, vids) in _wanted_spans(fi, f, base, top_n).items():
                sources = ["raw"] + [v for v in vids if audio.get(v) and audio.get(v) != raw]
                for src in sources:
                    stats["left_out"] += 1
                    if not cutter.available or budget <= 0:
                        continue
                    uri = cutter.cut(audio.get(src), s, e)
                    if uri is None or len(uri) > budget:
                        continue
                    budget -= len(uri)
                    stats["bytes"] += len(uri)
                    stats["clips"] += 1
                    stats["left_out"] -= 1
                    clips.setdefault(key, {})[src] = uri
        f["audio"] = {}  # paths stay out of the page; clips/urls carry the audio
    page = {
        "version": data.get("version", 1),
        "baseline": base,
        "variants": variants,
        "files": files,
        "summary": data.get("summary") or {},
        "summary_n": data.get("summary_n") or {},
        "summary_excluded_degraded": data.get("summary_excluded_degraded") or {},
        "embed": {**stats, "urls": urls},
    }
    return page, clips


def _cmp_options(files):
    out = []
    for i, f in enumerate(files):
        out.append(f'<option value="{i}">{html.escape(str(f.get("name", i)))}</option>')
    return "".join(out)


def _degraded_text(v) -> str:
    """'fell back to unfiltered audio in N/M files' (counts optional)."""
    n, m = v.get("degraded_files"), v.get("total_files")
    where = f" in {n}/{m} files" if isinstance(n, int) and isinstance(m, int) and m else ""
    return "fell back to unfiltered audio" + where


def _banner(variants) -> str:
    bad = [v for v in variants if v.get("degraded")]
    if not bad:
        return ""
    items = "".join(f"<li><strong>{html.escape(str(v.get('label') or v['id']))}</strong>: "
                    f"{html.escape(_degraded_text(v))}"
                    + (f" ({html.escape(str(v['degraded_reason']))})" if v.get("degraded_reason") else "")
                    + "</li>" for v in bad)
    return ('<div class="banner bad" role="alert"><strong>Degraded variants: the filter did not run, '
            'the audio is the unfiltered input.</strong> Their numbers say nothing about what the '
            f'filter does.<ul>{items}</ul></div>')


def _chips(variants, base):
    out = []
    for v in variants:
        vid = html.escape(v["id"], quote=True)
        label = html.escape(str(v.get("label") or v["id"]))
        flow = " | ".join(f"{name}: {' > '.join(map(str, v[key]))}" for name, key in (
            ("pre_diarization", "chain"), ("pre_separation", "chain_pre_separation"),
            ("pre_asr", "chain_pre_asr")) if v.get(key))
        tip = html.escape(str(v.get("note") or flow or "baseline"), quote=True)
        if v.get("set"):
            tip += html.escape(" | set: " + json.dumps(v["set"], ensure_ascii=False), quote=True)
        if v.get("degraded"):
            tip += html.escape(" | DEGRADED: " + str(v.get("degraded_reason") or _degraded_text(v)), quote=True)
        cls = "chip base" if v["id"] == base else "chip"
        if v.get("degraded"):
            cls += " degraded"
        out.append(f'<button type="button" class="{cls}" data-vid="{vid}" title="{tip}" '
                   f'aria-pressed="false">{label}'
                   + (f' <span class="grp">{html.escape(str(v["group"]))}</span>' if v.get("group") else "")
                   + (f' <span class="grp tag">{html.escape(str(v["tag"]))}</span>' if v.get("tag") else "")
                   + (f' <span class="badge bad">{html.escape(_degraded_text(v))}</span>' if v.get("degraded") else "")
                   + (' <span class="badge warn">output identical to input</span>'
                      if v.get("suspect_noop_files") else "")
                   + '</button>')
    return "".join(out)


def render_page(data: dict, files: list, mode="clips", max_mb=DEFAULT_MAX_MB,
                top_n=DEFAULT_TOP_SEGMENTS, out_path="compare.html") -> str:
    page, clips = prepare(data, files, mode, max_mb, top_n, out_path)
    base = page["baseline"]
    base_label = next((str(v.get("label") or v["id"]) for v in page["variants"] if v["id"] == base), base)
    title = "Noise-filter comparison"
    subs = {
        "TITLE": html.escape(title),
        "BASELINE": html.escape(base_label),
        "FILES": _cmp_options(page["files"]),
        "CHIPS": _chips(page["variants"], base),
        "DEGRADED": _banner(page["variants"]),
        "NFILES": str(len(page["files"])),
        "DATA": _js_json(page),
        "CLIPS": _js_json(clips),
        "SCRIPT": SCRIPT,
    }
    # one pass, so a placeholder-looking string inside the data is never expanded
    return re.sub(r"__(TITLE|BASELINE|FILES|CHIPS|DEGRADED|NFILES|DATA|CLIPS|SCRIPT)__",
                  lambda m: subs[m.group(1)], PAGE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", help="compare_data.json")
    ap.add_argument("-o", "--output", default="compare.html")
    ap.add_argument("--embed", choices=("clips", "link", "none"), default="clips")
    ap.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB,
                    help="cap on embedded audio, in MB of data: URIs")
    ap.add_argument("--top-segments", type=int, default=DEFAULT_TOP_SEGMENTS,
                    help="per variant, embed clips for this many worst-WER segments")
    ap.add_argument("--files", default="", help="comma-separated file names/stems to include")
    args = ap.parse_args(argv)

    data = load_data(args.data)
    stems = [s for s in args.files.split(",") if s.strip()]
    files = select_files(data, stems)
    if stems and not files:
        raise SystemExit(f"no file in {args.data} matches --files {args.files}")
    text = render_page(data, files, args.embed, args.max_mb, args.top_segments, args.output)
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {args.output} ({len(text) / 1e6:.1f} MB, {len(files)} file(s))", file=sys.stderr)


SCRIPT = r"""
(function (root) {
"use strict";
var DASH = "\u2014";
var D = null, CLIPS = {};
var S = {fi: 0, vid: null, tab: "text", onlyDiff: true, sort: "time", shown: 50,
         mmode: "file", zoom: null, sel: -1, rshown: 200};

/* ---------- pure helpers (also used by the tests) ---------- */
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
    return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c];
  });
}
function isNum(x) { return typeof x === "number" && isFinite(x); }
function fmt(v) {
  if (v === null || v === undefined) return DASH;
  if (typeof v === "boolean") return v ? "true" : "false";
  if (isNum(v)) {
    if (Math.floor(v) === v) return String(v);
    return Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 10 ? v.toFixed(2) : v.toFixed(3);
  }
  if (typeof v === "object") return DASH;
  return String(v);
}
function fmtDelta(d) {
  if (!isNum(d)) return "";
  var s = fmt(Math.round(d * 1000) / 1000);
  return (d > 0 ? "+" : "") + s;
}
function fmtTime(t) {
  if (!isNum(t)) return DASH;
  var m = Math.floor(t / 60), s = t - m * 60;
  return m + ":" + (s < 10 ? "0" : "") + s.toFixed(1);
}
function get(o, path) {
  for (var i = 0; i < path.length; i++) {
    if (o === null || o === undefined || typeof o !== "object") return null;
    o = o[path[i]];
  }
  return o === undefined ? null : o;
}
function mean(xs) {
  var a = xs.filter(isNum);
  if (!a.length) return null;
  return a.reduce(function (p, c) { return p + c; }, 0) / a.length;
}

/* direction hints: only where we are sure; everything else is neutral */
var DIRS = {
  "asr.wer_vs_baseline": "lower", "asr.wer_vs_reference": "lower",
  "asr.words_in_quiet_frames": "lower", "asr.empty_segments": "lower",
  "speakers.der_vs_baseline": "lower", "speakers.disagree_seconds": "lower",
  "audio.corr_with_raw": "higher", "audio.ecapa_cos": "higher",
  "audio.lag_samples": "zero", "audio.length_equal": "true",
  "separation.failed_spans": "lower"
};
var TIPS = {
  "audio.noise_floor_db": "Informative, not good or bad on its own: a lower floor can mean cleaner audio or over-suppression.",
  "audio.gated_frame_pct": "Informative, not good or bad on its own: share of frames the filter gated to near silence.",
  "audio.hf_energy_ratio": "Informative: high-frequency energy relative to raw; a big drop can mean muffled speech.",
  "audio.lag_samples": "Should be 0: the filter must stay sample-aligned with the raw audio.",
  "audio.length_equal": "Must be true: output length equals input length.",
  "audio.corr_with_raw": "Higher means closer to the raw waveform.",
  "audio.ecapa_cos": "Speaker-embedding similarity to raw; higher means the voice is preserved.",
  "diarization.n_speakers": "Informative: a different count is a difference, not an error.",
  "diarization.overlap_seconds": "Informative: more or less overlap is not good or bad by itself.",
  "diarization.n_segments": "Informative.",
  "diarization.median_segment_s": "Informative.",
  "diarization.short_segments_lt_0_5s": "Informative: many very short segments can indicate fragmentation.",
  "separation.overlap_spans": "Informative: spans the separation stage had to handle.",
  "separation.segments_separated": "Informative.",
  "separation.failed_spans": "Lower is better.",
  "separation.mean_similarity": "Informative (no fixed direction assumed).",
  "asr.n_words": "Informative: fewer words can be fewer hallucinations or lost speech.",
  "asr.n_segments": "Informative.",
  "asr.words_per_sec": "Informative.",
  "asr.wer_vs_baseline": "Distance from the baseline transcript. Not an error rate: the baseline is not ground truth.",
  "asr.wer_vs_reference": "Error rate against a human reference. The only column here that is ground truth.",
  "asr.words_in_quiet_frames": "Words emitted where the audio is quiet; lower suggests fewer hallucinations.",
  "speakers.der_vs_baseline": "Diarization error rate against the baseline diarization, not against ground truth.",
  "speakers.disagree_seconds": "Seconds where the variant's speaker differs from the baseline."
};
var MOS_KEYS = ["mos", "dnsmos_ovrl", "scoreq", "utmos", "nisqa_mos"];
var MOS_TIP = "Learned predictor (distilled DNSMOS/UTMOS/NISQA/SCOREQ plus a human-MOS head), 16 kHz so blind above 8 kHz. " +
  "It can reward hallucinated output. Delta only, never a gate: read it next to wer_vs_baseline and ecapa_cos.";
var ORDER = {
  audio: ["noise_floor_db", "gated_frame_pct", "lag_samples", "corr_with_raw", "hf_energy_ratio", "length_equal", "ecapa_cos"],
  diarization: ["n_speakers", "n_segments", "overlap_seconds", "median_segment_s", "short_segments_lt_0_5s"],
  separation: ["overlap_spans", "segments_separated", "failed_spans", "mean_similarity"],
  asr: ["n_words", "n_segments", "empty_segments", "words_per_sec", "wer_vs_baseline", "wer_vs_reference", "words_in_quiet_frames"],
  speakers: ["n_speakers", "disagree_seconds", "der_vs_baseline"]
};
var SECTIONS = [["audio", "Audio"], ["diarization", "Diarization"], ["separation", "Separation"],
                ["asr", "ASR"], ["speakers", "Speakers vs baseline"]];
var SUMMARY_KEYS = ["asr.wer_vs_baseline", "asr.wer_vs_reference", "speakers.der_vs_baseline",
                    "asr.words_in_quiet_frames", "audio.corr_with_raw", "audio.ecapa_cos",
                    "diarization.n_speakers", "separation.mean_similarity"];

/* judge a change: "good", "bad" or "" (neutral) */
function judge(key, v, b) {
  var dir = DIRS[key];
  if (!dir) return "";
  if (dir === "true") return v === true ? "good" : v === false ? "bad" : "";
  if (!isNum(v) || !isNum(b)) return "";
  if (dir === "lower") return v < b ? "good" : v > b ? "bad" : "";
  if (dir === "higher") return v > b ? "good" : v < b ? "bad" : "";
  if (dir === "zero") return Math.abs(v) < Math.abs(b) ? "good" : Math.abs(v) > Math.abs(b) ? "bad" : "";
  return "";
}
function dirLabel(key) {
  var d = DIRS[key];
  return d === "lower" ? "lower is better" : d === "higher" ? "higher is better" :
         d === "zero" ? "closer to 0 is better" : d === "true" ? "must be true" : "informative";
}

function diffHtml(ops, side, fallback) {
  if (!Array.isArray(ops) || !ops.length) return esc(fallback || "");
  var out = [];
  for (var i = 0; i < ops.length; i++) {
    var o = ops[i];
    if (!Array.isArray(o)) continue;
    var k = o[0];
    if (k === "equal") out.push(esc(o[1]));
    else if (k === "sub") out.push('<span class="w sub">' + esc(side === "base" ? o[1] : o[2]) + "</span>");
    else if (k === "del") { if (side === "base") out.push('<del class="w del">' + esc(o[1]) + "</del>"); }
    else if (k === "ins") { if (side !== "base") out.push('<ins class="w ins">' + esc(o[1]) + "</ins>"); }
  }
  return out.join(" ");
}

/* ---------- speaker colours ---------- */
function colorIndex(f, vid) {
  /* baseline labels first (sorted), then variant labels that map to nothing */
  var base = D.baseline, idx = {}, n = 0, i;
  var bl = {};
  ((get(f, ["text", base, "segments"])) || []).forEach(function (s) { bl[String(s.speaker)] = 1; });
  Object.keys(bl).sort().forEach(function (l) { idx["b:" + l] = n++; });
  var map = get(f, ["speakers", vid, "label_map"]) || {};
  var vl = {};
  ((get(f, ["text", vid, "segments"])) || []).forEach(function (s) { vl[String(s.speaker)] = 1; });
  Object.keys(vl).sort().forEach(function (l) {
    if (map[l] !== undefined && map[l] !== null && idx["b:" + map[l]] === undefined) idx["b:" + map[l]] = n++;
    if ((map[l] === undefined || map[l] === null) && idx["v:" + l] === undefined) idx["v:" + l] = n++;
  });
  return {
    base: function (l) { var x = idx["b:" + l]; return x === undefined ? -1 : x; },
    variant: function (l) {
      var m = map[l];
      var x = (m !== undefined && m !== null) ? idx["b:" + m] : idx["v:" + l];
      return x === undefined ? -1 : x;
    },
    n: n, labels: idx
  };
}
function spkClass(i) { return i < 0 ? "sx" : "s" + (i % 8); }

/* ---------- speaker timeline (SVG string) ---------- */
function timelineSvg(o) {
  /* o: {t0,t1,lanes:[{label,segs,cls(label)}],bands:[{start,end,kind}],sel} */
  var W = 1000, laneH = 26, titleH = 14, gap = 8, top = 18;
  var span = Math.max(1e-6, o.t1 - o.t0);
  function X(t) { return Math.max(0, Math.min(W, (t - o.t0) / span * W)); }
  var H = top + o.lanes.length * (titleH + laneH + gap);
  var p = ['<svg class="tl" viewBox="0 0 ' + W + " " + H + '" preserveAspectRatio="none" role="img" aria-label="speaker timelines">'];
  var step = niceStep(span / 8), t;
  for (t = Math.ceil(o.t0 / step) * step; t <= o.t1 + 1e-9; t += step) {
    p.push('<line class="grid" x1="' + X(t).toFixed(1) + '" x2="' + X(t).toFixed(1) + '" y1="' + top + '" y2="' + H + '"/>');
  }
  var bandsTop = top, bandsH = H - top;
  (o.bands || []).forEach(function (b, i) {
    if (b.end < o.t0 || b.start > o.t1) return;
    var x = X(b.start), w = Math.max(1.5, X(b.end) - x);
    p.push('<rect class="band band-' + esc(b.kind || "other") + (i === o.sel ? " sel" : "") + '" data-i="' + i +
           '" x="' + x.toFixed(1) + '" y="' + bandsTop + '" width="' + w.toFixed(1) + '" height="' + bandsH +
           '"><title>' + esc((b.kind || "?") + " " + fmtTime(b.start) + " - " + fmtTime(b.end)) + "</title></rect>");
  });
  o.lanes.forEach(function (lane, li) {
    var y = top + li * (titleH + laneH + gap);
    p.push('<text class="lane-title" x="2" y="' + (y + 10) + '">' + esc(lane.label) + "</text>");
    p.push('<rect class="lane" x="0" y="' + (y + titleH) + '" width="' + W + '" height="' + laneH + '"/>');
    (lane.segs || []).forEach(function (s) {
      var a = _n(s.start), b = _n(s.end);
      if (b < o.t0 || a > o.t1) return;
      var x = X(a), w = Math.max(0.6, X(b) - x);
      p.push('<rect class="seg ' + spkClass(lane.cls(String(s.speaker))) + '" x="' + x.toFixed(1) + '" y="' + (y + titleH) +
             '" width="' + w.toFixed(1) + '" height="' + laneH + '"><title>' + esc(s.speaker + " " + fmtTime(a) + " - " + fmtTime(b)) +
             "</title></rect>");
    });
  });
  for (t = Math.ceil(o.t0 / step) * step; t <= o.t1 + 1e-9; t += step) {
    p.push('<text class="tick" x="' + (X(t) + 2).toFixed(1) + '" y="11">' + esc(fmtTime(t)) + "</text>");
  }
  p.push("</svg>");
  return p.join("");
}
function niceStep(raw) {
  var steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600];
  for (var i = 0; i < steps.length; i++) if (steps[i] >= raw) return steps[i];
  return 3600;
}
function _n(x) { return isNum(x) ? x : 0; }

function rangesHtml(ranges, sel, limit) {
  if (!ranges.length) return '<p class="empty">No disagreements with the baseline.</p>';
  var rows = [];
  ranges.slice(0, limit).forEach(function (r, i) {
    var len = _n(r.end) - _n(r.start);
    rows.push('<tr class="rrow' + (i === sel ? " sel" : "") + '" data-i="' + i + '" tabindex="0">' +
      "<td>" + esc(fmtTime(r.start)) + "</td><td>" + esc(r.baseline == null ? DASH : r.baseline) + "</td><td>" +
      esc(r.variant == null ? DASH : r.variant) + '</td><td><span class="kind kind-' + esc(r.kind || "other") + '">' +
      esc(r.kind || "?") + "</span></td><td class=\"num\">" + esc(len.toFixed(2)) + " s</td>" +
      '<td><button type="button" class="mini" data-play-range="' + i + '">play both</button></td></tr>');
  });
  var more = ranges.length > limit ? '<p class="empty"><button type="button" id="rmore">show more (' + (ranges.length - limit) + " left)</button></p>" : "";
  return '<table class="rt"><thead><tr><th>time</th><th>baseline</th><th>variant</th><th>kind</th><th>length</th><th></th></tr></thead><tbody>' +
    rows.join("") + "</tbody></table>" + more;
}

/* ---------- metrics ---------- */
function metricKeys(sec) {
  var seen = {}, extra = [];
  (D.files || []).forEach(function (f) {
    D.variants.forEach(function (v) {
      var o = sec === "speakers" ? get(f, ["speakers", v.id]) : get(f, ["metrics", v.id, sec]);
      if (o && typeof o === "object") Object.keys(o).forEach(function (k) {
        if (k === "label_map" || k === "disagree_ranges" || k === "speaker_seconds" || k === "learned_mos" || k === "learned_mos_delta") return;
        if (ORDER[sec].indexOf(k) < 0 && !seen[k]) { seen[k] = 1; extra.push(k); }
      });
    });
  });
  var present = ORDER[sec].filter(function (k) { return hasMetric(sec, k); });
  return present.concat(extra.sort());
}
function hasMetric(sec, k) {
  return (D.files || []).some(function (f) {
    return D.variants.some(function (v) {
      var o = sec === "speakers" ? get(f, ["speakers", v.id]) : get(f, ["metrics", v.id, sec]);
      return o && typeof o === "object" && k in o;
    });
  });
}
function metricValue(sec, k, vid, mode, f) {
  if (mode === "file") return sec === "speakers" ? get(f, ["speakers", vid, k]) : get(f, ["metrics", vid, sec, k]);
  var s = get(D.summary, [vid, sec + "." + k]);
  if (isNum(s) || typeof s === "boolean") return s;
  return mean((D.files || []).map(function (ff) {
    return sec === "speakers" ? get(ff, ["speakers", vid, k]) : get(ff, ["metrics", vid, sec, k]);
  }));
}
/* informational block: learned MOS delta vs raw (absent in old data files) */
function mosHtml(mode, f, vars, bi) {
  if (!hasMetric("audio", "learned_mos_delta") && !hasMetric("audio", "learned_mos")) return "";
  function val(vid, k) {
    if (mode === "file") return get(f, ["metrics", vid, "audio", "learned_mos_delta", k]);
    var s = get(D.summary, [vid, "audio.learned_mos_delta." + k]);
    if (isNum(s)) return s;
    return mean((D.files || []).map(function (ff) { return get(ff, ["metrics", vid, "audio", "learned_mos_delta", k]); }));
  }
  var head = vars.map(function (v) { return "<th" + (v.id === D.baseline ? ' class="basecol"' : "") + ">" + esc(v.label || v.id) + "</th>"; }).join("");
  var rows = MOS_KEYS.map(function (k) {
    var cells = vars.map(function (v, i) {
      var x = val(v.id, k);
      return '<td class="num' + (i === bi ? " basecol" : "") + '">' + esc(isNum(x) ? fmtDelta(x) || "0" : DASH) + "</td>";
    }).join("");
    return '<tr><th scope="row" title="' + esc(MOS_TIP) + '">' + esc(k) + ' <small class="dir">informative</small></th>' + cells + "</tr>";
  }).join("");
  return '<h3 title="' + esc(MOS_TIP) + '">Learned MOS (delta vs raw, informational)</h3><p class="empty">' + esc(MOS_TIP) +
    '</p><div class="scroll"><table class="mt mos"><thead><tr><th>metric</th>' + head + "</tr></thead><tbody>" + rows + "</tbody></table></div>";
}
function miniBars(vals, baseIdx) {
  var nums = vals.filter(isNum), mx = Math.max.apply(null, nums.map(Math.abs).concat([1e-12]));
  var w = 8, g = 3, h = 22, p = ['<svg class="mb" width="' + (vals.length * (w + g)) + '" height="' + h + '" aria-hidden="true">'];
  vals.forEach(function (v, i) {
    if (!isNum(v)) { p.push('<rect class="nb" x="' + (i * (w + g)) + '" y="' + (h - 2) + '" width="' + w + '" height="2"/>'); return; }
    var bh = Math.max(1, Math.abs(v) / mx * (h - 2));
    p.push('<rect class="' + (i === baseIdx ? "bb" : "vb") + '" x="' + (i * (w + g)) + '" y="' + (h - bh) + '" width="' + w + '" height="' + bh + '"/>');
  });
  p.push("</svg>");
  return p.join("");
}
function metricsHtml(mode, f) {
  var vars = D.variants, bi = vars.map(function (v) { return v.id; }).indexOf(D.baseline);
  var out = [];
  SECTIONS.forEach(function (sd) {
    var sec = sd[0], keys = metricKeys(sec);
    if (!keys.length) return;
    var head = vars.map(function (v) { return "<th" + (v.id === D.baseline ? ' class="basecol"' : "") + ">" + esc(v.label || v.id) + "</th>"; }).join("");
    var rows = keys.map(function (k) {
      var key = sec + "." + k, vals = vars.map(function (v) { return metricValue(sec, k, v.id, mode, f); });
      var b = bi >= 0 ? vals[bi] : null;
      var cells = vals.map(function (v, i) {
        var isBase = i === bi, d = (!isBase && isNum(v) && isNum(b)) ? v - b : null;
        var cls = isBase ? "basecol" : "";
        var j = isBase ? (DIRS[key] === "true" ? judge(key, v, b) : "") : judge(key, v, b);
        var delta = d === null ? "" : ' <span class="delta ' + j + '">' + esc(fmtDelta(d)) + "</span>";
        return '<td class="num ' + cls + (j && DIRS[key] === "true" ? " " + j : "") + '">' + esc(fmt(v)) + delta + "</td>";
      }).join("");
      var tip = (TIPS[key] || "") + " (" + dirLabel(key) + ")";
      return "<tr><th scope=\"row\" title=\"" + esc(tip) + "\">" + esc(k) + ' <small class="dir">' + esc(dirLabel(key)) + "</small></th>" + cells +
        "<td>" + miniBars(vals, bi) + "</td></tr>";
    }).join("");
    out.push("<h3>" + esc(sd[1]) + '</h3><div class="scroll"><table class="mt"><thead><tr><th>metric</th>' + head + "<th>chart</th></tr></thead><tbody>" + rows + "</tbody></table></div>");
    if (sec === "audio") out.push(mosHtml(mode, f, vars, bi));
  });
  return out.length ? out.join("") : '<p class="empty">No metrics in this data.</p>';
}

function summaryHtml() {
  var vars = D.variants, bi = vars.map(function (v) { return v.id; }).indexOf(D.baseline);
  var keys = SUMMARY_KEYS.filter(function (k) {
    var p = k.split("."); return hasMetric(p[0], p[1]);
  });
  if (!keys.length) return '<p class="empty">No metrics to summarise.</p>';
  var head = keys.map(function (k) { return '<th title="' + esc((TIPS[k] || "") + " (" + dirLabel(k) + ")") + '">' + esc(k) + "</th>"; }).join("");
  var rows = vars.map(function (v, vi) {
    var cells = keys.map(function (k) {
      var p = k.split("."), x = metricValue(p[0], p[1], v.id, "summary", null);
      var b = bi >= 0 ? metricValue(p[0], p[1], D.baseline, "summary", null) : null;
      var j = vi === bi ? "" : judge(k, x, b);
      var d = (vi !== bi && isNum(x) && isNum(b)) ? ' <span class="delta ' + j + '">' + esc(fmtDelta(x - b)) + "</span>" : "";
      return '<td class="num">' + esc(fmt(x)) + d + "</td>";
    }).join("");
    return '<tr class="vrow' + (v.id === S.vid ? " sel" : "") + '" data-vid="' + esc(v.id) + '" tabindex="0"><th scope="row">' + esc(v.label || v.id) +
      (v.id === D.baseline ? " <small>(baseline)</small>" : "") + "</th>" + cells + "</tr>";
  }).join("");
  return '<div class="scroll"><table class="mt sum"><thead><tr><th>variant (mean over files)</th>' + head + "</tr></thead><tbody>" + rows + "</tbody></table></div>";
}

/* ---------- audio ---------- */
var player = null, queue = [];
function srcFor(fi, key, vid, s, e) {
  var c = CLIPS[key];
  if (c) {
    var src = vid === "raw" || vid === D.baseline ? (c.raw || c[vid]) : (c[vid] || null);
    if (src) return {src: src, clip: true};
  }
  var u = get(D.embed, ["urls", String(fi)]);
  if (u) {
    var url = (vid === "raw" || vid === D.baseline) ? (u.raw || u[vid]) : u[vid];
    if (url) return {src: url, clip: false, s: s, e: e};
  }
  return null;
}
function playList(items) {
  stopAudio();
  queue = items.slice();
  next();
}
function stopAudio() { if (player) { try { player.pause(); } catch (e) {} } player = null; queue = []; }
function next() {
  var it = queue.shift();
  if (!it) { player = null; return; }
  var a = new Audio();
  player = a;
  var done = function () { if (player === a) next(); };
  a.addEventListener("ended", done);
  a.addEventListener("error", done);
  if (it.clip) { a.src = it.src; } else {
    a.src = it.src;
    a.addEventListener("loadedmetadata", function () { try { a.currentTime = it.s; } catch (e) {} });
    a.addEventListener("timeupdate", function () { if (a.currentTime >= it.e) { a.pause(); done(); } });
  }
  var pr = a.play();
  if (pr && pr.catch) pr.catch(function () { done(); });
}
function playBtn(fi, key, vid, s, e, label, attr) {
  var it = srcFor(fi, key, vid, s, e);
  if (!it) return "";
  return '<button type="button" class="mini" ' + attr + ' data-vid-play="' + esc(vid) + '">' + esc(label) + "</button>";
}

/* ---------- views ---------- */
function $(id) { return document.getElementById(id); }
function curFile() { return (D.files || [])[S.fi] || null; }
function vlabel(id) {
  var v = D.variants.filter(function (x) { return x.id === id; })[0];
  return v ? (v.label || v.id) : id;
}
function speakerAt(f, t0, t1) {
  var segs = get(f, ["text", D.baseline, "segments"]) || [], best = null, bo = 0;
  for (var i = 0; i < segs.length; i++) {
    var o = Math.min(t1, _n(segs[i].end)) - Math.max(t0, _n(segs[i].start));
    if (o > bo) { bo = o; best = segs[i].speaker; }
  }
  return best;
}

/* warning line for the selected variant/file when its denoise verdict is not ok (absent in old data) */
function denoiseWarn(f, vid) {
  var d = get(f, ["denoise", vid]);
  if (!d || !d.verdict || d.verdict === "ok") return "";
  var err = (d.errors && d.errors.length) ? String(d.errors[0]).split("\n")[0] : "";
  if (d.verdict === "suspect_noop")
    return '<p class="vwarn warn" role="note">Suspect: this filter\'s output is identical to its input for this file (corr_with_input ' +
      esc(fmt(d.corr_with_input)) + "). It may not have run.</p>";
  return '<p class="vwarn" role="alert"><strong>' + (d.verdict === "missing" ? "No denoise.json for this file" : "Filter fell back to the unfiltered audio") +
    "</strong>: these numbers do not show what the filter does." + (err ? " First error: " + esc(err) : "") + "</p>";
}

function renderText() {
  var f = curFile(), box = $("tab-text");
  if (!f) { box.innerHTML = '<p class="empty">No files in this data.</p>'; return; }
  if (S.vid === D.baseline) {
    box.innerHTML = '<p class="empty">The baseline is the reference side. Pick a variant chip above to compare it with ' + esc(vlabel(D.baseline)) + ".</p>";
    return;
  }
  var td = get(f, ["text_diff", S.vid]);
  if (!td) { box.innerHTML = '<p class="empty">No text comparison for ' + esc(vlabel(S.vid)) + " in this file.</p>"; return; }
  var ops = td.ops || {};
  var tot = denoiseWarn(f, S.vid) + '<div class="stats">' +
    stat("WER vs baseline", td.wer_vs_baseline, "distance from the baseline transcript, not an error rate") +
    stat("WER vs reference", td.wer_vs_reference, "against a human reference; " + DASH + " when none exists") +
    stat("sub", ops.sub) + stat("ins", ops.ins) + stat("del", ops.del) + stat("equal", ops.equal) + "</div>";
  var segs = (td.segments || []).filter(function (s) { return s && typeof s === "object"; });
  var total = segs.length;
  if (S.onlyDiff) segs = segs.filter(function (s) { return _n(s.wer) > 0 || (s.baseline_text || "") !== (s.variant_text || ""); });
  segs = segs.slice();
  if (S.sort === "wer") segs.sort(function (a, b) { return _n(b.wer) - _n(a.wer) || _n(a.start) - _n(b.start); });
  else segs.sort(function (a, b) { return _n(a.start) - _n(b.start); });
  var shown = segs.slice(0, S.shown);
  var cards = shown.map(function (s) {
    var key = s._k, st = _n(s.start), en = _n(s.end);
    var spk = speakerAt(f, st, en);
    var pb = playBtn(S.fi, key, "raw", st, en, "play raw", 'data-play="raw"') +
             playBtn(S.fi, key, S.vid, st, en, "play " + vlabel(S.vid), 'data-play="var"') +
             (srcFor(S.fi, key, S.vid, st, en) && srcFor(S.fi, key, "raw", st, en) ? '<button type="button" class="mini" data-play="both">raw, then variant</button>' : "");
    return '<article class="seg-card" data-k="' + esc(key) + '" data-s="' + st + '" data-e="' + en + '">' +
      '<header><span class="time">' + esc(fmtTime(st)) + " - " + esc(fmtTime(en)) + "</span>" +
      (spk != null ? '<span class="spk">' + esc(spk) + "</span>" : "") +
      '<span class="wer" title="segment WER vs baseline">WER ' + esc(fmt(s.wer)) + "</span>" +
      '<span class="spacer"></span>' + pb + "</header>" +
      '<div class="cols"><div><h4>' + esc(vlabel(D.baseline)) + "</h4><p lang=\"vi\">" + diffHtml(s.ops, "base", s.baseline_text) + "</p></div>" +
      "<div><h4>" + esc(vlabel(S.vid)) + '</h4><p lang="vi">' + diffHtml(s.ops, "var", s.variant_text) + "</p></div></div></article>";
  }).join("");
  var more = segs.length > shown.length ? '<p class="empty"><button type="button" id="tmore">show more (' + (segs.length - shown.length) + " left)</button></p>" : "";
  var empty = !segs.length ? '<p class="empty">' + (total ? "No segments differ." : "No segment-level data for this variant.") + "</p>" : "";
  var ctl = '<div class="ctl"><label><input type="checkbox" id="onlydiff"' + (S.onlyDiff ? " checked" : "") + '> only segments that differ</label>' +
    '<label>sort <select id="sortsel"><option value="time"' + (S.sort === "time" ? " selected" : "") + '>by time</option><option value="wer"' +
    (S.sort === "wer" ? " selected" : "") + '>by segment WER</option></select></label><span class="meta">' + segs.length + " of " + total + " segments</span></div>";
  box.innerHTML = tot + ctl + empty + '<div class="cards">' + cards + "</div>" + more;
}
function stat(label, v, tip) {
  return '<div class="stat" title="' + esc(tip || "") + '"><b>' + esc(fmt(v)) + "</b><span>" + esc(label) + "</span></div>";
}

function ranges() {
  var f = curFile();
  return (get(f, ["speakers", S.vid, "disagree_ranges"]) || []).filter(function (r) { return r && typeof r === "object"; });
}
function renderSpeakers() {
  var f = curFile(), box = $("tab-speakers");
  if (!f) { box.innerHTML = '<p class="empty">No files in this data.</p>'; return; }
  if (S.vid === D.baseline) {
    box.innerHTML = '<p class="empty">The baseline is the reference side. Pick a variant chip above.</p>';
    return;
  }
  var sp = get(f, ["speakers", S.vid]);
  var bsegs = get(f, ["text", D.baseline, "segments"]) || [], vsegs = get(f, ["text", S.vid, "segments"]) || [];
  if (!sp && !vsegs.length) { box.innerHTML = '<p class="empty">No speaker comparison for ' + esc(vlabel(S.vid)) + " in this file.</p>"; return; }
  var warn = denoiseWarn(f, S.vid);
  var dur = _n(f.duration);
  [bsegs, vsegs].forEach(function (a) { a.forEach(function (s) { dur = Math.max(dur, _n(s.end)); }); });
  if (!(dur > 0)) dur = 1;
  if (!S.zoom) S.zoom = [0, dur];
  var ci = colorIndex(f, S.vid), rs = ranges();
  var svg = timelineSvg({
    t0: S.zoom[0], t1: S.zoom[1], bands: rs, sel: S.sel,
    lanes: [{label: vlabel(D.baseline) + " (baseline)", segs: bsegs, cls: ci.base},
            {label: vlabel(S.vid), segs: vsegs, cls: ci.variant}]
  });
  var legend = Object.keys(ci.labels).sort(function (a, b) { return ci.labels[a] - ci.labels[b]; }).map(function (k) {
    return '<span class="lg"><i class="sw ' + spkClass(ci.labels[k]) + '"></i>' + esc(k.slice(2)) + (k.charAt(0) === "v" ? " (variant only)" : "") + "</span>";
  }).join("");
  var kinds = ["swap", "missing", "extra", "overlap"].map(function (k) { return '<span class="lg"><i class="sw band-' + k + '"></i>' + k + "</span>"; }).join("");
  var secs = sp && sp.speaker_seconds && typeof sp.speaker_seconds === "object" ? sp.speaker_seconds : null;
  var secRows = secs ? Object.keys(secs).sort().map(function (l) {
    return "<tr><td>" + esc(l) + '</td><td class="num">' + esc(fmt(secs[l])) + "</td></tr>";
  }).join("") : "";
  var map = sp && sp.label_map && typeof sp.label_map === "object" ? sp.label_map : {};
  var mapTxt = Object.keys(map).sort().map(function (k) { return esc(k) + " " + "\u2192" + " " + esc(map[k]); }).join(", ");
  var stats = '<div class="stats">' + stat("n_speakers", sp && sp.n_speakers, "variant speaker count") +
    stat("baseline n_speakers", get(f, ["speakers", D.baseline, "n_speakers"]) != null ? get(f, ["speakers", D.baseline, "n_speakers"]) : (ci.n ? Object.keys(ci.labels).filter(function (k) { return k.charAt(0) === "b"; }).length : null)) +
    stat("disagree_seconds", sp && sp.disagree_seconds, "seconds where the speaker differs") +
    stat("der_vs_baseline", sp && sp.der_vs_baseline, "DER against the baseline diarization, not ground truth") +
    stat("ranges", rs.length) + "</div>";
  box.innerHTML = warn + stats +
    '<div class="ctl"><button type="button" id="zin">zoom in</button><button type="button" id="zout">zoom out</button><button type="button" id="zreset">reset zoom</button>' +
    '<span class="meta">' + esc(fmtTime(S.zoom[0])) + " - " + esc(fmtTime(S.zoom[1])) + "</span></div>" +
    '<div class="tlwrap" id="tlwrap">' + svg + '</div><div class="legend">' + legend + " " + kinds + "</div>" +
    '<div class="two"><div><h3>Disagreements</h3>' + rangesHtml(rs, S.sel, S.rshown) + "</div>" +
    "<div><h3>Speaker seconds</h3>" + (secRows ? '<table class="rt"><thead><tr><th>speaker (variant label)</th><th>seconds</th></tr></thead><tbody>' + secRows + "</tbody></table>" : '<p class="empty">' + DASH + "</p>") +
    (mapTxt ? '<p class="meta">label_map: ' + mapTxt + "</p>" : "") + "</div></div>";
}

/* how many files each variant's summary mean covers, and how many fell back (absent in old data) */
function summaryNote() {
  var n = D.summary_n || {}, ex = D.summary_excluded_degraded || {};
  if (!Object.keys(n).length && !Object.keys(ex).length) return "";
  var total = (D.files || []).length;
  var parts = D.variants.filter(function (v) { return isNum(n[v.id]) && (n[v.id] < total || ex[v.id] > 0); }).map(function (v) {
    return esc(v.label || v.id) + ": mean over " + n[v.id] + "/" + total + " files" +
      (ex[v.id] > 0 ? ", <strong>" + ex[v.id] + " fell back to unfiltered audio (still counted in the mean)</strong>" : "");
  });
  return '<p class="empty" id="sumnote">' + (parts.length ? "Summary means: " + parts.join("; ") + "." :
    "Summary means cover all " + total + " file(s) for every variant; none fell back.") + "</p>";
}
function renderMetrics() {
  var f = curFile(), box = $("tab-metrics");
  var ctl = '<div class="ctl"><label><input type="radio" name="mm" value="summary"' + (S.mmode === "summary" ? " checked" : "") +
    '> summary (mean over files)</label><label><input type="radio" name="mm" value="file"' + (S.mmode === "file" ? " checked" : "") +
    "> this file</label></div>";
  box.innerHTML = ctl + summaryNote() + (S.mmode === "file" ? denoiseWarn(f, S.vid) : "") + metricsHtml(S.mmode, f);
}
function renderHeader() {
  $("summary").innerHTML = summaryHtml();
  var chips = document.querySelectorAll(".chip");
  for (var i = 0; i < chips.length; i++) chips[i].setAttribute("aria-pressed", chips[i].getAttribute("data-vid") === S.vid ? "true" : "false");
  var hasRef = D.variants.some(function (v) { return (D.files || []).some(function (f) { return isNum(get(f, ["metrics", v.id, "asr", "wer_vs_reference"])) || isNum(get(f, ["text_diff", v.id, "wer_vs_reference"])); }); });
  $("refnote").textContent = hasRef ? " wer_vs_reference is shown where a human reference exists; it is the only column that measures correctness." :
    " No human reference in this data, so nothing here measures correctness.";
  var e = D.embed || {};
  var m = "";
  if (e.mode === "clips") m = e.clips + " clip(s) embedded" + (e.left_out ? ", " + e.left_out + " left out (size cap, missing audio or --top-segments)" : "") + (e.note ? ". " + e.note : "");
  else if (e.mode === "link") m = "audio linked by path: keep the wav files where they are";
  else m = "audio players are off (--embed none)";
  $("embednote").textContent = m;
}
function renderTab() {
  var tabs = ["text", "speakers", "metrics"];
  tabs.forEach(function (t) {
    $("tab-" + t).hidden = t !== S.tab;
    var b = $("btn-" + t); b.setAttribute("aria-selected", t === S.tab ? "true" : "false"); b.tabIndex = t === S.tab ? 0 : -1;
  });
  if (S.tab === "text") renderText(); else if (S.tab === "speakers") renderSpeakers(); else renderMetrics();
}
function renderAll() { renderHeader(); renderTab(); }

function selectRange(i, play) {
  var rs = ranges(), r = rs[i];
  if (!r) return;
  S.sel = i;
  var pad = Math.max(2, (_n(r.end) - _n(r.start)) * 1.5);
  S.zoom = [Math.max(0, _n(r.start) - pad), _n(r.end) + pad];
  renderSpeakers();
  var row = document.querySelector('.rrow[data-i="' + i + '"]');
  if (row && row.scrollIntoView) row.scrollIntoView({block: "nearest"});
  if (play) playRange(i);
}
function playRange(i) {
  var r = ranges()[i]; if (!r) return;
  var a = srcFor(S.fi, r._k, "raw", _n(r.start), _n(r.end)), b = srcFor(S.fi, r._k, S.vid, _n(r.start), _n(r.end));
  playList([a, b].filter(Boolean));
}
function zoomBy(f) {
  var file = curFile(), z = S.zoom || [0, 1], c = (z[0] + z[1]) / 2, h = (z[1] - z[0]) / 2 * f;
  var dur = _n(file && file.duration) || z[1];
  S.zoom = [Math.max(0, c - h), Math.min(Math.max(dur, z[1]), c + h)];
  renderSpeakers();
}

function setVariant(v) { S.vid = v; S.shown = 50; S.sel = -1; S.zoom = null; S.rshown = 200; renderAll(); }
function setFile(i) { S.fi = i; S.shown = 50; S.sel = -1; S.zoom = null; renderAll(); }

function init() {
  D = JSON.parse($("data").textContent);
  CLIPS = JSON.parse($("clips").textContent);
  var nonBase = D.variants.filter(function (v) { return v.id !== D.baseline; });
  S.vid = nonBase.length ? nonBase[0].id : D.baseline;
  var fsel = $("filesel");
  fsel.addEventListener("change", function () { setFile(parseInt(fsel.value, 10) || 0); });
  document.addEventListener("click", function (ev) {
    var t = ev.target, el;
    if ((el = t.closest(".chip"))) return setVariant(el.getAttribute("data-vid"));
    if ((el = t.closest(".vrow"))) return setVariant(el.getAttribute("data-vid"));
    if ((el = t.closest("[data-tab]"))) { S.tab = el.getAttribute("data-tab"); return renderTab(); }
    if (t.id === "tmore") { S.shown += 100; return renderText(); }
    if (t.id === "rmore") { S.rshown += 400; return renderSpeakers(); }
    if (t.id === "zin") return zoomBy(0.5);
    if (t.id === "zout") return zoomBy(2);
    if (t.id === "zreset") { S.zoom = null; S.sel = -1; return renderSpeakers(); }
    if ((el = t.closest("[data-play]"))) {
      var card = t.closest(".seg-card"), k = card.getAttribute("data-k"), s = parseFloat(card.getAttribute("data-s")), e = parseFloat(card.getAttribute("data-e"));
      var w = el.getAttribute("data-play"), mk = function (v) { return srcFor(S.fi, k, v, s, e); };
      return playList(w === "raw" ? [mk("raw")] : w === "var" ? [mk(S.vid)] : [mk("raw"), mk(S.vid)]);
    }
    if ((el = t.closest("[data-play-range]"))) return selectRange(parseInt(el.getAttribute("data-play-range"), 10), true);
    if ((el = t.closest(".rrow"))) return selectRange(parseInt(el.getAttribute("data-i"), 10), false);
    if ((el = t.closest(".band"))) return selectRange(parseInt(el.getAttribute("data-i"), 10), false);
  });
  document.addEventListener("change", function (ev) {
    var t = ev.target;
    if (t.id === "onlydiff") { S.onlyDiff = t.checked; S.shown = 50; renderText(); }
    else if (t.id === "sortsel") { S.sort = t.value; renderText(); }
    else if (t.name === "mm") { S.mmode = t.value; renderMetrics(); }
  });
  document.addEventListener("keydown", function (ev) {
    var t = ev.target, tag = t && t.tagName;
    if ((ev.key === "Enter" || ev.key === " ") && t.classList && (t.classList.contains("rrow") || t.classList.contains("vrow"))) {
      ev.preventDefault(); return t.click();
    }
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
    var tabs = ["text", "speakers", "metrics"], ti = tabs.indexOf(S.tab);
    if (t.getAttribute && t.getAttribute("role") === "tab" && (ev.key === "ArrowRight" || ev.key === "ArrowLeft")) {
      S.tab = tabs[(ti + (ev.key === "ArrowRight" ? 1 : 2)) % 3]; renderTab(); $("btn-" + S.tab).focus(); return;
    }
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (ev.key >= "1" && ev.key <= "3") { S.tab = tabs[+ev.key - 1]; renderTab(); }
    else if (ev.key === "[" || ev.key === "]") {
      var ids = D.variants.map(function (v) { return v.id; }), i = ids.indexOf(S.vid);
      setVariant(ids[(i + (ev.key === "]" ? 1 : ids.length - 1)) % ids.length]);
    } else if (ev.key === "Escape") stopAudio();
  });
  renderAll();
}

root.__cmp = {esc: esc, fmt: fmt, fmtDelta: fmtDelta, diffHtml: diffHtml, timelineSvg: timelineSvg, rangesHtml: rangesHtml,
              judge: judge, DIRS: DIRS, setData: function (d, c) { D = d; CLIPS = c || {}; }, metricsHtml: metricsHtml, denoiseWarn: denoiseWarn, summaryNote: summaryNote,
              miniBars: miniBars, summaryHtml: summaryHtml, setState: function (s) { for (var k in s) S[k] = s[k]; }};
if (root.document) {
  if (root.document.readyState === "loading") root.document.addEventListener("DOMContentLoaded", init); else init();
}
})(typeof window !== "undefined" ? window : globalThis);
"""

PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href="data:,">
<style>
:root{
  color-scheme:light dark;
  --bg:#f6f5f1;--panel:#ffffff;--panel2:#efede6;--fg:#1d2327;--fg2:#4d575d;--dim:#8a9399;
  --line:#dcd8cc;--accent:#0b6e75;--accent-bg:rgba(11,110,117,.10);
  --good:#1a7f45;--good-bg:rgba(26,127,69,.12);--bad:#b3261e;--bad-bg:rgba(179,38,30,.12);
  --sub:#fde68a;--subfg:#5c4300;--ins:#bbf7d0;--insfg:#14532d;--del:#fecaca;--delfg:#7f1d1d;
  --s0:#2b7bba;--s1:#d97706;--s2:#2e9e6a;--s3:#b8509e;--s4:#7c6bd6;--s5:#c2410c;--s6:#0f9aa8;--s7:#8a7a1e;--sx:#7b848a;
  --k-swap:#d6361f;--k-missing:#d99a00;--k-extra:#7b3fd0;--k-overlap:#0a8f9c;
  --font:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans","Helvetica Neue",Arial,"DejaVu Sans",sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#14181b;--panel:#1c2226;--panel2:#242c31;--fg:#e6e9ea;--fg2:#aab3b8;--dim:#7d878d;
  --line:#33403f;--accent:#4cc3cb;--accent-bg:rgba(76,195,203,.14);
  --good:#59d08a;--good-bg:rgba(89,208,138,.14);--bad:#ff8a80;--bad-bg:rgba(255,138,128,.14);
  --sub:#6b5400;--subfg:#fde68a;--ins:#14532d;--insfg:#bbf7d0;--del:#6b1f1f;--delfg:#fecaca;
  --s0:#5aa9e6;--s1:#f0a030;--s2:#52c98e;--s3:#e07cc4;--s4:#a496f0;--s5:#f08050;--s6:#46c8d4;--s7:#d2c050;--sx:#8a949a;
  --k-swap:#ff6b57;--k-missing:#f2b824;--k-extra:#a678ee;--k-overlap:#35c4d0;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 var(--font)}
header.top{padding:12px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
.row{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center;margin:4px 0}
h1{font-size:16px;margin:0;font-weight:650}
h3{font-size:13px;margin:16px 0 6px;color:var(--fg2);text-transform:uppercase;letter-spacing:.04em}
h4{margin:0 0 4px;font-size:12px;color:var(--fg2);font-weight:600}
.meta,small{color:var(--fg2);font-size:12px}
.spacer{flex:1}
select,button{font:inherit;font-size:13px;color:var(--fg);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 10px}
button{cursor:pointer}button:hover{border-color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:#fff}
.chip.base{border-style:dashed}
.chip .grp{font-size:10px;opacity:.7;border:1px solid currentColor;border-radius:8px;padding:0 4px;margin-left:4px}
.chip.degraded{border-color:var(--bad)}
.badge{font-size:10px;font-weight:700;border-radius:8px;padding:0 6px;margin-left:4px;border:1px solid currentColor}
.badge.bad{color:#fff;background:var(--bad);border-color:var(--bad)}
.badge.warn{color:#5c4300;background:#f5b942;border-color:#d99a00}
.banner.bad{border-left-color:var(--bad);background:var(--bad-bg)}.banner ul{margin:4px 0 0;padding-left:20px}
.vwarn{margin:6px 0;padding:6px 10px;border-left:4px solid var(--bad);background:var(--bad-bg);border-radius:4px;font-size:13px}
.vwarn.warn{border-left-color:var(--k-missing);background:var(--panel2)}
.banner{margin:8px 0 4px;padding:8px 12px;border-left:4px solid var(--k-missing);background:var(--panel2);font-size:13px;border-radius:4px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:13px}
th{font-weight:600;color:var(--fg2)}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.basecol{background:var(--panel2)}
.good{color:var(--good)}.bad{color:var(--bad)}
td.good{background:var(--good-bg)}td.bad{background:var(--bad-bg)}
.delta{font-size:11px;color:var(--dim)}.delta.good{color:var(--good)}.delta.bad{color:var(--bad)}
.dir{margin-left:6px;color:var(--dim)}
tr.vrow,tr.rrow{cursor:pointer}tr.vrow:hover,tr.rrow:hover{background:var(--accent-bg)}
tr.sel{background:var(--accent-bg)}
.sum th:first-child{white-space:nowrap}
nav[role=tablist]{display:flex;gap:2px;padding:0 16px;margin-top:10px;border-bottom:1px solid var(--line)}
nav button[role=tab]{border:0;border-bottom:3px solid transparent;border-radius:0;background:none;padding:8px 16px;font-weight:600;color:var(--fg2)}
nav button[aria-selected=true]{color:var(--accent);border-bottom-color:var(--accent)}
main{padding:12px 16px 60px;max-width:1500px;margin:0 auto}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:10px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 12px;min-width:90px}
.stat b{display:block;font-size:18px;font-variant-numeric:tabular-nums}.stat span{font-size:11px;color:var(--fg2)}
.ctl{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;margin:8px 0}
.empty{color:var(--fg2);padding:10px 0}
.cards{display:flex;flex-direction:column;gap:8px}
.seg-card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
.seg-card header{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;margin-bottom:4px}
.seg-card .time{font-variant-numeric:tabular-nums;font-weight:600}
.seg-card .spk{background:var(--accent-bg);color:var(--accent);border-radius:10px;padding:0 8px;font-size:12px}
.seg-card .wer{font-size:12px;color:var(--fg2)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.cols p{margin:0;font-size:15px}
@media (max-width:720px){.cols,.two{grid-template-columns:1fr!important}}
button.mini{font-size:12px;padding:1px 8px}
.w.sub{background:var(--sub);color:var(--subfg);border-radius:3px;padding:0 2px}
del.w.del{background:var(--del);color:var(--delfg);text-decoration:line-through;border-radius:3px;padding:0 2px}
ins.w.ins{background:var(--ins);color:var(--insfg);text-decoration:underline;border-radius:3px;padding:0 2px}
.tlwrap{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px}
svg.tl{width:100%;height:150px;display:block}
svg.tl .lane{fill:var(--panel2)}svg.tl .grid{stroke:var(--line);stroke-width:1;vector-effect:non-scaling-stroke}
svg.tl text{fill:var(--fg2);font-size:11px;font-family:var(--font)}
svg.tl .seg{stroke:var(--panel);stroke-width:.5;vector-effect:non-scaling-stroke}
.s0{fill:var(--s0);background:var(--s0)}.s1{fill:var(--s1);background:var(--s1)}.s2{fill:var(--s2);background:var(--s2)}
.s3{fill:var(--s3);background:var(--s3)}.s4{fill:var(--s4);background:var(--s4)}.s5{fill:var(--s5);background:var(--s5)}
.s6{fill:var(--s6);background:var(--s6)}.s7{fill:var(--s7);background:var(--s7)}.sx{fill:var(--sx);background:var(--sx)}
svg.tl .band{fill-opacity:.18;stroke-width:1.5;vector-effect:non-scaling-stroke;cursor:pointer}
svg.tl .band.sel{fill-opacity:.38}
.band-swap{fill:var(--k-swap);stroke:var(--k-swap);background:var(--k-swap)}
.band-missing{fill:var(--k-missing);stroke:var(--k-missing);stroke-dasharray:6 3;background:var(--k-missing)}
.band-extra{fill:var(--k-extra);stroke:var(--k-extra);stroke-dasharray:2 3;background:var(--k-extra)}
.band-overlap{fill:var(--k-overlap);stroke:var(--k-overlap);stroke-dasharray:10 2 2 2;background:var(--k-overlap)}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:6px 0;font-size:12px;color:var(--fg2)}
.lg{display:inline-flex;align-items:center;gap:5px}.sw{display:inline-block;width:12px;height:12px;border-radius:3px}
.kind{font-size:12px;font-weight:600}
.kind-swap{color:var(--k-swap)}.kind-missing{color:var(--k-missing)}.kind-extra{color:var(--k-extra)}.kind-overlap{color:var(--k-overlap)}
.two{display:grid;grid-template-columns:2fr 1fr;gap:16px}
svg.mb .vb{fill:var(--accent)}svg.mb .bb{fill:var(--dim)}svg.mb .nb{fill:var(--line)}
</style>
</head>
<body>
<header class="top">
  <div class="row"><h1>__TITLE__</h1>
    <label class="meta">file <select id="filesel" aria-label="file">__FILES__</select></label>
    <span class="meta">baseline: <strong>__BASELINE__</strong> &middot; __NFILES__ file(s)</span>
    <span class="spacer"></span><span class="meta" id="embednote"></span>
  </div>
  <div class="row"><span class="meta">variants</span><div class="chips" role="group" aria-label="variants">__CHIPS__</div>
    <span class="meta">keys: 1 2 3 tabs, [ ] variant, Esc stops audio</span></div>
  __DEGRADED__
  <div class="banner" role="note"><strong>Different does not mean better.</strong> Every comparison here is against the baseline transcript and diarization, which are not ground truth.<span id="refnote"></span></div>
  <div id="summary"></div>
  <noscript><p class="banner">This page needs JavaScript to show its data.</p></noscript>
</header>
<nav role="tablist" aria-label="views">
  <button type="button" role="tab" id="btn-text" data-tab="text" aria-controls="tab-text" aria-selected="true">Text</button>
  <button type="button" role="tab" id="btn-speakers" data-tab="speakers" aria-controls="tab-speakers" aria-selected="false">Speakers</button>
  <button type="button" role="tab" id="btn-metrics" data-tab="metrics" aria-controls="tab-metrics" aria-selected="false">Metrics</button>
</nav>
<main>
  <section id="tab-text" role="tabpanel" aria-labelledby="btn-text"></section>
  <section id="tab-speakers" role="tabpanel" aria-labelledby="btn-speakers" hidden></section>
  <section id="tab-metrics" role="tabpanel" aria-labelledby="btn-metrics" hidden></section>
</main>
<script type="application/json" id="data">__DATA__</script>
<script type="application/json" id="clips">__CLIPS__</script>
<script>__SCRIPT__</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
