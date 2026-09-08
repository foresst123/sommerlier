"""Build a single-file HTML review page for one processed recording.

The page puts each segment's original audio, its processed audio, all three ASR
outputs and the fused text side by side, with an editable copy of the fused text
and a note field. Saving writes a corrected transcript back to disk.

The audio is embedded as data: URIs so the page is one file that can be moved,
copied or opened straight from a download -- a page referring to clips by path
stops working the moment it leaves the directory it was built in.

Usage:
    python tools/make_review_page.py OUTPUT_DIR [-o review.html]

OUTPUT_DIR is a per-file directory under _final/, the one holding
{name}.json alongside 02_diarization/ and the rest.
"""

import argparse
import base64
import html
import json
import mimetypes
import os
import sys

# Above this the page becomes slow to open and awkward to scroll; the audio
# dominates the size, so the cap is on total bytes rather than segment count.
MAX_EMBED_BYTES = 400 * 1024 * 1024


def _load_transcript(out_dir: str):
    """The final transcript, plus the name it was keyed under."""
    name = os.path.basename(os.path.normpath(out_dir))
    path = os.path.join(out_dir, f"{name}.json")
    if not os.path.exists(path):
        candidates = [f for f in os.listdir(out_dir)
                      if f.endswith(".json") and not f.startswith("manifest")]
        if not candidates:
            raise SystemExit(f"no transcript JSON in {out_dir}")
        path = os.path.join(out_dir, candidates[0])
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    segments = data["segments"] if isinstance(data, dict) else data
    return name, segments, path


def _audio_index(out_dir: str, name: str):
    """Map segment index -> (original clip, processed clip), by filename prefix.

    Both directories name their files "{index}_{speaker}...", so the index is
    the prefix up to the first underscore. Matching on that rather than
    rebuilding the whole filename keeps this working when the speaker label
    changes between runs.
    """
    original, processed = {}, {}
    for src, target in ((os.path.join(out_dir, name), original),
                        (os.path.join(out_dir, "separation"), processed)):
        if not os.path.isdir(src):
            continue
        for fn in os.listdir(src):
            key = fn.split("_", 1)[0]
            target.setdefault(key, os.path.join(src, fn))
    return original, processed


def _data_uri(path: str, budget: list):
    """Embed a clip, or return None once the size budget is spent."""
    if not path or not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    if size > budget[0]:
        return None
    budget[0] -= size
    mime = mimetypes.guess_type(path)[0] or "audio/mpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def _rows(segments, original, processed, budget):
    out = []
    for seg in segments:
        idx = str(seg.get("index", ""))
        out.append({
            "index": idx,
            "speaker": str(seg.get("speaker", "")),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "whisper": seg.get("text_whisper") or "",
            "phowhisper": seg.get("text_phowhisper") or "",
            "qwen3": seg.get("text_qwen3") or "",
            "final": seg.get("text") or "",
            "edited": seg.get("text_edited") or seg.get("text") or "",
            "note": seg.get("note") or "",
            "audio_src": _data_uri(original.get(idx), budget),
            "audio_out": _data_uri(processed.get(idx), budget),
            # Two things a reviewer cannot hear their way to reliably: whether
            # the detector found music under the speech, and whether an overlap
            # was left unseparated so the audio still holds two voices.
            "music": bool(seg.get("has_music")),
            "bs_roformer": bool(seg.get("bs_roformer")),
            "unseparated": [
                {"start": float(u.get("start", 0.0)), "end": float(u.get("end", 0.0)),
                 "reason": str(u.get("reason", ""))}
                for u in (seg.get("unseparated") or [])
            ],
            "bss": bool(seg.get("bss")),
            # Two judgements only a person can make, so they start empty and
            # stay empty until someone ticks them. Read back from the segment
            # as well, which is what lets an edited JSON round-trip through a
            # regenerated page without losing the marks already made.
            "mark_music": bool(seg.get("mark_music")),
            "mark_multi": bool(seg.get("mark_multi")),
        })
    return out


PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review · __TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
/* === Warm beige / sand theme === */
:root {
  --bg: #f5eedc; /* Light yellow-brown/sand */
  --bg-elevated: #fdfbf7; --bg-card: #fdfbf7;
  --bg-card-hover: #fcf6e8; --fg: #332d27; /* Dark brown grey */
  --fg-secondary: #5c544d;
  --fg-dim: #a69c91; --accent: #92400e; /* Deep amber/brown accent */
  --accent-hover: #713f12;
  --accent-bg: rgba(146,64,14,.08); --accent-border: rgba(146,64,14,.25);
  --green: #065f46; --green-bg: rgba(6,95,70,.07); --green-fg: #064e3b;
  --amber: #b45309; --amber-bg: rgba(180,83,9,.08); --amber-fg: #78350f;
  --red: #b91c1c; --red-bg: rgba(185,28,28,.07); --red-fg: #7f1d1d;
  --border: rgba(120,110,100,.18); --border-strong: rgba(120,110,100,.28);
  --edit-bg: #fffcf5; --note-bg: #f3efe6;
  --shadow-card: 0 1px 3px rgba(100,90,80,.1), 0 1px 2px rgba(100,90,80,.06);
  --shadow-hover: 0 4px 14px rgba(100,90,80,.15);
  --radius: 10px; --radius-sm: 6px; --ok: #065f46; --hh: 62px;
}

*{box-sizing:border-box;margin:0;}
body{
  background:var(--bg);color:var(--fg);
  font-family:'Inter',-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  font-size:14px;line-height:1.5;overflow:hidden;
}

/* === Header === */
header{
  position:sticky;top:0;z-index:10;
  background:var(--bg-elevated);
  border-bottom:1px solid var(--border-strong);
  padding:10px 20px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;
  box-shadow:0 1px 4px rgba(0,0,0,.04);
}
h1{font-size:15px;font-weight:700;letter-spacing:-.01em;color:var(--fg);}
.meta{color:var(--fg-secondary);font-size:12px;}
.spacer{flex:1;}

.progress-wrap{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--fg-secondary);}
.progress-bar{width:80px;height:4px;background:var(--border-strong);border-radius:2px;overflow:hidden;}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent-hover));border-radius:2px;transition:width .4s ease;}

button{
  font:inherit;font-size:13px;font-weight:500;
  padding:6px 14px;border-radius:var(--radius-sm);cursor:pointer;
  border:1px solid var(--border-strong);background:var(--bg-elevated);color:var(--fg);
  transition:all .15s ease;
}
button:hover{border-color:var(--accent);color:var(--accent);}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;}
button.primary:hover{background:var(--accent-hover);border-color:var(--accent-hover);}
#status{color:var(--ok);font-size:12px;min-width:12ch;}

.pick{font-size:12px;color:var(--fg-secondary);display:inline-flex;align-items:center;gap:6px;}
.pick select{
  font:inherit;font-size:12px;padding:5px 10px;border-radius:var(--radius-sm);
  border:1px solid var(--border-strong);background:var(--bg-elevated);color:var(--fg);cursor:pointer;
}
kbd.hint{
  font:inherit;font-size:11px;color:var(--fg-dim);cursor:help;
  border:1px dashed var(--border-strong);border-radius:var(--radius-sm);padding:4px 8px;
}
kbd.hint:hover{color:var(--fg-secondary);}

/* === Card List === */
.wrap{
  overflow-y:auto;overflow-x:hidden;
  height:calc(100vh - var(--hh) - 74px);
  padding:10px 16px;overscroll-behavior:contain;
}
.card-list{display:flex;flex-direction:column;gap:5px;max-width:1600px;margin:0 auto;}

/* === Card: 3-column grid — Left meta | Center content+edit | Right note+marks === */
.card{
  display:grid;grid-template-columns:80px 1fr 240px;
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:var(--radius);box-shadow:var(--shadow-card);
  transition:box-shadow .2s,border-color .2s,background .15s;
  overflow:hidden;border-left:3px solid transparent;
}
.card:hover{
  background:var(--bg-card-hover);box-shadow:var(--shadow-hover);
  border-color:var(--border-strong);
}
.card.cur{
  border-left-color:var(--accent);
  background:color-mix(in srgb,var(--accent) 4%,var(--bg-card));
}
.card.playing{border-left-color:var(--accent);animation:pulse-border 2s ease-in-out infinite;}
@keyframes pulse-border{
  0%,100%{border-left-color:var(--accent);}
  50%{border-left-color:var(--accent-hover);}
}
.card.changed .seg-idx{color:var(--accent);}
.card.marked{border-left-color:var(--accent);}
.card.hidden{display:none;}

/* --- Card Left: ID & Meta --- */
.card-left{
  display:flex;flex-direction:column;align-items:center;
  padding:0;border-right:1px solid var(--border);
  background:color-mix(in srgb,var(--bg) 60%,var(--bg-card));
  position:relative;
}
.seg-idx{
  position:absolute;top:0;left:0;
  padding:3px 6px 2px 4px;
  border-bottom:1px solid var(--border-strong);
  border-right:1px solid var(--border-strong);
  border-bottom-right-radius:5px;
  font-size:10px;font-weight:600;color:var(--fg-secondary);
  font-variant-numeric:tabular-nums;line-height:1;transition:color .2s;
  background:var(--bg-elevated);
}
.seg-spk{
  flex:1;display:flex;align-items:center;justify-content:center;
  font-size:14px;font-weight:700;color:var(--fg);
  text-align:center;padding:15px 5px 5px;
}
.seg-time{
  font-size:10px;color:var(--fg-secondary);font-variant-numeric:tabular-nums;
  text-align:center;line-height:1.2;padding-bottom:10px;
}
.seg-dash{color:var(--fg-dim);font-size:9px;}
.seg-dur{color:var(--fg-dim);font-weight:500;font-size:9.5px;}

/* --- Card Center: Audio + ASR + Final + Edit --- */
.card-center{display:flex;flex-direction:column;padding:8px 14px;gap:6px;min-width:0;}

.card-top-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.card-audio{display:flex;gap:5px;}
.card-flags{display:flex;gap:5px;margin-left:auto;}

/* Play buttons */
button.play{
  padding:4px 11px;font-size:11px;font-weight:600;
  border-radius:var(--radius-sm);border:1px solid var(--border-strong);
  background:var(--bg-elevated);color:var(--fg);
  display:inline-flex;align-items:center;gap:4px;min-width:70px;justify-content:center;
}
button.play:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-bg);}
button.play.on{background:var(--accent);border-color:var(--accent);color:#fff;}
.play-ico{font-size:9px;}
.no-audio{color:var(--fg-dim);font-size:11px;}

/* Badges for music/separation */
.badge{
  display:inline-flex;align-items:center;justify-content:center;
  padding:4px 8px;border-radius:12px;
  border:1px solid transparent;cursor:help;
  position:relative;
}
/* Custom instant tooltip */
.badge[title]:hover::after {
  content: attr(title);
  position: absolute;
  top: 100%; right: 0;
  margin-top: 5px;
  background: var(--fg); color: var(--bg);
  padding: 5px 10px; border-radius: 5px;
  font-size: 11px; font-weight: 500;
  white-space: nowrap; z-index: 50;
  pointer-events: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.badge-muted{color:var(--fg-dim);border-color:transparent;}
.badge-ok{color:var(--green-fg);background:var(--green-bg);border-color:var(--green);}
.badge-warn{color:var(--amber-fg);background:var(--amber-bg);border-color:var(--amber);}
.badge-bad{color:var(--red-fg);background:var(--red-bg);border-color:var(--red);}
.badge-icon{font-size:14px;line-height:1;}

/* ASR lines */
.card-asr{display:flex;flex-direction:column;gap:2px;}
.asr-line{
  font-size:12.5px;color:var(--fg-secondary);
  display:flex;align-items:flex-start;gap:6px;line-height:1.45;
  cursor:pointer;padding:2px 4px;border-radius:4px;
  transition:background-color .15s;
}
.asr-line:hover{background:var(--bg-card-hover);}
.asr-tag{
  display:inline-block;min-width:16px;font-size:9px;font-weight:700;
  color:var(--fg-dim);text-transform:uppercase;letter-spacing:.04em;margin-right:5px;
  background:var(--border);padding:1px 4px;border-radius:3px;
}

/* Final text */
.card-final{
  font-size:13px;line-height:1.45;color:var(--fg);padding:6px 10px;
  border-radius:var(--radius-sm);background:var(--accent-bg);border-left:3px solid var(--accent);
  font-weight:500;
}
.final-icon{color:var(--accent);margin-right:5px;font-weight:700;}

/* Edit textarea — directly under final text */
.card-edit-wrap{margin-top:2px;}
textarea.edit{
  width:100%;font:inherit;font-size:13.5px;padding:7px 10px;
  border:1px solid var(--border-strong);border-radius:var(--radius-sm);
  resize:vertical;color:var(--fg);line-height:1.45;
  background:var(--edit-bg);
  min-height:40px;overflow:hidden;
  transition:border-color .15s,box-shadow .15s;
}
textarea.edit:focus{
  outline:none;border-color:var(--accent);
  /* A wide, opaque ring rather than a 6%-alpha tint: while you are typing,
     which box has focus should be readable from across the desk. */
  box-shadow:0 0 0 3px rgba(194,65,12,.22);
  background:#fffdf7;
}
textarea.edit::placeholder{color:var(--fg-dim);font-size:11px;}

/* --- Card Right: Note + Marks --- */
.card-right{
  display:flex;flex-direction:column;padding:8px 10px;gap:6px;
  border-left:1px solid var(--border);
}
.card-right-label{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:var(--fg-dim);}
textarea.note{
  width:100%;font:inherit;font-size:13px;padding:7px 9px;
  border:1px solid var(--border-strong);border-radius:var(--radius-sm);
  resize:vertical;color:var(--fg);line-height:1.45;
  background:var(--note-bg);flex:1;min-height:48px;overflow:hidden;
  transition:border-color .15s,box-shadow .15s;
}
textarea.note:focus{
  outline:none;border-color:var(--accent);
  box-shadow:0 0 0 3px var(--accent-bg);
}
textarea.note::placeholder{color:var(--fg-dim);font-size:11px;}

/* Mark checkboxes */
.card-marks{display:flex;gap:4px;margin-top:auto;flex-wrap:wrap;}
.mark-label{
  display:flex;align-items:center;gap:5px;cursor:pointer;user-select:none;
  font-size:11px;color:var(--fg-secondary);padding:4px 8px;
  border-radius:var(--radius-sm);border:1px solid var(--border);
  transition:background .12s,border-color .12s;
}
.mark-label:hover{background:var(--accent-bg);border-color:var(--accent-border);}
.mark-label input[type="checkbox"]{
  appearance:none;-webkit-appearance:none;margin:0;cursor:pointer;
  width:15px;height:15px;border:1.5px solid var(--fg-dim);border-radius:3px;
  background:var(--bg-elevated);position:relative;display:block;flex-shrink:0;transition:all .12s;
}
.mark-label input:checked{background:var(--accent);border-color:var(--accent);}
.mark-label input:checked::after{
  content:"";position:absolute;left:4px;top:1px;
  width:4.5px;height:8.5px;border:solid #fff;
  border-width:0 2px 2px 0;transform:rotate(42deg);
}
.mark-label input:checked+span{color:var(--accent);font-weight:600;}
.mark-label input:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}

/* === Transport Bar === */
#bar{
  position:fixed;left:0;right:0;bottom:0;z-index:20;
  background:var(--bg-elevated);
  border-top:1px solid var(--border-strong);padding:10px 20px;
  display:flex;gap:14px;align-items:center;
  box-shadow:0 -2px 8px rgba(0,0,0,.06);
}
#bar audio{flex:1;width:auto;height:36px;}
#bar .who{font-size:13px;min-width:15ch;}
#bar .who b{display:block;font-weight:600;}
#bar .who span{color:var(--fg-secondary);font-size:12px;}
#bar .src{font-size:11px;color:var(--fg-secondary);min-width:9ch;}

/* === Responsive === */
@media(max-width:1000px){
  .card{grid-template-columns:70px 1fr 200px;}
}
@media(max-width:800px){
  .card{grid-template-columns:1fr;grid-template-rows:auto auto auto;}
  .card-left{flex-direction:row;justify-content:flex-start;gap:10px;padding:8px 14px;border-right:none;border-bottom:1px solid var(--border);}
  .card-right{border-left:none;border-top:1px solid var(--border);}
}
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="meta">__COUNT__ đoạn</span>
  <span class="meta" id="changed"></span>
  <span class="meta" id="shown"></span>
  <div class="progress-wrap" id="progress-wrap">
    <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    <span id="progress-text"></span>
  </div>
  <label class="pick">Xem
    <select id="filter">
      <option value="all">tất cả</option>
      <option value="todo">chưa đánh dấu</option>
      <option value="marked">đã đánh dấu</option>
      <option value="music">còn nhạc</option>
      <option value="multi">nhiều giọng</option>
      <option value="issue">máy báo lỗi</option>
    </select>
  </label>
  <label class="pick" style="margin-left:8px;">Tốc độ
    <select id="rate">
      <option value="1">1.0x</option>
      <option value="1.25">1.25x</option>
      <option value="1.5">1.5x</option>
      <option value="2">2.0x</option>
    </select>
  </label>
  <kbd class="hint" title="j/k hoặc ↑/↓ chuyển dòng · Space nghe · M đánh dấu còn nhạc · V đánh dấu nhiều giọng · E sửa text">? phím tắt</kbd>
  <span class="spacer"></span>
  <span id="status"></span>
  <button id="playall">▶ Phát toàn bộ</button>
  <button id="export">Tải JSON</button>
  <button id="save" class="primary">Lưu</button>
</header>

<div class="wrap">
  <div id="tbody" class="card-list"></div>
</div>

<div id="bar">
  <div class="who"><b id="bar-id">—</b><span id="bar-time"></span></div>
  <audio id="player" controls preload="none"></audio>
  <div class="src" id="bar-src"></div>
  <button id="bar-stop">Dừng</button>
</div>

<script id="payload" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const NAME = __NAME__;
const tbody = document.getElementById("tbody");

const esc = s => (s ?? "").replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const clock = t => {
  const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
};

function playCell(i, which, has) {
  if (!has) return '<span class="no-audio">—</span>';
  const label = which === "src" ? "Gốc" : "Xử lý";
  return `<button class="play" data-i="${i}" data-which="${which}"><span class="play-ico">▶</span> ${label}</button>`;
}

function musicCell(r) {
  if (r.bss) return '<span class="badge badge-muted" title="Đoạn đã tách — không qua bộ dò nhạc">·</span>';
  if (!r.music) return '<span class="badge badge-muted" title="Không phát hiện nhạc">·</span>';
  return r.bs_roformer
    ? '<span class="badge badge-ok" title="Thành công: Phát hiện nhạc, đã lọc bằng BS-RoFormer"><span class="badge-icon">🎵</span></span>'
    : '<span class="badge badge-bad" title="Lỗi: Phát hiện nhạc nhưng CHƯA lọc!"><span class="badge-icon">🎵</span></span>';
}

const SEP_LABEL = {
  multi_speaker: "còn 2 giọng",
  no_enroll: "thiếu mẫu giọng",
  empty_track: "tách ra rỗng",
  not_a_fail: "không chắc đúng người",
  qc_sim: "độ khớp thấp",
  unscorable: "không chấm được",
};

function sepCell(r) {
  const u = r.unseparated || [];
  if (!u.length) {
    return r.bss
      ? '<span class="badge badge-ok" title="Thành công: Đã tách chồng tiếng"><span class="badge-icon">👥</span></span>'
      : '<span class="badge badge-muted" title="Không có chồng tiếng">·</span>';
  }
  const total = u.reduce((a, x) => a + (x.end - x.start), 0);
  const reasons = [...new Set(u.map(x => SEP_LABEL[x.reason] || x.reason))].join(", ");
  return `<span class="badge badge-bad" title="Lỗi: Còn ${total.toFixed(1)}s chưa tách — ${esc(reasons)}"><span class="badge-icon">👥</span></span>`;
}

DATA.forEach((r, i) => {
  const card = document.createElement("div");
  card.className = "card";
  card.dataset.i = i;
  card.innerHTML = `
    <div class="card-left">
      <div class="seg-idx">${+r.index}</div>
      <div class="seg-spk">${esc(r.speaker)}</div>
      <div class="seg-time">
        ${clock(r.start)} <span class="seg-dash">–</span><br>
        ${clock(r.end)} <span class="seg-dur">(${(r.end - r.start).toFixed(1)}s)</span>
      </div>
    </div>
    <div class="card-center">
      <div class="card-top-row">
        <div class="card-audio">
          ${playCell(i, "src", !!r.audio_src)}
          ${playCell(i, "out", !!r.audio_out)}
        </div>
        <div class="card-flags">
          ${musicCell(r)}
          ${sepCell(r)}
        </div>
      </div>
      <div class="card-asr">
        <div class="asr-line"><span class="asr-tag">W</span>${esc(r.whisper)}</div>
        <div class="asr-line"><span class="asr-tag">P</span>${esc(r.phowhisper)}</div>
        <div class="asr-line"><span class="asr-tag">Q</span>${esc(r.qwen3)}</div>
      </div>
      <div class="card-final"><span class="final-icon">✦</span>${esc(r.final)}</div>
      <div class="card-edit-wrap">
        <textarea rows="1" class="edit" placeholder="Sửa text nếu cần...">${esc(r.edited)}</textarea>
      </div>
    </div>
    <div class="card-right">
      <span class="card-right-label">Ghi chú</span>
      <textarea rows="2" class="note" placeholder="Ghi chú...">${esc(r.note)}</textarea>
      <div class="card-marks">
        <label class="mark-label" title="Còn nhạc nền"><input type="checkbox" class="mk-music"${r.mark_music ? " checked" : ""}><span>🎵 Nhạc</span></label>
        <label class="mark-label" title="Nhiều hơn một giọng"><input type="checkbox" class="mk-multi"${r.mark_multi ? " checked" : ""}><span>👥 Giọng</span></label>
        <label class="mark-label" title="Sai người nói"><input type="checkbox" class="mk-spk"${r.mark_spk ? " checked" : ""}><span>👤 Sai Spk</span></label>
      </div>
    </div>`;
  tbody.appendChild(card);
});

// Keep in-memory rows in step with the boxes
function syncRow(target) {
  const card = target.closest(".card");
  if (!card) return;
  const r = DATA[+card.dataset.i];
  if (target.classList.contains("edit")) r.edited = target.value;
  else if (target.classList.contains("note")) r.note = target.value;
  else if (target.classList.contains("mk-music")) r.mark_music = target.checked;
  else if (target.classList.contains("mk-multi")) r.mark_multi = target.checked;
  else if (target.classList.contains("mk-spk")) r.mark_spk = target.checked;
  else return;
  card.classList.toggle("changed", r.edited !== r.final || !!r.note);
  card.classList.toggle("marked", !!(r.mark_music || r.mark_multi || r.mark_spk));
  countChanged();
  dirty = true;
  if (filterSel.value !== "all") setTimeout(applyFilter, 0);
}
tbody.addEventListener("input", e => {
  if (e.target.tagName === 'TEXTAREA') {
    e.target.style.height = 'auto';
    e.target.style.height = e.target.scrollHeight + 'px';
  }
  syncRow(e.target);
});
tbody.addEventListener("change", e => syncRow(e.target));
tbody.addEventListener("dblclick", e => {
  const asrLine = e.target.closest(".asr-line");
  if (!asrLine) return;
  const tag = asrLine.querySelector(".asr-tag");
  let text = asrLine.textContent;
  if (tag) text = text.substring(tag.textContent.length);
  const card = asrLine.closest(".card");
  const editBox = card.querySelector("textarea.edit");
  if (editBox) {
    editBox.value = text;
    syncRow(editBox);
    editBox.style.height = "auto";
    editBox.style.height = editBox.scrollHeight + "px";
    editBox.style.transition = "none";
    editBox.style.backgroundColor = "var(--green-bg)";
    setTimeout(() => {
      editBox.style.transition = "background-color 0.5s";
      editBox.style.backgroundColor = "";
    }, 50);
  }
});

function countChanged() {
  const edited = DATA.filter(r => r.edited !== r.final || r.note).length;
  const music = DATA.filter(r => r.mark_music).length;
  const multi = DATA.filter(r => r.mark_multi).length;
  const spk = DATA.filter(r => r.mark_spk).length;
  const bits = [];
  if (edited) bits.push(`${edited} đã sửa`);
  if (music) bits.push(`${music} còn nhạc`);
  if (multi) bits.push(`${multi} nhiều giọng`);
  if (spk) bits.push(`${spk} sai spk`);
  document.getElementById("changed").textContent = bits.join(" · ");
  const total = DATA.length;
  const reviewed = DATA.filter(r => r.mark_music || r.mark_multi || r.mark_spk || r.edited !== r.final || r.note).length;
  const pct = total ? Math.round(reviewed / total * 100) : 0;
  document.getElementById("progress-fill").style.width = pct + "%";
  document.getElementById("progress-text").textContent = `${reviewed}/${total}`;
}
countChanged();
DATA.forEach((r, i) => {
  const card = tbody.children[i];
  if (r.edited !== r.final || r.note) card.classList.add("changed");
  if (r.mark_music || r.mark_multi || r.mark_spk) card.classList.add("marked");
});

const _hdr = document.querySelector("header");
const _pin = () => document.documentElement.style.setProperty(
  "--hh", _hdr.getBoundingClientRect().height + "px");
_pin();
addEventListener("resize", _pin);

// --- filtering and keyboard -------------------------------------------------
const FILTERS = {
  all:    () => true,
  todo:   r => !r.mark_music && !r.mark_multi,
  marked: r => r.mark_music || r.mark_multi,
  music:  r => r.mark_music,
  multi:  r => r.mark_multi,
  issue:  r => (r.unseparated && r.unseparated.length) || (r.music && !r.bs_roformer),
};
const filterSel = document.getElementById("filter");

function applyFilter() {
  const keep = FILTERS[filterSel.value] || FILTERS.all;
  let shown = 0;
  DATA.forEach((r, i) => {
    const on = keep(r);
    tbody.children[i].classList.toggle("hidden", !on);
    if (on) shown++;
  });
  document.getElementById("shown").textContent =
    shown === DATA.length ? "" : `hiện ${shown}/${DATA.length}`;
  if (cur >= 0 && tbody.children[cur].classList.contains("hidden")) setCur(nextVisible(0, 1));
}
filterSel.addEventListener("change", applyFilter);

let cur = -1;

function nextVisible(from, step) {
  for (let i = from; i >= 0 && i < DATA.length; i += step) {
    if (!tbody.children[i].classList.contains("hidden")) return i;
  }
  return -1;
}

function setCur(i) {
  tbody.querySelectorAll(".card.cur").forEach(c => c.classList.remove("cur"));
  cur = i;
  if (i < 0) return;
  const card = tbody.children[i];
  card.classList.add("cur");
  card.scrollIntoView({block: "nearest"});
}

function toggleMark(field) {
  if (cur < 0) return;
  const card = tbody.children[cur];
  card.querySelector(field === "mark_music" ? ".mk-music" : ".mk-multi").click();
}

addEventListener("keydown", e => {
  const typing = /^(TEXTAREA|INPUT|SELECT)$/.test(e.target.tagName)
                 && e.target.type !== "checkbox";
  if (typing) {
    if (e.key === "Escape") e.target.blur();
    return;
  }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "j" || e.key === "ArrowDown") { e.preventDefault(); setCur(nextVisible(cur < 0 ? 0 : cur + 1, 1)); }
  else if (k === "k" || e.key === "ArrowUp") { e.preventDefault(); setCur(nextVisible(cur <= 0 ? 0 : cur - 1, -1)); }
  else if (k === "m") { e.preventDefault(); toggleMark("mark_music"); }
  else if (k === "v") { e.preventDefault(); toggleMark("mark_multi"); }
  else if (k === "e") {
    e.preventDefault();
    if (cur >= 0) tbody.children[cur].querySelector("textarea.edit").focus();
  } else if (e.key === " ") {
    e.preventDefault();
    if (cur < 0) return;
    const r = DATA[cur];
    if (cur === current && !player.paused) { player.pause(); clearRow(); current = -1; }
    else { chain = false; play(cur, r.audio_out ? "out" : "src"); }
  }
});

tbody.addEventListener("mousedown", e => {
  const card = e.target.closest(".card");
  if (card) setCur(+card.dataset.i);
});

applyFilter();

let dirty = false;
addEventListener("beforeunload", e => { if (dirty) e.preventDefault(); });

// ---------------------------------------------------------------- transport
const player = document.getElementById("player");
document.getElementById("rate").addEventListener("change", e => {
  if (!player.paused) player.playbackRate = parseFloat(e.target.value) || 1.0;
});
let current = -1;
let source = "src";
let chain = false;

function clearRow() {
  tbody.querySelectorAll(".card.playing").forEach(c => c.classList.remove("playing"));
  tbody.querySelectorAll("button.play.on").forEach(b => {
    b.classList.remove("on");
    const label = b.dataset.which === "src" ? "Gốc" : "Xử lý";
    b.innerHTML = `<span class="play-ico">▶</span> ${label}`;
  });
}

function play(i, which) {
  const r = DATA[i];
  const url = which === "out" ? r.audio_out : r.audio_src;
  if (!url) return false;

  clearRow();
  current = i;
  source = which;

  const card = tbody.children[i];
  card.classList.add("playing");
  const btn = card.querySelector(`button.play[data-which="${which}"]`);
  if (btn) { btn.classList.add("on"); btn.innerHTML = '<span class="play-ico">⏸</span> Đang'; }

  document.getElementById("bar-id").textContent = `${r.index} · SP ${r.speaker}`;
  document.getElementById("bar-time").textContent =
    `${clock(r.start)} – ${clock(r.end)}`;
  document.getElementById("bar-src").textContent =
    which === "out" ? "sau xử lý" : "gốc";

  const rate = parseFloat(document.getElementById("rate").value) || 1.0;
  player.playbackRate = rate;
  player.src = url;
  player.play();
  const box = card.getBoundingClientRect();
  if (box.top < 80 || box.bottom > innerHeight - 90) {
    card.scrollIntoView({ block: "center", behavior: "smooth" });
  }
  return true;
}

function nextWith(from, which) {
  for (let i = from; i < DATA.length; i++) {
    if (which === "out" ? DATA[i].audio_out : DATA[i].audio_src) return i;
  }
  return -1;
}

tbody.addEventListener("click", e => {
  const btn = e.target.closest("button.play");
  if (!btn) return;
  const i = +btn.dataset.i, which = btn.dataset.which;
  if (i === current && source === which && !player.paused) {
    player.pause();
    clearRow();
    current = -1;
    chain = false;
    return;
  }
  chain = false;
  play(i, which);
});

player.addEventListener("ended", () => {
  if (!chain) { clearRow(); current = -1; return; }
  const next = nextWith(current + 1, source);
  if (next === -1) {
    chain = false;
    clearRow();
    current = -1;
    setPlayAll(false);
    flash("Đã phát hết");
    return;
  }
  play(next, source);
});

function setPlayAll(on) {
  const b = document.getElementById("playall");
  b.textContent = on ? "■ Dừng phát" : "▶ Phát toàn bộ";
  b.classList.toggle("primary", on);
}

document.getElementById("playall").onclick = () => {
  if (chain) {
    chain = false;
    player.pause();
    clearRow();
    current = -1;
    setPlayAll(false);
    return;
  }
  const from = current >= 0 ? current : 0;
  const i = nextWith(from, source);
  if (i === -1) { flash("Không có audio để phát"); return; }
  chain = true;
  setPlayAll(true);
  play(i, source);
};

document.getElementById("bar-stop").onclick = () => {
  chain = false;
  player.pause();
  clearRow();
  current = -1;
  setPlayAll(false);
};

addEventListener("keydown", e => {
  if (e.code !== "Space" || /^(TEXTAREA|INPUT)$/.test(e.target.tagName)) return;
  e.preventDefault();
  if (player.paused && player.src) player.play();
  else player.pause();
});

function flash(msg) {
  const el = document.getElementById("status");
  el.textContent = msg;
  setTimeout(() => { el.textContent = ""; }, 2600);
}

function download(filename, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

document.getElementById("save").onclick = () => {
  const doc = document.documentElement.cloneNode(true);
  doc.querySelector("#payload").textContent = JSON.stringify(DATA);
  doc.querySelectorAll("textarea").forEach(t => t.textContent = t.value);
  const p = doc.querySelector("#player");
  if (p) p.removeAttribute("src");
  doc.querySelectorAll(".card.playing").forEach(c => c.classList.remove("playing"));
  doc.querySelectorAll("button.play.on").forEach(b => {
    b.classList.remove("on");
    const label = b.dataset.which === "src" ? "Gốc" : "Xử lý";
    b.innerHTML = `<span class="play-ico">▶</span> ${label}`;
  });
  download(NAME + "_review.html",
           "<!doctype html>\\n" + doc.outerHTML, "text/html");
  dirty = false;
  flash("Đã lưu — ghi đè file cũ");
};

document.getElementById("export").onclick = () => {
  const rows = DATA.map(r => ({
    index: r.index, speaker: r.speaker, start: r.start, end: r.end,
    text: r.final, text_edited: r.edited, note: r.note,
    mark_music: r.mark_music, mark_multi: r.mark_multi,
  }));
  download(NAME + "_edited.json", JSON.stringify(rows, null, 2),
           "application/json");
  flash("Đã tải JSON");
};
</script>
</body>
</html>
"""


def build_review_page(output_dir: str, target: str = None, max_mb: int = None,
                      limit: int = None, logger=None):
    """Write the review page for one processed recording. Returns its path.

    Kept separate from main() so the pipeline can call it at the end of a run
    without going through argparse.
    """
    out_dir = os.path.abspath(output_dir)
    name, segments, _src = _load_transcript(out_dir)
    if limit:
        segments = segments[:limit]

    original, processed = _audio_index(out_dir, name)
    cap = (max_mb * 1024 * 1024) if max_mb else MAX_EMBED_BYTES
    budget = [cap]
    rows = _rows(segments, original, processed, budget)

    page = (PAGE
            .replace("__TITLE__", html.escape(name))
            .replace("__COUNT__", str(len(rows)))
            .replace("__NAME__", json.dumps(name))
            .replace("__DATA__", json.dumps(rows, ensure_ascii=False)))

    target = target or os.path.join(out_dir, f"{name}_review.html")
    with open(target, "w", encoding="utf-8") as f:
        f.write(page)

    if logger:
        size_mb = os.path.getsize(target) / (1024 * 1024)
        clips = sum(bool(r["audio_src"]) + bool(r["audio_out"]) for r in rows)
        flagged = sum(1 for r in rows if r["unseparated"])
        music = sum(1 for r in rows if r["music"] and not r["bs_roformer"])
        msg = (f"Review page: {len(rows)} segment(s), {clips} clip(s), "
               f"{size_mb:.1f} MB -> {target}")
        if flagged or music:
            msg += (f"  ({flagged} segment(s) with unseparated overlap, "
                    f"{music} with music left in)")
        if budget[0] <= 0:
            msg += "  (size cap reached; later clips were left out)"
        logger.info(msg)
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", help="per-file directory under _final/")
    ap.add_argument("-o", "--out", default=None, help="where to write the page")
    ap.add_argument("--max-mb", type=int, default=MAX_EMBED_BYTES // (1024 * 1024),
                    help="cap on embedded audio (default: %(default)s MB)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N segments")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    name, segments, src = _load_transcript(out_dir)
    if args.limit:
        segments = segments[:args.limit]

    original, processed = _audio_index(out_dir, name)
    budget = [args.max_mb * 1024 * 1024]
    rows = _rows(segments, original, processed, budget)

    embedded = sum(1 for r in rows if r["audio_src"]) + \
        sum(1 for r in rows if r["audio_out"])
    missing = sum(1 for r in rows if not r["audio_src"] or not r["audio_out"])

    page = (PAGE
            .replace("__TITLE__", html.escape(name))
            .replace("__COUNT__", str(len(rows)))
            .replace("__NAME__", json.dumps(name))
            .replace("__DATA__", json.dumps(rows, ensure_ascii=False)))

    target = args.out or os.path.join(out_dir, f"{name}_review.html")
    with open(target, "w", encoding="utf-8") as f:
        f.write(page)

    size_mb = os.path.getsize(target) / (1024 * 1024)
    print(f"transcript : {src}")
    print(f"segments   : {len(rows)}")
    print(f"audio      : {embedded} clip(s) embedded"
          + (f", {missing} row(s) missing one" if missing else ""))
    print(f"page       : {target}  ({size_mb:.1f} MB)")
    if budget[0] <= 0:
        print("warning    : the size cap was reached; later clips were skipped. "
              "Raise --max-mb or use --limit.", file=sys.stderr)


if __name__ == "__main__":
    main()
