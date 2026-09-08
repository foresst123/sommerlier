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
<title>__TITLE__</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #6b6b6b; --line: #e2e2e2;
    --row: #fafafa; --accent: #2563eb; --edit-bg: #fffbea; --ok: #16a34a;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #14161a; --fg: #e8e8e8; --muted: #9aa0a6; --line: #2c3038;
      --row: #1a1d22; --accent: #60a5fa; --edit-bg: #2a2620; --ok: #4ade80;
    }
  }
  :root[data-theme="dark"] {
    --bg: #14161a; --fg: #e8e8e8; --muted: #9aa0a6; --line: #2c3038;
    --row: #1a1d22; --accent: #60a5fa; --edit-bg: #2a2620; --ok: #4ade80;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  header {
    position: sticky; top: 0; z-index: 10; background: var(--bg);
    border-bottom: 1px solid var(--line); padding: 12px 16px;
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  }
  h1 { font-size: 16px; margin: 0; font-weight: 600; }
  .meta { color: var(--muted); font-size: 13px; }
  .spacer { flex: 1; }
  button {
    font: inherit; padding: 7px 14px; border-radius: 6px; cursor: pointer;
    border: 1px solid var(--line); background: var(--bg); color: var(--fg);
  }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:hover { filter: brightness(1.08); }
  #status { color: var(--ok); font-size: 13px; min-width: 12ch; }
  .pick { font-size: 13px; color: var(--muted); display: inline-flex;
          align-items: center; gap: 6px; }
  .pick select { font: inherit; padding: 5px 8px; border-radius: 6px;
                 border: 1px solid var(--line); background: var(--bg); color: var(--fg); }
  kbd.hint { font: inherit; font-size: 12px; color: var(--muted); cursor: help;
             border: 1px dashed var(--line); border-radius: 5px; padding: 4px 8px; }
  /* The table scrolls inside this, not the page, and that is what makes the
     column names stay put. `overflow-x: auto` alone does not work: the spec
     computes overflow-y to auto alongside it, so the wrapper silently becomes
     a vertical scroll container that never scrolls -- and a sticky <th> pins
     itself to that instead of to the window, scrolling away with the rows.
     Giving it a real height makes it the scroller it was already pretending
     to be. 74px is the transport bar at the foot. */
  .wrap {
    overflow: auto;
    height: calc(100vh - var(--hh, 62px) - 74px);
    overscroll-behavior: contain;
  }
  table { border-collapse: collapse; width: 100%; min-width: 1180px; }
  /* State reads at a glance from shape and colour together, so the two flag
     columns can be scanned without stopping to read every cell. */
  /* Read as a symbol, not a sentence. These two columns are scanned down the
     page rather than read row by row, and the words were costing 100px each
     for information a glyph carries; the wording moved into the tooltip. */
  .flag {
    display: inline-block; white-space: nowrap; font-size: 13px;
    font-weight: 600; line-height: 1.4; text-align: center; min-width: 22px;
  }
  th.c-flag { width: 42px; text-align: center; }
  td.c-flag { text-align: center; padding-left: 4px; padding-right: 4px; }
  .flag.none { color: var(--muted); }
  .flag.ok   { color: var(--ok); border-color: var(--ok); }
  .flag.warn { color: #b45309; border-color: #b45309; }
  .flag.bad  { color: #dc2626; border-color: #dc2626; font-weight: 600; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) .flag.warn { color: #d99331; border-color: #d99331; }
    :root:not([data-theme="light"]) .flag.bad  { color: #f87171; border-color: #f87171; }
  }
  :root[data-theme="dark"] .flag.warn { color: #d99331; border-color: #d99331; }
  :root[data-theme="dark"] .flag.bad  { color: #f87171; border-color: #f87171; }
  th, td {
    border-bottom: 1px solid var(--line); padding: 6px 8px;
    vertical-align: top; text-align: left;
  }
  /* Pinned to the top of .wrap rather than of the window: the page header
     already owns the top of the window, and a thead pinned there would slide
     underneath it and disappear exactly when a long table needs it. */
  th {
    position: sticky; top: 0; background: var(--bg); z-index: 5;
    font-size: 11px; text-transform: uppercase; letter-spacing: .03em;
    color: var(--muted); font-weight: 600; padding: 7px 8px;
    box-shadow: inset 0 -1px 0 var(--line);
  }
  tbody tr:nth-child(odd) { background: var(--row); }
  /* First column carries everything needed to find the row again: which
     segment it is, and where in the recording. Stacked rather than spread
     across two columns -- on a form you look in one place for the reference. */
  td.id { white-space: nowrap; padding: 7px 8px;
          font-variant-numeric: tabular-nums; line-height: 1.3; }
  td.id b { font-size: 14px; font-weight: 700; }
  td.id i { font-style: normal; font-size: 11px; color: var(--muted); margin-left: 6px; }
  td.id u { display: block; text-decoration: none; font-size: 10px; color: var(--muted); }
  td.spk { text-align: center; }
  td.spk span {
    display: inline-block; min-width: 24px; padding: 2px 7px; border-radius: 4px;
    border: 1px solid var(--line); font-size: 12px; font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  audio { width: 190px; height: 32px; display: block; }
  .no-audio { color: var(--muted); font-size: 12px; font-style: italic; }
  /* Three transcripts stacked with a label on its own line above each was
     three lines of chrome for three lines of text. The label sits inline now
     and each variant is capped at two lines -- long enough to compare wording,
     short enough that the row does not decide the height of the table. */
  .asr { font-size: 12.5px; line-height: 1.4; }
  .asr div {
    margin-bottom: 3px; display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden;
  }
  .asr div:last-child { margin-bottom: 0; }
  .asr span {
    color: var(--muted); font-size: 10px; text-transform: uppercase;
    letter-spacing: .04em; margin-right: 5px; font-weight: 600;
  }
  /* Row height is the whole reading experience here. A 200px edit box meant
     two or three rows on a screen, so reviewing 300 segments was 100 screens
     of scrolling; at this height it is nearer 12, which is what a dense table
     is supposed to give. The box grows to its content and grows again while
     it has focus, so editing a long line is still comfortable -- the space is
     spent when it is needed instead of reserved on every row. */
  textarea {
    width: 100%; min-width: 160px; font: inherit; padding: 5px 7px;
    border: 1px solid var(--line); border-radius: 5px; resize: vertical;
    background: var(--edit-bg); color: var(--fg);
    min-height: 46px; max-height: 120px; overflow-y: auto; line-height: 1.45;
  }
  textarea:focus { max-height: 300px; min-height: 90px; outline: 2px solid var(--accent);
                   outline-offset: -1px; border-color: var(--accent); }
  textarea.note { background: var(--bg); min-height: 46px; }
  td.final {
    font-size: 12.5px; line-height: 1.4; display: -webkit-box;
    -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
  }
  tr.changed td.id b { color: var(--accent); }
  td.time { white-space: nowrap; font-variant-numeric: tabular-nums;
            font-size: 12px; color: var(--muted); }
  td.time b { color: var(--fg); font-weight: 600; display: block; }
  tr.playing { background: color-mix(in srgb, var(--accent) 14%, transparent) !important; }
  /* Keyboard focus needs somewhere to be. Without a visible current row the
     shortcuts below are unusable: you cannot tell what j/k is about to act on. */
  tr.cur td { background: color-mix(in srgb, var(--accent) 8%, transparent); }
  tr.cur td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
  tr.cur.marked td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
  tr.hidden { display: none; }
  button.play {
    padding: 4px 10px; font-size: 12px; border-radius: 4px; min-width: 62px;
  }
  button.play.on { background: var(--accent); border-color: var(--accent); color: #fff; }

  /* A single transport at the foot of the window, the way a music player
     works: one element to control, and the row it belongs to stays visible
     while the table scrolls. Per-row <audio> tags meant hunting for whichever
     one was playing. */
  #bar {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 20;
    background: var(--bg); border-top: 1px solid var(--line);
    padding: 10px 16px; display: flex; gap: 14px; align-items: center;
    box-shadow: 0 -2px 12px rgba(0,0,0,.09);
  }
  #bar audio { flex: 1; width: auto; height: 36px; }
  #bar .who { font-size: 13px; min-width: 15ch; }
  #bar .who b { display: block; }
  #bar .who span { color: var(--muted); font-size: 12px; }
  #bar .src { font-size: 12px; color: var(--muted); min-width: 9ch; }
  body { overflow: hidden; }

  /* The two reviewer columns, built the way a paper form builds them: the
     question is asked once in the column head, and every row is just a box.
     Repeating the label in each cell is what made them wide and slow to scan.
     The label wraps the box so the whole cell is the target -- a reviewer
     ticking a thousand rows should not have to hit a 17px square. */
  th.mark-h { text-align: center; width: 44px; line-height: 1.15; font-size: 10px; }
  td.mark { padding: 0; vertical-align: middle; }
  td.mark label {
    display: flex; align-items: center; justify-content: center;
    min-height: 40px; height: 100%; cursor: pointer; user-select: none;
  }
  td.mark label:hover { background: color-mix(in srgb, var(--accent) 9%, transparent); }
  td.mark input {
    appearance: none; -webkit-appearance: none; margin: 0; cursor: pointer;
    width: 18px; height: 18px; border: 1.5px solid var(--muted); border-radius: 3px;
    background: var(--bg); position: relative; display: block;
  }
  td.mark input:checked { background: var(--accent); border-color: var(--accent); }
  td.mark input:checked::after {
    content: ""; position: absolute; left: 5px; top: 1px;
    width: 5px; height: 10px; border: solid #fff;
    border-width: 0 2px 2px 0; transform: rotate(42deg);
  }
  td.mark input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  /* A marked row has to be findable while scrolling a thousand of them, but
     the mark is a reviewer's note and not an error -- a rule on the edge, not
     a wash of colour across the row. */
  tr.marked td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="meta">__COUNT__ đoạn</span>
  <span class="meta" id="changed"></span>
  <span class="meta" id="shown"></span>
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
  <kbd class="hint" title="j/k hoặc ↑/↓ chuyển dòng · Space nghe · M đánh dấu còn nhạc · V đánh dấu nhiều giọng · E sửa text">? phím tắt</kbd>
  <span class="spacer"></span>
  <span id="status"></span>
  <button id="playall">▶ Phát toàn bộ</button>
  <button id="export">Tải JSON</button>
  <button id="save" class="primary">Lưu</button>
</header>

<div class="wrap">
<table>
  <thead>
    <tr>
      <th class="c-id">STT</th>
      <th class="c-spk">Giọng</th>
      <th>Audio gốc</th>
      <th>Sau xử lý</th>
      <th class="c-flag" title="Bộ dò nhạc thấy gì, và đã lọc chưa">Nhạc</th>
      <th class="c-flag" title="Chồng tiếng đã tách được chưa">Tách</th>
      <th>3 bản ASR</th>
      <th>Text đã chọn</th>
      <th>Sửa</th>
      <th>Ghi chú</th>
      <th class="mark-h" title="Người nghe tự đánh dấu: sau xử lý vẫn còn nhạc nền">Còn<br>nhạc</th>
      <th class="mark-h" title="Người nghe tự đánh dấu: đoạn này còn nhiều hơn một giọng">Nhiều<br>giọng</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>
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
  return has
    ? `<button class="play" data-i="${i}" data-which="${which}">▶ Phát</button>`
    : `<span class="no-audio">không có</span>`;
}

// A separated segment never reaches the music detector, so "no music" would be
// a claim the pipeline never made. Say "chưa xét" instead of implying clean.
function musicCell(r) {
  if (r.bss) return `<span class="flag none" title="Đoạn đã tách không qua bộ dò nhạc">·</span>`;
  if (!r.music) return `<span class="flag none" title="Không phát hiện nhạc">·</span>`;
  return r.bs_roformer
    ? `<span class="flag ok" title="Phát hiện nhạc, đã lọc">♪</span>`
    : `<span class="flag warn" title="Phát hiện nhạc nhưng CHƯA lọc">♪!</span>`;
}

// Reasons come from the separation report; a reviewer needs to know the audio
// still holds two voices before trusting what they hear in it.
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
      ? `<span class="flag ok" title="Đã tách chồng tiếng thành công">✓</span>`
      : `<span class="flag none" title="Không có chồng tiếng">·</span>`;
  }
  const total = u.reduce((a, x) => a + (x.end - x.start), 0);
  const reasons = [...new Set(u.map(x => SEP_LABEL[x.reason] || x.reason))].join(", ");
  return `<span class="flag bad" title="Còn ${total.toFixed(1)}s chưa tách — ${esc(reasons)}">✗</span>`;
}

DATA.forEach((r, i) => {
  const tr = document.createElement("tr");
  tr.dataset.i = i;
  tr.innerHTML = `
    <td class="id"><b>${+r.index}</b><i>${clock(r.start)}</i><u>${(r.end - r.start).toFixed(1)}s</u></td>
    <td class="spk"><span>${esc(r.speaker)}</span></td>
    <td>${playCell(i, "src", !!r.audio_src)}</td>
    <td>${playCell(i, "out", !!r.audio_out)}</td>
      <td class="c-flag">${musicCell(r)}</td>
      <td class="c-flag">${sepCell(r)}</td>
    <td class="asr">
      <div><span>Whisper</span>${esc(r.whisper)}</div>
      <div><span>PhoWhisper</span>${esc(r.phowhisper)}</div>
      <div><span>Qwen3</span>${esc(r.qwen3)}</div>
    </td>
    <td class="final">${esc(r.final)}</td>
    <td><textarea rows="2" class="edit">${esc(r.edited)}</textarea></td>
    <td><textarea rows="2" class="note">${esc(r.note)}</textarea></td>
    <td class="mark"><label title="Còn nhạc nền"><input type="checkbox" class="mk-music"${r.mark_music ? " checked" : ""}></label></td>
    <td class="mark"><label title="Còn nhiều hơn một giọng"><input type="checkbox" class="mk-multi"${r.mark_multi ? " checked" : ""}></label></td>`;
  tbody.appendChild(tr);
});

// Keep the in-memory rows in step with the boxes, so a save always writes
// what is on screen rather than what was loaded.
// Checkboxes fire "change" as well as "input" depending on the browser, and a
// mark that silently failed to save is worse than one that never existed.
// Listening to both, with an idempotent handler, costs nothing.
function syncRow(target) {
  const tr = target.closest("tr");
  if (!tr) return;
  const r = DATA[+tr.dataset.i];
  if (target.classList.contains("edit")) r.edited = target.value;
  else if (target.classList.contains("note")) r.note = target.value;
  else if (target.classList.contains("mk-music")) r.mark_music = target.checked;
  else if (target.classList.contains("mk-multi")) r.mark_multi = target.checked;
  else return;
  tr.classList.toggle("changed", r.edited !== r.final || !!r.note);
  tr.classList.toggle("marked", !!(r.mark_music || r.mark_multi));
  countChanged();
  dirty = true;
  // A row that no longer matches the filter should leave, but not while the
  // pointer is still on it -- so this runs on the next tick, after the click
  // has finished.
  if (filterSel.value !== "all") setTimeout(applyFilter, 0);
}
tbody.addEventListener("input", e => syncRow(e.target));
tbody.addEventListener("change", e => syncRow(e.target));

// Three counts, because they answer different questions: how much text was
// touched, and how much audio a listener judged unusable for each reason.
function countChanged() {
  const edited = DATA.filter(r => r.edited !== r.final || r.note).length;
  const music = DATA.filter(r => r.mark_music).length;
  const multi = DATA.filter(r => r.mark_multi).length;
  const bits = [];
  if (edited) bits.push(`${edited} đã sửa`);
  if (music) bits.push(`${music} còn nhạc`);
  if (multi) bits.push(`${multi} nhiều giọng`);
  document.getElementById("changed").textContent = bits.join(" · ");
}
countChanged();
DATA.forEach((r, i) => {
  const tr = tbody.children[i];
  if (r.edited !== r.final || r.note) tr.classList.add("changed");
  if (r.mark_music || r.mark_multi) tr.classList.add("marked");
});

// The table head pins directly under the page header, so it has to know how
// tall that is -- and it changes when the buttons wrap on a narrow window.
const _hdr = document.querySelector("header");
const _pin = () => document.documentElement.style.setProperty(
  "--hh", _hdr.getBoundingClientRect().height + "px");
_pin();
addEventListener("resize", _pin);

// --- filtering and keyboard -------------------------------------------------
//
// A reviewer does not work through a thousand rows once, top to bottom. They
// work a subset: the ones the pipeline flagged, then the ones they marked, then
// what is left. Without a filter that means scrolling past everything already
// dealt with, every pass.
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

// The current row. Shortcuts act on it, so it has to exist and be visible
// before any of them mean anything.
let cur = -1;

function nextVisible(from, step) {
  for (let i = from; i >= 0 && i < DATA.length; i += step) {
    if (!tbody.children[i].classList.contains("hidden")) return i;
  }
  return -1;
}

function setCur(i) {
  tbody.querySelectorAll("tr.cur").forEach(tr => tr.classList.remove("cur"));
  cur = i;
  if (i < 0) return;
  const tr = tbody.children[i];
  tr.classList.add("cur");
  tr.scrollIntoView({block: "nearest"});
}

function toggleMark(field) {
  if (cur < 0) return;
  const tr = tbody.children[cur];
  tr.querySelector(field === "mark_music" ? ".mk-music" : ".mk-multi").click();
}

// One hand on the keyboard is what makes this quick: move, listen, judge,
// without reaching for the mouse between rows. Typing must never trigger any
// of it, so anything with a text field focused falls straight through.
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

// Clicking anywhere in a row makes it the current one, so mouse and keyboard
// agree about where you are rather than each keeping their own idea.
tbody.addEventListener("mousedown", e => {
  const tr = e.target.closest("tr");
  if (tr) setCur(+tr.dataset.i);
});

applyFilter();

let dirty = false;
addEventListener("beforeunload", e => { if (dirty) e.preventDefault(); });

// ---------------------------------------------------------------- transport
const player = document.getElementById("player");
let current = -1;        // row being played, -1 for none
let source = "src";      // which column: original or processed
let chain = false;       // continue into the next row when this one ends

function clearRow() {
  tbody.querySelectorAll("tr.playing").forEach(tr => tr.classList.remove("playing"));
  tbody.querySelectorAll("button.play.on").forEach(b => {
    b.classList.remove("on");
    b.textContent = "▶ Phát";
  });
}

function play(i, which) {
  const r = DATA[i];
  const url = which === "out" ? r.audio_out : r.audio_src;
  if (!url) return false;

  clearRow();
  current = i;
  source = which;

  const tr = tbody.children[i];
  tr.classList.add("playing");
  const btn = tr.querySelector(`button.play[data-which="${which}"]`);
  if (btn) { btn.classList.add("on"); btn.textContent = "❚❚ Đang"; }

  document.getElementById("bar-id").textContent = `${r.index} · SP ${r.speaker}`;
  document.getElementById("bar-time").textContent =
    `${clock(r.start)} – ${clock(r.end)}`;
  document.getElementById("bar-src").textContent =
    which === "out" ? "sau xử lý" : "gốc";

  player.src = url;
  player.play();
  // Only scroll when the row has left the viewport, so a manual click does not
  // yank the page around.
  const box = tr.getBoundingClientRect();
  if (box.top < 80 || box.bottom > innerHeight - 90) {
    tr.scrollIntoView({ block: "center", behavior: "smooth" });
  }
  return true;
}

/** The next row that has audio in the current column. */
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
  chain = false;                      // a manual click plays just that row
  play(i, which);
});

// Walking to the next segment when one finishes is the point of the play-all
// button, but it also makes listening through a stretch by hand work without
// clicking every row.
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
  // Start from the row after the one showing, so pressing it again resumes
  // rather than jumping back to the top.
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

// Space toggles playback unless a text box has focus, where it types a space.
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

// Saving rewrites this page's own payload and hands back the whole file, so
// reopening it shows the edits. A browser cannot overwrite the file it is
// displaying -- the download replaces it in place once you confirm.
document.getElementById("save").onclick = () => {
  const doc = document.documentElement.cloneNode(true);
  doc.querySelector("#payload").textContent = JSON.stringify(DATA);
  doc.querySelectorAll("textarea").forEach(t => t.textContent = t.value);
  // The player holds whichever clip was last loaded as a data: URI. Cloning it
  // would write that clip into the file a second time, so clear it.
  const p = doc.querySelector("#player");
  if (p) p.removeAttribute("src");
  doc.querySelectorAll("tr.playing").forEach(tr => tr.classList.remove("playing"));
  doc.querySelectorAll("button.play.on").forEach(b => {
    b.classList.remove("on");
    b.textContent = "▶ Phát";
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
    // The reason this page exists: a downstream filter needs to know which
    // segments a person rejected, not only which ones they retyped.
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
