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
{name}.json alongside 01_diarization/ and the rest.
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
  .wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; min-width: 1100px; }
  th, td {
    border-bottom: 1px solid var(--line); padding: 8px 10px;
    vertical-align: top; text-align: left;
  }
  th {
    position: sticky; top: 57px; background: var(--bg); z-index: 5;
    font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
    color: var(--muted); font-weight: 600;
  }
  tbody tr:nth-child(odd) { background: var(--row); }
  td.id { white-space: nowrap; font-variant-numeric: tabular-nums; color: var(--muted); }
  td.id b { color: var(--fg); display: block; font-variant-numeric: normal; }
  audio { width: 190px; height: 32px; display: block; }
  .no-audio { color: var(--muted); font-size: 12px; font-style: italic; }
  .asr { font-size: 13px; }
  .asr div { margin-bottom: 5px; }
  .asr span { color: var(--muted); font-size: 11px; display: block; }
  textarea {
    width: 100%; min-width: 210px; font: inherit; padding: 6px 8px;
    border: 1px solid var(--line); border-radius: 5px; resize: vertical;
    background: var(--edit-bg); color: var(--fg);
  }
  textarea.note { background: var(--bg); min-height: 42px; }
  td.final { font-size: 13px; }
  tr.changed td.id b { color: var(--accent); }
  td.time { white-space: nowrap; font-variant-numeric: tabular-nums;
            font-size: 12px; color: var(--muted); }
  td.time b { color: var(--fg); font-weight: 600; display: block; }
  tr.playing { background: color-mix(in srgb, var(--accent) 14%, transparent) !important; }
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
  body { padding-bottom: 74px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="meta">__COUNT__ đoạn</span>
  <span class="meta" id="changed"></span>
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
      <th>Đoạn</th>
      <th>Thời gian</th>
      <th>Audio gốc</th>
      <th>Sau xử lý</th>
      <th>3 bản ASR</th>
      <th>Text đã chọn</th>
      <th>Sửa</th>
      <th>Ghi chú</th>
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

DATA.forEach((r, i) => {
  const tr = document.createElement("tr");
  tr.dataset.i = i;
  tr.innerHTML = `
    <td class="id"><b>${esc(r.index)}</b>SP ${esc(r.speaker)}</td>
    <td class="time"><b>${clock(r.start)}</b>${clock(r.end)}<br>${(r.end - r.start).toFixed(2)}s</td>
    <td>${playCell(i, "src", !!r.audio_src)}</td>
    <td>${playCell(i, "out", !!r.audio_out)}</td>
    <td class="asr">
      <div><span>Whisper</span>${esc(r.whisper)}</div>
      <div><span>PhoWhisper</span>${esc(r.phowhisper)}</div>
      <div><span>Qwen3</span>${esc(r.qwen3)}</div>
    </td>
    <td class="final">${esc(r.final)}</td>
    <td><textarea rows="3" class="edit">${esc(r.edited)}</textarea></td>
    <td><textarea rows="3" class="note">${esc(r.note)}</textarea></td>`;
  tbody.appendChild(tr);
});

// Keep the in-memory rows in step with the boxes, so a save always writes
// what is on screen rather than what was loaded.
tbody.addEventListener("input", e => {
  const tr = e.target.closest("tr");
  const r = DATA[+tr.dataset.i];
  if (e.target.classList.contains("edit")) r.edited = e.target.value;
  else r.note = e.target.value;
  tr.classList.toggle("changed", r.edited !== r.final || !!r.note);
  countChanged();
  dirty = true;
});

function countChanged() {
  const n = DATA.filter(r => r.edited !== r.final || r.note).length;
  document.getElementById("changed").textContent = n ? `${n} đã sửa` : "";
}
countChanged();
DATA.forEach((r, i) => {
  if (r.edited !== r.final || r.note) tbody.children[i].classList.add("changed");
});

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
        msg = (f"Review page: {len(rows)} segment(s), {clips} clip(s), "
               f"{size_mb:.1f} MB -> {target}")
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
