#!/usr/bin/env python3
"""One HTML page per recording: every variant as a block of rows, one waveform row per
speaker, the ASR text of that speaker right under it.

    python tools/make_variant_lanes_page.py --manifest OUT/variants_manifest.json \
        --out-dir OUT/lanes [--files stem1,stem2] [--transcript asr|refinement|latest] \
        [--peaks-per-sec 50] [--audio link|none]

Input is what tools/run_variants.py wrote. For each variant the audio drawn and played is
the one its later stages worked on: the latest filter point that wrote a wav
(04b_denoise_pre_asr, 02b_denoise_pre_separation, 01b_denoise), else 01_music/after_music.wav,
else the variant's input file (a run with the music stage off has no after_music.wav).

A speaker row draws the whole waveform faintly and that speaker's segments in colour, so
overlap shows up in two rows at once. Speaker labels are per variant: diarization ran
separately in each, so SPEAKER_00 of one variant need not be SPEAKER_00 of another.

--audio link (default) refers to the wav files by a path relative to the page, so keep the
page inside the same tree and serve OUT with `python tools/serve_results.py --root OUT --audio-dir AUDIO` (it answers
HTTP Range, so a long file plays and seeks without being downloaded whole, and it serves the original recordings at
/_audio/, which is where a variant with no wav of its own plays from). The page itself has no
CDN or network dependency.
"""

import argparse
import base64
import glob
import html
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.compare_lib.io_stage import _norm_segments, _rows, read_json, read_wav  # noqa: E402

DENOISE_POINTS = ("04b_denoise_pre_asr", "02b_denoise_pre_separation", "01b_denoise")
TRANSCRIPTS = {
    "asr": ("05_asr",),
    "refinement": ("06_refinement",),
    # what the dataset was cut from: word alignment runs last and rewrites the transcript
    "latest": ("08_word_alignment", "06_refinement", "05_asr"),
}
AUDIO_EXT = (".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg")


def input_dirs_of(out_root: str, vid: str) -> List[str]:
    """Where the original recording of a variant may be linked: the old runner's OUT/<id>/input, then the folders
    the resident line makes (OUT/_resident/input_*), newest first."""
    resident = sorted(glob.glob(os.path.join(glob.escape(out_root), "_resident", "input_*")),
                      key=lambda p: os.path.getmtime(p), reverse=True)
    return [os.path.join(out_root, vid, "input"), *resident]


def variant_audio(file_dir: str, input_dir, stem: str) -> Optional[str]:
    """The wav the variant's later stages worked on, see the module docstring. `input_dir` is one folder or a list."""
    for d in DENOISE_POINTS:
        p = os.path.join(file_dir, d, "after_denoise.wav")
        if os.path.isfile(p):
            return p
    p = os.path.join(file_dir, "01_music", "after_music.wav")
    if os.path.isfile(p):
        return p
    for folder in ([input_dir] if isinstance(input_dir, str) else (input_dir or [])):
        for p in sorted(glob.glob(os.path.join(glob.escape(folder), glob.escape(stem) + ".*"))):
            if p.lower().endswith(AUDIO_EXT):
                return os.path.realpath(p)
    return None


def variant_segments(file_dir: str, transcript: str) -> Optional[List[dict]]:
    for d in TRANSCRIPTS[transcript]:
        segs = _norm_segments(_rows(read_json(os.path.join(file_dir, d, "transcripts.json"))),
                              with_text=True)
        if segs is not None:
            return [{"s": round(s["start"], 3), "e": round(s["end"], 3),
                     "spk": s["speaker"], "t": s["text"]} for s in segs]
    return None


def peaks(path: Optional[str], per_sec: int):
    """(base64 of uint8 peak per bin, duration s); 255 is full scale, so levels compare across variants."""
    wav, sr = read_wav(path)
    if wav is None or not len(wav):
        return None, None
    hop = max(1, int(round(sr / per_sec)))
    n = int(np.ceil(len(wav) / hop))
    pad = np.zeros(n * hop, dtype=np.float32)
    pad[:len(wav)] = np.abs(wav)
    pk = np.minimum(pad.reshape(n, hop).max(axis=1), 1.0)
    return base64.b64encode(np.round(pk * 255).astype(np.uint8).tobytes()).decode("ascii"), len(wav) / sr


def _inside(path: str, folder: str) -> bool:
    """Is `path` under `folder` as written (symlinks not followed)?"""
    path, folder = os.path.abspath(path), os.path.abspath(folder)
    return path == folder or path.startswith(folder + os.sep)


def chain_text(v: dict) -> str:
    """The variant's full flow, written the way the plan writes it."""
    def part(key):
        return list(v.get(key) or [])
    flow = ["audio gốc"]
    flow += ["tiền xử lý"] if not v.get("set") else []
    flow += part("chain") + ["diarization"] + part("chain_pre_separation") + ["separation"]
    flow += part("chain_pre_asr") + ["asr"]
    return " → ".join(flow)


def add_text_stats(rows: List[dict]) -> None:
    """words and the word-level difference to the baseline text (not an error rate: there is no reference)."""
    from tools.compare_lib.text_align import compare_transcripts, tokens
    as_ref = lambda segs: [{"start": s["s"], "end": s["e"], "text": s["t"]} for s in segs]
    base = next((r for r in rows if r.get("baseline") and r.get("segments")), None)
    for r in rows:
        segs = r.get("segments")
        if segs is None:
            continue
        r["words"] = sum(len(tokens(s["t"])) for s in segs)
        if base is not None and r is not base:
            res = compare_transcripts(as_ref(base["segments"]), as_ref(segs), detail=False)
            if res and res["wer"] is not None:
                r["diff_vs_base"] = round(res["wer"], 4)


def build_page_data(manifest_path: str, stem: str, out_dir: str, transcript: str = "asr",
                    per_sec: int = 50, audio_mode: str = "link") -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    out_root = os.path.dirname(os.path.abspath(manifest_path))
    base_id = manifest.get("baseline", "none")
    rows = []
    for v in manifest.get("variants", []):
        file_dir = (v.get("files") or {}).get(stem)
        row = {"id": v["id"], "label": v.get("label") or v["id"], "flow": chain_text(v),
               "status": v.get("status", "ok"), "group": v.get("group"),
               "baseline": v["id"] == base_id}
        if not file_dir or not os.path.isdir(file_dir):
            row["missing"] = f"no output for {stem} (status {row['status']})"
            rows.append(row)
            continue
        audio = variant_audio(file_dir, input_dirs_of(out_root, v["id"]), stem)
        row["peaks"], row["duration"] = peaks(audio, per_sec)
        row["segments"] = variant_segments(file_dir, transcript)
        if audio and audio_mode == "link":
            if _inside(audio, out_root):
                row["audio"] = os.path.relpath(audio, out_dir).replace(os.sep, "/")
            else:   # the original recording lives outside OUT: tools/serve_results.py --audio-dir serves it here
                row["audio"] = "../_audio/" + os.path.basename(audio)
                row["audio_source"] = "input"
        if row["segments"] is None:
            row["missing"] = "no transcripts.json"
        rows.append(row)
    rows.sort(key=lambda r: not r["baseline"])  # baseline on top, plan order after it
    add_text_stats(rows)
    return {"file": stem, "peaks_per_sec": per_sec, "transcript": transcript, "variants": rows}


def stems_of(manifest_path: str) -> List[str]:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    out = []
    for v in manifest.get("variants", []):
        for s in (v.get("files") or {}):
            if s not in out:
                out.append(s)
    return out


def render(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (PAGE.replace("__TITLE__", html.escape(data["file"]))
                .replace("__DATA__", payload))


PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Variant lanes · __TITLE__</title>
<style>
:root{--bg:#fbfaf7;--panel:#fff;--ink:#1d1d1b;--mute:#6f6c66;--line:#e4e1da;--faint:#d9d6cf;
--hi:#f2efe8;--play:#c2410c;--warn:#b45309;--name:240px}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--panel:#1e1e1c;--ink:#ecebe7;--mute:#9b978f;
--line:#2f2e2b;--faint:#3a3935;--hi:#262523;--play:#fb923c;--warn:#f59e0b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);
padding:8px 12px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header h1{font-size:14px;margin:0 8px 0 0;font-weight:600}
button{font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;
padding:3px 9px;cursor:pointer}
button:hover{background:var(--hi)}
.clock{font-variant-numeric:tabular-nums;min-width:120px}
.mute{color:var(--mute)}
input[type=range]{width:140px}
#wrap{position:relative}
.block{border-bottom:1px solid var(--line);background:var(--panel)}
.block.active{background:var(--hi)}
.bhead{display:flex;gap:8px;align-items:baseline;padding:6px 12px 2px;cursor:pointer}
.bhead b{font-weight:600}
.flow{color:var(--mute);font-size:12px}
.tag{font-size:11px;border:1px solid var(--line);border-radius:10px;padding:0 6px;color:var(--mute)}
.tag.warn{color:var(--warn);border-color:var(--warn)}
.lane{display:flex;align-items:stretch}
.lname{width:var(--name);flex:none;padding:4px 12px;font-size:12px;color:var(--mute);display:flex;align-items:center}
.lname i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}
canvas{display:block;flex:1;min-width:0;cursor:crosshair}
#scroll{position:sticky;bottom:0;margin-left:var(--name);overflow-x:auto;overflow-y:hidden;height:16px;
background:var(--panel);border-top:1px solid var(--line)}
#spacer{height:1px}
#detail{position:sticky;bottom:16px;background:var(--panel);border-top:1px solid var(--line);padding:6px 12px;
min-height:30px;font-size:13px}
.missing{padding:4px 12px 8px;color:var(--warn)}
@media (max-width:700px){:root{--name:110px}.flow{display:none}}
</style></head><body>
<header>
  <h1>__TITLE__</h1>
  <button id="play">▶ Phát</button>
  <button id="back">−5 s</button><button id="fwd">+5 s</button>
  <span class="clock" id="clock">0:00.0</span>
  <label class="mute">Zoom <input type="range" id="zoom" min="0" max="100" value="30"></label>
  <span class="mute">Space: phát/dừng · ←/→: ±5 s · ↑/↓: đổi variant · bấm vào sóng/chữ: nhảy tới đó</span>
</header>
<div id="wrap"></div>
<div id="detail" class="mute">Di chuột lên một đoạn chữ để xem đầy đủ.</div>
<div id="scroll"><div id="spacer"></div></div>
<script>
const DATA = __DATA__;
const PPS_DATA = DATA.peaks_per_sec;
const COLORS = ["#2563eb","#dc2626","#059669","#7c3aed","#d97706","#0891b2","#db2777","#4d7c0f"];
const WAVE_H = 46, TEXT_H = 34;
const wrap = document.getElementById("wrap"), scroller = document.getElementById("scroll"),
      spacer = document.getElementById("spacer"), detail = document.getElementById("detail");
const css = getComputedStyle(document.documentElement);
const col = n => css.getPropertyValue(n).trim();

let duration = 0;
for (const v of DATA.variants) duration = Math.max(duration, v.duration || 0,
  ...(v.segments || []).map(s => s.e));
duration = Math.max(duration, 1);

let pxPerSec = 20, t0 = 0, now = 0, active = 0, playing = false;
const audios = [], lanes = [];

function fmt(t){t=Math.max(0,t);const m=Math.floor(t/60),s=t-m*60;return m+":"+(s<10?"0":"")+s.toFixed(1);}
function decode(b64){if(!b64)return null;const s=atob(b64),a=new Uint8Array(s.length);
  for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a;}

DATA.variants.forEach((v, vi) => {
  const block = document.createElement("div"); block.className = "block";
  const head = document.createElement("div"); head.className = "bhead";
  const speakers = [...new Set((v.segments || []).map(s => s.spk))].sort();
  const tags = [];
  if (v.baseline) tags.push('<span class="tag">baseline</span>');
  if (v.group) tags.push(`<span class="tag">${esc(v.group)}</span>`);
  if (v.status && v.status !== "ok") tags.push(`<span class="tag warn">${esc(v.status)}</span>`);
  if (v.diff_vs_base != null) tags.push(`<span class="tag" title="khác baseline ở mức từ; không có bản chuẩn nên chưa nói bên nào đúng">` +
    `khác baseline ${(v.diff_vs_base*100).toFixed(1)}%</span>`);
  head.innerHTML = `<b>${esc(v.id)}</b>${tags.join("")}<span class="mute">${speakers.length} speaker · ` +
    `${(v.segments||[]).length} đoạn${v.words != null ? " · " + v.words + " từ" : ""}</span><span class="flow">${esc(v.flow)}</span>`;
  head.onclick = () => select(vi, true);
  block.appendChild(head);
  if (v.missing) { const m = document.createElement("div"); m.className = "missing";
    m.textContent = v.missing; block.appendChild(m); }
  const pk = decode(v.peaks);
  speakers.forEach((spk, si) => {
    const color = COLORS[si % COLORS.length];
    const segs = v.segments.filter(s => s.spk === spk);
    for (const kind of ["wave", "text"]) {
      const lane = document.createElement("div"); lane.className = "lane";
      const name = document.createElement("div"); name.className = "lname";
      name.innerHTML = kind === "wave" ? `<i style="background:${color}"></i>${esc(spk)}` : "";
      const cv = document.createElement("canvas");
      cv.style.height = (kind === "wave" ? WAVE_H : TEXT_H) + "px";
      lane.append(name, cv); block.appendChild(lane);
      const L = {cv, kind, segs, pk, color, vi};
      lanes.push(L);
      cv.addEventListener("click", e => { const t = xToT(cv, e.offsetX); select(vi, false); seek(t); });
      cv.addEventListener("mousemove", e => hover(L, e.offsetX));
    }
  });
  wrap.appendChild(block);
  let a = null;
  if (v.audio) { a = new Audio(); a.preload = "none"; a.src = v.audio;
    a.addEventListener("ended", () => { playing = false; updateBtn(); }); }
  audios.push(a);
  v._block = block;
});

function esc(s){return String(s ?? "").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function viewW(){ const L = lanes[0]; return L ? L.cv.clientWidth : wrap.clientWidth; }
function xToT(cv, x){ return t0 + x / pxPerSec; }

function setZoom(){
  const z = +document.getElementById("zoom").value / 100;      // 0..1 -> fit .. 400 px/s
  const fit = Math.max(viewW(), 200) / duration;
  pxPerSec = fit * Math.pow(400 / fit, z);
  spacer.style.width = Math.ceil(duration * pxPerSec) + "px";
  scroller.scrollLeft = t0 * pxPerSec;
  draw();
}

function draw(){
  const w = viewW(), dpr = window.devicePixelRatio || 1;
  const faint = col("--faint"), ink = col("--ink"), mute = col("--mute"), line = col("--line"), play = col("--play");
  for (const L of lanes) {
    const cv = L.cv, h = cv.clientHeight;
    if (cv.width !== Math.round(w*dpr) || cv.height !== Math.round(h*dpr)) {
      cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr); }
    const g = cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h);
    const t1 = t0 + w / pxPerSec;
    if (L.kind === "wave") {
      g.fillStyle = line; g.fillRect(0, h/2, w, 0.5);
      if (L.pk) {
        const mid = h/2, amp = h/2 - 2;
        const inSeg = new Uint8Array(Math.ceil(w));
        for (const s of L.segs) { if (s.e < t0 || s.s > t1) continue;
          const a = Math.max(0, Math.floor((s.s - t0) * pxPerSec)), b = Math.min(w, Math.ceil((s.e - t0) * pxPerSec));
          for (let x = a; x < b; x++) inSeg[x] = 1; }
        for (let x = 0; x < w; x++) {
          const ta = t0 + x / pxPerSec, tb = ta + 1 / pxPerSec;
          let i0 = Math.floor(ta * PPS_DATA), i1 = Math.max(i0 + 1, Math.ceil(tb * PPS_DATA)), m = 0;
          if (i0 >= L.pk.length) break;
          for (let i = i0; i < i1 && i < L.pk.length; i++) if (L.pk[i] > m) m = L.pk[i];
          const y = Math.max(0.5, m / 255 * amp);
          g.fillStyle = inSeg[x] ? L.color : faint;
          g.fillRect(x, mid - y, 1, 2 * y);
        }
      }
    } else {
      g.font = "12px system-ui,sans-serif"; g.textBaseline = "middle";
      for (const s of L.segs) { if (s.e < t0 || s.s > t1) continue;
        const x = (s.s - t0) * pxPerSec, bw = Math.max(2, (s.e - s.s) * pxPerSec);
        g.fillStyle = L.color; g.globalAlpha = 0.12; g.fillRect(x, 2, bw, h - 4); g.globalAlpha = 1;
        g.fillRect(x, 2, 2, h - 4);
        if (bw > 14) { g.save(); g.beginPath(); g.rect(x + 4, 0, bw - 6, h); g.clip();
          g.fillStyle = ink; g.fillText(s.t, x + 5, h/2); g.restore(); }
      }
    }
    const px = (now - t0) * pxPerSec;
    if (px >= 0 && px <= w) { g.fillStyle = L.vi === active ? play : mute; g.fillRect(px, 0, 1.5, h); }
  }
  document.getElementById("clock").textContent = fmt(now) + " / " + fmt(duration);
}

function hover(L, x){
  const t = xToT(L.cv, x), s = L.segs.find(s => t >= s.s && t <= s.e);
  if (!s) return;
  const v = DATA.variants[L.vi];
  detail.className = "";
  detail.innerHTML = `<b>${esc(v.id)}</b> · ${esc(s.spk)} · ${fmt(s.s)}–${fmt(s.e)}: ${esc(s.t)}`;
}

function select(vi, keepTime){
  const was = playing; if (playing) pause();
  active = vi;
  DATA.variants.forEach((v, i) => v._block.classList.toggle("active", i === vi));
  if (was) play(); else draw();
}
function seek(t){
  now = Math.min(Math.max(0, t), duration);
  const a = audios[active]; if (a && playing) a.currentTime = now;
  follow(); draw();
}
function play(){
  const a = audios[active];
  if (!a) { detail.textContent = "Variant này không có audio để phát."; return; }
  audios.forEach(x => x && x !== a && x.pause());
  const start = () => { a.currentTime = now; a.play().then(() => { playing = true; updateBtn(); tick(); })
    .catch(e => { detail.textContent = "Không phát được audio: " + e.message +
      " (mở trang qua http server nếu trình duyệt chặn file cục bộ)"; }); };
  if (a.readyState >= 1) start(); else { a.preload = "auto"; a.addEventListener("loadedmetadata", start, {once:true}); a.load(); }
}
function pause(){ const a = audios[active]; if (a) a.pause(); playing = false; updateBtn(); }
function updateBtn(){ document.getElementById("play").textContent = playing ? "⏸ Dừng" : "▶ Phát"; }
function tick(){ if (!playing) return; const a = audios[active]; if (a) now = a.currentTime;
  follow(); draw(); requestAnimationFrame(tick); }
function follow(){ const w = viewW();
  if (now < t0 || now > t0 + w / pxPerSec * 0.95) { t0 = Math.max(0, now - w / pxPerSec * 0.1);
    scroller.scrollLeft = t0 * pxPerSec; } }

scroller.addEventListener("scroll", () => { t0 = scroller.scrollLeft / pxPerSec; draw(); });
wrap.addEventListener("wheel", e => { if (!e.ctrlKey && !e.metaKey && Math.abs(e.deltaX) < Math.abs(e.deltaY)) return;
  e.preventDefault(); if (e.ctrlKey || e.metaKey) { const z = document.getElementById("zoom");
    z.value = Math.min(100, Math.max(0, +z.value - Math.sign(e.deltaY) * 4)); setZoom(); }
  else scroller.scrollLeft += e.deltaX; }, {passive:false});
document.getElementById("zoom").addEventListener("input", setZoom);
document.getElementById("play").onclick = () => playing ? pause() : play();
document.getElementById("back").onclick = () => seek(now - 5);
document.getElementById("fwd").onclick = () => seek(now + 5);
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); playing ? pause() : play(); }
  else if (e.key === "ArrowLeft") seek(now - 5);
  else if (e.key === "ArrowRight") seek(now + 5);
  else if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault();
    const n = DATA.variants.length; select((active + (e.key === "ArrowDown" ? 1 : n - 1)) % n, true);
    DATA.variants[active]._block.scrollIntoView({block: "nearest"}); }
});
window.addEventListener("resize", setZoom);
select(0, true); setZoom();
</script></body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", required=True, help="variants_manifest.json written by run_variants.py")
    ap.add_argument("--out-dir", help="where the pages go (default: <manifest dir>/lanes)")
    ap.add_argument("--files", help="comma-separated file stems (default: every file in the manifest)")
    ap.add_argument("--transcript", choices=sorted(TRANSCRIPTS), default="asr",
                    help="asr = 05_asr (default); refinement = 06_refinement; latest = the last stage that wrote one")
    ap.add_argument("--peaks-per-sec", type=int, default=50, help="waveform resolution stored in the page")
    ap.add_argument("--audio", choices=["link", "none"], default="link",
                    help="link = relative path to each variant's wav; none = waveform and text only")
    a = ap.parse_args(argv)
    manifest = os.path.abspath(a.manifest)
    out_dir = os.path.abspath(a.out_dir or os.path.join(os.path.dirname(manifest), "lanes"))
    os.makedirs(out_dir, exist_ok=True)
    stems = [s for s in (a.files.split(",") if a.files else stems_of(manifest)) if s]
    if not stems:
        print("no files in the manifest", file=sys.stderr)
        return 1
    for stem in stems:
        data = build_page_data(manifest, stem, out_dir, a.transcript, a.peaks_per_sec, a.audio)
        path = os.path.join(out_dir, f"{stem}_lanes.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render(data))
        ok = sum(1 for v in data["variants"] if not v.get("missing"))
        print(f"{path}  ({ok}/{len(data['variants'])} variants with data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
