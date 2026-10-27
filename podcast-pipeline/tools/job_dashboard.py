#!/usr/bin/env python3
"""A web page that shows how far the three jobs are.  Read-only: it only looks at OUT, so it can start (and stop)
at any time without touching the running processes.

    python tools/job_dashboard.py OUT [--final FINAL] [--port 8765] [--host 127.0.0.1]
    ssh -L 8765:localhost:8765 <dgx>          # from the laptop, then http://localhost:8765/
    python tools/job_dashboard.py OUT --once  # print the numbers as JSON and exit

Per stage (denoise, diarization, separation_plan, separation, asr, conversation_selection) it counts the files
that are done (.ready_<stage>.<stem>), running (.claim_ with no ready/failed yet) and failed, with files per
minute over the last 10 minutes and the time left at that speed. It also lists the failures with their reason,
the last line and age of every process log, the GPUs of the host the page runs on and the final dataset
(<FINAL>/lhotse). Standard library only.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import line_status as ls  # noqa: E402

STAGES = ("denoise", "diarization", "separation_plan", "separation", "asr", "conversation_selection")
RATE_WINDOW = 600


def _mtime(p):
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def variant_dirs(out):
    skip = {"final", "lanes"}
    return [out / v for v in ls.variant_ids(out) if v not in skip and (out / v).is_dir()]


def stage_counts(out, stems, now):
    """{stage: {total, done, failed, running, rate, eta_min}} summed over the variants of the run."""
    result = {}
    vdirs = variant_dirs(out)
    for stage in STAGES:
        done_times, failed, running = [], 0, 0
        for vdir in vdirs:
            for stem in stems:
                ready = vdir / f".ready_{stage}.{stem}"
                bad = vdir / f".failed_{stage}.{stem}"
                if ready.is_file():
                    done_times.append(_mtime(ready))
                elif bad.is_file():
                    failed += 1
                elif (vdir / f".claim_{stage}.{stem}").is_file():
                    running += 1
        total = len(stems) * max(1, len(vdirs))
        recent = sum(1 for t in done_times if now - t <= RATE_WINDOW)
        rate = recent / (RATE_WINDOW / 60)
        left = total - len(done_times) - failed
        result[stage] = {"total": total, "done": len(done_times), "failed": failed, "running": running,
                         "rate": round(rate, 2),
                         "eta_min": round(left / rate) if rate > 0 and left > 0 else None,
                         "first": min(done_times) if done_times else None,
                         "last": max(done_times) if done_times else None}
    return result


def failures(out, stems):
    rows = []
    for vdir in variant_dirs(out):
        for stage in STAGES:
            for stem in stems:
                bad = vdir / f".failed_{stage}.{stem}"
                if bad.is_file():
                    rows.append({"stage": stage, "file": stem, "reason": ls._reason(bad),
                                 "age": ls.age(bad)})
    return rows[:200]


def processes(out):
    rows = []
    logs = out / "_resident" / "logs"
    pids = {}
    for f in logs.glob("*.pids"):
        for line in f.read_text(errors="replace").splitlines():
            pid, _, name = line.partition(" ")
            if pid.isdigit():
                pids[name.strip()] = (f.stem, int(pid))
    for name, when, text in ls.last_lines(out):
        job_pid = next((v for k, v in pids.items() if name.endswith(k)), None)
        alive = None
        if job_pid and Path(f"/proc/{job_pid[1]}").exists():
            alive = True
        elif job_pid and sys.platform.startswith("linux"):
            alive = False   # only meaningful on the host that started it; the page says so
        age_s = time.time() - _mtime(logs / f"{name}.log")
        rows.append({"name": name, "job": job_pid[0] if job_pid else "", "alive": alive,
                     "age": when, "age_s": round(age_s), "line": text})
    return rows


def gpus():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in r.stdout.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 5:
            rows.append({"index": p[0], "name": p[1], "used": int(p[2]), "total": int(p[3]), "util": int(p[4])})
    return rows


def final_state(final):
    manifest = final / "lhotse" / "manifest.jsonl.gz"
    parts = list((final / "lhotse" / ".parts").glob("*.jsonl.gz")) if (final / "lhotse").is_dir() else []
    exported = {k: len(list((final / k).glob("*/"))) for k in ("interrupt", "backchannel", "turn_by_turn")
                if (final / k).is_dir()}
    return {"manifest": manifest.is_file(), "manifest_mb": round(manifest.stat().st_size / 1e6, 2)
            if manifest.is_file() else 0, "manifest_age": ls.age(manifest) if manifest.is_file() else "",
            "recordings": len(parts), "conversations": exported}


def collect(out, final):
    now = time.time()
    stems = ls.file_stems(out)
    disk = shutil.disk_usage(out)
    return {"now": now, "host": os.uname().nodename, "out": str(out), "files": len(stems),
            "stages": stage_counts(out, stems, now), "failures": failures(out, stems),
            "claims": [{"stage": st, "file": stem, "who": who}
                       for _v, st, stem, who in ls.claims(out) if who][:100],
            "processes": processes(out), "gpus": gpus(), "final": final_state(final),
            "disk": {"free_gb": round(disk.free / 1e9), "total_gb": round(disk.total / 1e9)},
            "warning": ls.frozen_warning(out, ls._resident()) or ""}


PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Job dashboard</title><style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2330;--mute:#667085;--line:#e3e6ec;--ok:#1a7f4b;--run:#2563eb;--bad:#c0392b;--warn:#b7791f}
@media(prefers-color-scheme:dark){:root{--bg:#10141b;--card:#171c25;--ink:#e6e9ef;--mute:#8b94a5;--line:#273041;--ok:#4cc38a;--run:#6ea0ff;--bad:#ff7a6b;--warn:#e0b24d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:18px;margin:0 0 4px}h2{font-size:14px;margin:0 0 10px}
.sub{color:var(--mute);margin-bottom:14px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.row{display:grid;grid-template-columns:150px 1fr 170px;gap:12px;align-items:center;padding:6px 0}
.bar{height:12px;background:var(--line);border-radius:6px;overflow:hidden;display:flex}.bar i{display:block;height:100%}
.d{background:var(--ok)}.r{background:var(--run)}.f{background:var(--bad)}.num{font-variant-numeric:tabular-nums;text-align:right}
.mute{color:var(--mute)}table{width:100%;border-collapse:collapse}td,th{padding:5px 6px;border-top:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--mute);font-weight:500;border-top:0}.bad{color:var(--bad)}.ok{color:var(--ok)}.warn{color:var(--warn)}
.log{font:12px ui-monospace,monospace;word-break:break-all}.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}
@media(max-width:640px){.row{grid-template-columns:1fr}.num{text-align:left}}
</style></head><body><main>
<h1>Tiến độ job</h1><div class="sub" id="sub">đang tải…</div><div id="warn" class="warn"></div>
<div class="card"><h2>Các stage</h2><div id="stages"></div></div>
<div class="card"><h2>Kết quả cuối</h2><div id="final"></div></div>
<div class="card"><h2>GPU của máy đang chạy trang này</h2><div class="g" id="gpus"></div></div>
<div class="card"><h2>Lỗi</h2><div id="fail"></div></div>
<div class="card"><h2>Tiến trình (dòng log cuối)</h2><div id="proc"></div></div>
<div class="card"><h2>Đang giữ file</h2><div id="claims"></div></div>
</main><script>
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const fmt=m=>m==null?'—':m<90?m+' phút':(m/60).toFixed(1)+' giờ';
async function tick(){
 let d;try{d=await (await fetch('api',{cache:'no-store'})).json()}catch(e){$('sub').textContent='mất kết nối, thử lại…';return}
 $('sub').textContent=`${d.out} · ${d.files} file · máy ${d.host} · ổ trống ${d.disk.free_gb}/${d.disk.total_gb} GB · cập nhật ${new Date(d.now*1000).toLocaleTimeString()}`;
 $('warn').textContent=d.warning;
 $('stages').innerHTML=Object.entries(d.stages).map(([k,s])=>{const t=s.total||1,p=x=>(100*x/t).toFixed(1);
  return `<div class="row"><b>${k}</b><div class="bar" title="xong ${s.done}, chạy ${s.running}, lỗi ${s.failed}"><i class="d" style="width:${p(s.done)}%"></i><i class="r" style="width:${p(s.running)}%"></i><i class="f" style="width:${p(s.failed)}%"></i></div>
  <div class="num">${s.done}/${s.total}${s.running?` · <span style="color:var(--run)">${s.running} chạy</span>`:''}${s.failed?` · <span class="bad">${s.failed} lỗi</span>`:''}<br><span class="mute">${s.rate}/phút · còn ${fmt(s.eta_min)}</span></div></div>`}).join('');
 const f=d.final;$('final').innerHTML=(f.manifest?`<span class="ok">manifest.jsonl.gz</span> ${f.manifest_mb} MB (cập nhật ${f.manifest_age} trước)`:'<span class="mute">chưa có manifest.jsonl.gz</span>')+` · ${f.recordings} bản ghi đã vào Lhotse`+Object.entries(f.conversations).map(([k,v])=>` · ${k}: ${v}`).join('');
 $('gpus').innerHTML=d.gpus.length?d.gpus.map(g=>`<div><b>GPU ${g.index}</b> <span class="mute">${esc(g.name)}</span><div class="bar"><i class="r" style="width:${100*g.used/g.total}%"></i></div>${(g.used/1024).toFixed(1)}/${(g.total/1024).toFixed(0)} GiB · tải ${g.util}%</div>`).join(''):'<span class="mute">không có nvidia-smi trên máy này</span>';
 $('fail').innerHTML=d.failures.length?`<table><tr><th>stage</th><th>file</th><th>lý do</th><th>cách đây</th></tr>${d.failures.map(r=>`<tr><td>${r.stage}</td><td>${esc(r.file)}</td><td class="bad">${esc(r.reason)}</td><td>${r.age}</td></tr>`).join('')}</table>`:'<span class="ok">không có lỗi</span>';
 $('proc').innerHTML=`<table>${d.processes.map(p=>`<tr><td>${p.name}</td><td class="${p.age_s>600?'warn':'mute'}">${p.age}</td><td class="log">${esc(p.line)}</td></tr>`).join('')}</table>`;
 $('claims').innerHTML=d.claims.length?`<table>${d.claims.map(c=>`<tr><td>${c.stage}</td><td>${esc(c.file)}</td><td class="mute">${esc(c.who)}</td></tr>`).join('')}</table>`:'<span class="mute">không có</span>';
}
tick();setInterval(tick,5000);
</script></body></html>"""


def make_handler(out, final):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path in ("", "/index.html"):
                body, kind = PAGE.encode(), "text/html; charset=utf-8"
            elif path == "/api":
                body, kind = json.dumps(collect(out, final)).encode(), "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--final", type=Path, default=None, help="default OUT/final")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open it to the network")
    ap.add_argument("--once", action="store_true", help="print the numbers as JSON and exit")
    args = ap.parse_args(argv)
    if not args.out.is_dir():
        sys.exit(f"not a directory: {args.out}")
    final = args.final or args.out / "final"
    if args.once:
        print(json.dumps(collect(args.out, final), indent=2, ensure_ascii=False))
        return 0
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.out, final))
    print(f"http://{args.host}:{args.port}/   (Ctrl+C to stop; the jobs are not touched)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
