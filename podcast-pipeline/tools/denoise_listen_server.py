#!/usr/bin/env python3
"""Live listening table for the denoise variants, with a notes column.

Scans OUT/<variant>/<stem>/01b_denoise/after_denoise.wav for every variant, serves them on a
port as one page (an audio player per row, the unfiltered audio on top as the reference) and
keeps your notes in OUT/denoise_notes.json. The list refreshes by itself, so variants that
finish while it is running show up without a reload. Standard library only.

    python tools/denoise_listen_server.py --out-root /home/lamkd2/data/denoise_all --port 8765
    # on your laptop:  ssh -L 8765:localhost:8765 <dgx>   then open http://localhost:8765

Rows can be hidden with a checkbox; the choice is kept in OUT/denoise_hidden.json and can be undone
from the page. --hide-initial v1,v2 hides those variants the first time (while that file is absent).

It binds to 127.0.0.1 by default. There is no login: pass --host 0.0.0.0 only on a network you
trust, since anyone who can reach the port can read the audio and write notes.
"""
import argparse
import hashlib
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

KINDS = {"denoise": os.path.join("01b_denoise", "after_denoise.wav"),
         "music": os.path.join("01_music", "after_music.wav")}
SAFE = re.compile(r"^[^/\\]+$")
MAX_NOTE = 20000
CHUNK = 1 << 16

PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Denoise nghe thử</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--text:#1d2530;--muted:#667085;--line:#dfe3ea;--accent:#2563eb;--warn:#b45309;--warnbg:#fef3c7;--ok:#15803d}
@media (prefers-color-scheme:dark){:root{--bg:#12161c;--card:#1a2029;--text:#e6e9ee;--muted:#94a0b2;--line:#2b3441;--accent:#6ea0ff;--warn:#f5b45a;--warnbg:#3a2c12;--ok:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:3;background:var(--bg);padding:12px 16px;border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0 0 8px}.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input[type=search]{padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--text);min-width:220px}
label{color:var(--muted)}#state{color:var(--muted);margin-left:auto}
main{padding:12px 16px 40px;overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:760px;background:var(--card);border:1px solid var(--line)}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
th{background:var(--card);color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
tr.ref td{background:color-mix(in srgb,var(--accent) 8%,var(--card))}
td.v{font-weight:600;word-break:break-all;min-width:200px}td.v small{display:block;color:var(--muted);font-weight:400}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;background:var(--warnbg);color:var(--warn);margin-left:6px}
audio{width:320px;max-width:100%}
.seek{display:flex;gap:6px;margin-top:4px}.seek button{padding:2px 9px;font-size:12px}
textarea{width:100%;min-width:240px;min-height:54px;resize:vertical;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--text);font:inherit}
tr.dim{opacity:.45}td.h{text-align:center;width:52px}
.bar input[type=text]{padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--text);width:150px}
textarea:focus{outline:2px solid var(--accent)}.saved{font-size:11px;color:var(--ok);height:14px}
button{padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--text);cursor:pointer}
@media (max-width:640px){header,main{padding-left:12px;padding-right:12px}}
</style></head><body>
<header><h1>Denoise: nghe thử và ghi chú</h1>
<div class="bar"><input id="q" type="search" placeholder="Lọc theo tên variant..." aria-label="Lọc">
<label title="Một bộ lọc duy nhất (blend chỉ là tỉ lệ trộn, không tính là bộ lọc)"><input id="tDon" type="checkbox" checked> Đơn <span id="nDon"></span></label>
<label title="Hai bộ lọc trở lên nối tiếp"><input id="tKet" type="checkbox" checked> Kết hợp <span id="nKet"></span></label>
<label><input id="onlyNotes" type="checkbox"> chỉ hàng có ghi chú</label>
<label><input id="hideDeg" type="checkbox"> ẩn variant lỗi (DEGRADED)</label>
<label title="Hiện lại các hàng đã tích Ẩn để xem hoặc bỏ ẩn"><input id="showHid" type="checkbox"> hiện cả hàng đã ẩn <span id="nHid"></span></label>
<input id="prefix" type="text" placeholder="tên bắt đầu bằng..." aria-label="Ẩn theo tiền tố"><button id="hidePrefix" title="Ẩn mọi variant có tên bắt đầu bằng chuỗi này">Ẩn theo tên</button>
<button id="unhideAll">Bỏ ẩn tất cả</button>
<span title="Tua hàng đang nghe; phím ← và → cũng tua 5 giây, phím cách để phát/dừng"><button id="gBack">◀ 5s</button> <button id="gFwd">5s ▶</button></span>
<label title="Khi bấm phát một hàng khác, bắt đầu đúng vị trí của hàng vừa nghe để so sánh cùng một đoạn"><input id="syncPos" type="checkbox" checked> giữ cùng vị trí khi đổi hàng</label> <span id="pos"></span>
<button id="csv">Xuất ghi chú CSV</button><span id="state">đang tải...</span></div></header>
<main><table><thead><tr><th>#</th><th>Variant</th><th>Âm thanh sau lọc</th><th>Ghi chú</th><th title="Tích để ẩn hàng này (có thể bỏ ẩn sau)">Ẩn</th></tr></thead>
<tbody id="rows"></tbody></table></main>
<script>
const $=s=>document.querySelector(s);let items=[],notes={},timers={},hidden={},active=null,lastPos=0;
function badge(it){return it.status&&it.status!=='ok'?`<span class="badge">${it.status.toUpperCase()}</span>`:''}
async function api(path,opt){const r=await fetch(path,opt);if(!r.ok)throw new Error(r.status);return r.json()}
function mk(it,i){const tr=document.createElement('tr');if(it.ref)tr.className='ref';tr.dataset.key=it.key;
 const kind=it.has_denoise?'denoise':'music';
 tr.innerHTML=`<td></td><td class="v"></td><td><audio controls preload="none"></audio><div class="seek"><button type="button" class="sk" data-d="-5">−5s</button><button type="button" class="sk" data-d="5">+5s</button></div></td><td><textarea></textarea><div class="saved"></div></td><td class="h"></td>`;
 if(!it.ref){const cb=document.createElement('input');cb.type='checkbox';cb.className='hid';cb.title='Ẩn hàng này';cb.checked=!!hidden[it.key];
  cb.addEventListener('change',()=>setHidden([it.key],cb.checked));tr.children[4].appendChild(cb)}
 tr.children[0].textContent=i+1;const v=tr.children[1];v.textContent=it.label||it.variant;
 v.insertAdjacentHTML('beforeend',badge(it)+`<small></small>`);
 v.querySelector('small').textContent=(it.ref?'Âm thanh gốc (chưa lọc) · ':(it.type==='ket_hop'?'kết hợp · ':'đơn · '))+it.stem+' · '+(it.size/1048576).toFixed(1)+' MB · mã '+(it.fp||'?');
 const a=tr.querySelector('audio');a.dataset.fp=it.fp||'';a.src=`/audio/${encodeURIComponent(it.variant)}/${encodeURIComponent(it.stem)}/${kind}`;
 a.addEventListener('play',()=>{if($('#syncPos').checked&&active&&active!==a&&Math.abs(a.currentTime-lastPos)>0.3)a.currentTime=lastPos;active=a;
  document.querySelectorAll('audio').forEach(o=>{if(o!==a)o.pause()});showPos()});
 a.addEventListener('timeupdate',()=>{if(a===active){lastPos=a.currentTime;showPos()}});
 tr.querySelectorAll('.sk').forEach(b=>b.addEventListener('click',()=>seek(a,Number(b.dataset.d))));
 const t=tr.querySelector('textarea');t.value=(notes[it.key]||{}).note||'';
 t.addEventListener('input',()=>{const s=tr.querySelector('.saved');s.textContent='đang lưu...';clearTimeout(timers[it.key]);
  timers[it.key]=setTimeout(async()=>{try{await api('/api/notes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:it.key,note:t.value})});
  notes[it.key]={note:t.value};s.textContent='đã lưu';apply()}catch(e){s.textContent='LỖI lưu: '+e.message}},600)});
 return tr}
function seek(a,d){if(!a)return;const dur=isFinite(a.duration)?a.duration:Infinity;a.currentTime=Math.max(0,Math.min(dur,a.currentTime+d));active=a;lastPos=a.currentTime;showPos()}
function showPos(){const el=$('#pos');if(!el)return;const tr=active&&active.closest('tr');el.textContent=active?((tr&&tr.children[1].childNodes[0].textContent)||'')+' · mã '+(active.dataset.fp||'?')+' · '+active.currentTime.toFixed(1)+'s':''}
function target(){return active||[...document.querySelectorAll('#rows tr')].filter(t=>t.style.display!=='none').map(t=>t.querySelector('audio'))[0]}
async function setHidden(keys,flag){keys.forEach(k=>{if(flag)hidden[k]=true;else delete hidden[k]});apply();
 try{await api('/api/hidden',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keys,hidden:flag})})}catch(e){$('#state').textContent='LỖI lưu trạng thái ẩn: '+e.message}}
function syncHidden(){document.querySelectorAll('#rows tr').forEach(tr=>{const cb=tr.querySelector('input.hid');if(cb)cb.checked=!!hidden[tr.dataset.key]})}
function apply(){syncHidden();const showH=$('#showHid').checked;$('#nHid').textContent='('+items.filter(x=>!x.ref&&hidden[x.key]).length+')';const q=$('#q').value.toLowerCase(),on=$('#onlyNotes').checked,hd=$('#hideDeg').checked,don=$('#tDon').checked,ket=$('#tKet').checked;
 $('#nDon').textContent='('+items.filter(x=>x.type==='don').length+')';$('#nKet').textContent='('+items.filter(x=>x.type==='ket_hop').length+')';
 document.querySelectorAll('#rows tr').forEach(tr=>{const it=items.find(x=>x.key===tr.dataset.key)||{};
 const has=((notes[it.key]||{}).note||'').trim()!=='';
 const typeOk=it.ref||(it.type==='ket_hop'?ket:don);
 tr.style.display=(it.variant||'').toLowerCase().includes(q)&&(!on||has)&&(!hd||!it.status||it.status==='ok'||it.ref)&&typeOk&&(!hidden[it.key]||showH||it.ref)?'':'none';
 tr.classList.toggle('dim',!!hidden[it.key]&&showH)})}
async function refresh(){try{const [list,n,hd2]=await Promise.all([api('/api/list'),api('/api/notes'),api('/api/hidden')]);notes=n;hidden=hd2;
 const sig=JSON.stringify(list.items.map(x=>[x.key,x.size,x.status,x.type]));
 if(sig!==refresh.sig){refresh.sig=sig;items=list.items;const body=$('#rows');
  const keep={};body.querySelectorAll('tr').forEach(tr=>{keep[tr.dataset.key]=tr.querySelector('textarea').value});
  body.innerHTML='';items.forEach((it,i)=>{const tr=mk(it,i);if(keep[it.key]!==undefined)tr.querySelector('textarea').value=keep[it.key];body.appendChild(tr)});}
 else document.querySelectorAll('#rows tr').forEach(tr=>{const t=tr.querySelector('textarea');
  if(document.activeElement!==t){const v=(notes[tr.dataset.key]||{}).note||'';if(t.value!==v)t.value=v}});
 apply();$('#state').textContent=`${items.length} hàng · cập nhật ${new Date().toLocaleTimeString()}`}
 catch(e){$('#state').textContent='mất kết nối: '+e.message}}
['q','onlyNotes','hideDeg','tDon','tKet','showHid'].forEach(id=>$('#'+id).addEventListener('input',apply));
$('#hidePrefix').addEventListener('click',()=>{const p=$('#prefix').value.trim();if(!p)return;
 const keys=items.filter(x=>!x.ref&&x.variant.startsWith(p)&&!hidden[x.key]).map(x=>x.key);if(keys.length)setHidden(keys,true);$('#prefix').value=''});
$('#unhideAll').addEventListener('click',()=>{const keys=Object.keys(hidden);if(keys.length)setHidden(keys,false)});
$('#gBack').addEventListener('click',()=>seek(target(),-5));$('#gFwd').addEventListener('click',()=>seek(target(),5));
document.addEventListener('keydown',e=>{if(/^(TEXTAREA|INPUT|SELECT|BUTTON)$/.test((e.target||{}).tagName||'')||e.ctrlKey||e.metaKey||e.altKey)return;
 if(e.key==='ArrowLeft'){e.preventDefault();seek(target(),-5)}else if(e.key==='ArrowRight'){e.preventDefault();seek(target(),5)}
 else if(e.key===' '&&active){e.preventDefault();active.paused?active.play():active.pause()}});
$('#csv').addEventListener('click',()=>{const q=s=>'"'+String(s).replace(/"/g,'""')+'"';
 const rows=[['variant','stem','note']].concat(items.map(it=>[it.variant,it.stem,(notes[it.key]||{}).note||'']).filter(r=>r[2].trim()));
 const b=new Blob([rows.map(r=>r.map(q).join(',')).join('\n')],{type:'text/csv'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='denoise_notes.csv';a.click()});
refresh();setInterval(refresh,10000);
</script></body></html>"""


def row_status(row):
    if not row:
        return None
    return row.get("status") or ("degraded" if row.get("degraded") else None)


def variant_type(variant_id, row=None):
    """"don" for one filter, "ket_hop" for two or more in a chain, "goc" for none.

    The chain comes from the manifest when it has one, otherwise from the id up to its first
    dot (what follows the dot is a pipeline setting such as music-removal-off, not a filter).
    A blend ratio (blend50) mixes the filtered audio back with the input, so it is not a filter.
    """
    chain = (row or {}).get("chain")
    if not isinstance(chain, str) or not chain:
        chain = variant_id.split(".", 1)[0]
    steps = [t for t in re.split(r"[>+]", chain) if t and t != "none" and not re.match(r"^blend\d*$", t)]
    return "goc" if not steps else ("don" if len(steps) == 1 else "ket_hop")


class State:
    def __init__(self, out_root):
        self.root = os.path.abspath(out_root)
        self.notes_path = os.path.join(self.root, "denoise_notes.json")
        self.hidden_path = os.path.join(self.root, "denoise_hidden.json")
        self._fp = {}
        self.lock = threading.Lock()

    def fingerprint(self, path):
        """Short id of a file's content: size plus the start, middle and end 64 KB (cached by size and mtime).

        Two rows with the same value hold the same audio; it is shown on the page so a mix-up is visible.
        """
        try:
            st = os.stat(path)
        except OSError:
            return ""
        key = (path, st.st_size, st.st_mtime_ns)
        if key not in self._fp:
            h = hashlib.sha1(str(st.st_size).encode())
            with open(path, "rb") as f:
                for off in (0, max(0, st.st_size // 2 - 32768), max(0, st.st_size - 65536)):
                    f.seek(off)
                    h.update(f.read(65536))
            self._fp[key] = h.hexdigest()[:8]
        return self._fp[key]

    def manifest_rows(self):
        try:
            with open(os.path.join(self.root, "variants_manifest.json"), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        rows = data.get("variants") if isinstance(data, dict) else data
        return {r["id"]: r for r in rows if isinstance(r, dict) and r.get("id")} if isinstance(rows, list) else {}

    def scan(self):
        rows = self.manifest_rows()
        items, ref = [], None
        try:
            variants = sorted(d for d in os.listdir(self.root)
                              if SAFE.match(d) and not d.startswith(".") and d != "logs"
                              and os.path.isdir(os.path.join(self.root, d)))
        except OSError:
            return []
        for v in variants:
            vdir = os.path.join(self.root, v)
            for stem in sorted(os.listdir(vdir)):
                sdir = os.path.join(vdir, stem)
                if stem == "input" or not os.path.isdir(sdir):
                    continue
                dn = os.path.join(sdir, KINDS["denoise"])
                mu = os.path.join(sdir, KINDS["music"])
                if os.path.isfile(dn):
                    items.append({"key": f"{v}/{stem}", "variant": v, "stem": stem, "size": os.path.getsize(dn), "fp": self.fingerprint(dn), "label": v,
                                  "status": row_status(rows.get(v)), "type": variant_type(v, rows.get(v)),
                                  "has_denoise": True, "ref": False})
                elif os.path.isfile(mu) and (v == "none" or ref is None):
                    ref = {"key": f"{v}/{stem}", "variant": v, "label": f"{v} (gốc)", "stem": stem,
                           "size": os.path.getsize(mu), "fp": self.fingerprint(mu), "status": None, "type": "goc", "has_denoise": False, "ref": True}
        if ref:
            items.insert(0, ref)
        return items

    def resolve(self, variant, stem, kind):
        if kind not in KINDS or not (SAFE.match(variant) and SAFE.match(stem)) or ".." in (variant, stem):
            return None
        path = os.path.realpath(os.path.join(self.root, variant, stem, KINDS[kind]))
        if not path.startswith(self.root + os.sep) or not os.path.isfile(path):
            return None
        return path

    def load_notes(self):
        try:
            with open(self.notes_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def load_hidden(self):
        try:
            with open(self.hidden_path, encoding="utf-8") as f:
                data = json.load(f)
            return {k: True for k, v in data.items() if v} if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def set_hidden(self, keys, hidden):
        with self.lock:
            cur = self.load_hidden()
            for k in keys:
                if hidden:
                    cur[k] = True
                else:
                    cur.pop(k, None)
            tmp = f"{self.hidden_path}.tmp{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cur, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.hidden_path)

    def seed_hidden(self, variants):
        """Hide these variants once, only if nothing has been hidden or unhidden yet."""
        if variants and not os.path.exists(self.hidden_path):
            want = set(variants)
            self.set_hidden([it["key"] for it in self.scan() if it["variant"] in want and not it["ref"]], True)

    def save_note(self, key, note):
        with self.lock:
            notes = self.load_notes()
            if note.strip():
                notes[key] = {"note": note, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
            else:
                notes.pop(key, None)
            tmp = f"{self.notes_path}.tmp{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(notes, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.notes_path)


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        server_version = "denoise-listen"

        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/list":
                self._json({"items": state.scan()})
            elif path == "/api/notes":
                self._json(state.load_notes())
            elif path == "/api/hidden":
                self._json(state.load_hidden())
            elif path.startswith("/audio/"):
                parts = [unquote(p) for p in path[len("/audio/"):].split("/")]
                real = state.resolve(*parts) if len(parts) == 3 else None
                if real is None:
                    self.send_error(404)
                    return
                self._send_file(real)
            else:
                self.send_error(404)

        def do_POST(self):
            route = urlparse(self.path).path
            if route == "/api/hidden":
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                    data = json.loads(self.rfile.read(min(n, 1 << 20)))
                    keys, flag = [str(k) for k in data["keys"]], bool(data["hidden"])
                except (ValueError, KeyError, TypeError):
                    self.send_error(400)
                    return
                known = {it["key"] for it in state.scan() if not it["ref"]}
                state.set_hidden([k for k in keys if k in known], flag)
                self._json({"ok": True})
                return
            if route != "/api/notes":
                self.send_error(404)
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(min(n, 4 * MAX_NOTE)))
                key, note = str(data["key"]), str(data["note"])
            except (ValueError, KeyError, TypeError):
                self.send_error(400)
                return
            known = {it["key"] for it in state.scan()}
            if key not in known or len(note) > MAX_NOTE:
                self.send_error(400)
                return
            state.save_note(key, note)
            self._json({"ok": True})

        def _send_file(self, real):
            size = os.path.getsize(real)
            start, end, code = 0, size - 1, 200
            m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", "").strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                code = 206
            self.send_response(code)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with open(real, "rb") as f:
                    f.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = f.read(min(CHUNK, left))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--hide-initial", default="", help="comma-separated variant ids to hide the first time")
    a = ap.parse_args()
    state = State(a.out_root)
    if not os.path.isdir(state.root):
        raise SystemExit(f"no such folder: {state.root}")
    state.seed_hidden([v.strip() for v in a.hide_initial.split(",") if v.strip()])
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(state))
    print(f"serving {state.root} on http://{a.host}:{a.port}  (notes: {state.notes_path})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
