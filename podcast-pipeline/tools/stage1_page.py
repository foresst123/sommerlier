"""The HTML the matrix is read in. Data is injected, not fetched."""
import json
import os
import sys

TEMPLATE = r"""<title>Ma trận lọc nhiễu bước 1</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#F5F7F9; --surface:#FFFFFF; --sunk:#EDF1F4;
  --ink:#10151A; --muted:#5D6976; --rule:#DFE5EB;
  --accent:#B85A0C; --accent-soft:#F6E3D0;
  --keep:#1C6B47; --keep-soft:#D8EDE3;
  --drop:#9E3527; --drop-soft:#F6DCD7;
  --warn:#8A6A16; --warn-soft:#F7EBCF;
  --shadow:0 1px 2px rgba(16,21,26,.05),0 8px 24px -12px rgba(16,21,26,.16);
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --ground:#0C1014; --surface:#141A20; --sunk:#1B232B;
  --ink:#E4EAF0; --muted:#8695A4; --rule:#232D37;
  --accent:#E89446; --accent-soft:#3A2A18;
  --keep:#54B98A; --keep-soft:#17301F;
  --drop:#E37C68; --drop-soft:#331C1A;
  --warn:#D8B25C; --warn-soft:#2E2716;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -14px rgba(0,0,0,.7);
}}
:root[data-theme="dark"]{
  --ground:#0C1014; --surface:#141A20; --sunk:#1B232B;
  --ink:#E4EAF0; --muted:#8695A4; --rule:#232D37;
  --accent:#E89446; --accent-soft:#3A2A18;
  --keep:#54B98A; --keep-soft:#17301F;
  --drop:#E37C68; --drop-soft:#331C1A;
  --warn:#D8B25C; --warn-soft:#2E2716;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -14px rgba(0,0,0,.7);
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  font-size:15px;line-height:1.6;margin:0;padding:0 20px 96px}
.wrap{max-width:1060px;margin:0 auto}
h1,h2,h3{font-family:Archivo,system-ui,sans-serif;text-wrap:balance;margin:0}
h1{font-size:clamp(26px,4vw,38px);font-weight:700;letter-spacing:-.02em;line-height:1.15}
h2{font-size:22px;font-weight:600;letter-spacing:-.01em}
h3{font-size:16px;font-weight:600}
p{margin:0}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.9em}
.num{font-variant-numeric:tabular-nums}
header{padding:56px 0 32px;border-bottom:1px solid var(--rule);margin-bottom:40px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent);font-weight:600;margin-bottom:14px}
.lede{color:var(--muted);max-width:64ch;margin-top:16px;font-size:16px}
section{margin:0 0 52px}
.shead{display:flex;align-items:baseline;gap:14px;margin-bottom:6px}
.shead .n{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted);font-weight:500}
.sdesc{color:var(--muted);max-width:66ch;margin:0 0 22px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:10px;
  padding:20px 22px;box-shadow:var(--shadow)}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;font-family:"IBM Plex Mono",monospace;font-size:11px;font-weight:600;
  letter-spacing:.08em;text-transform:uppercase;color:var(--muted);
  padding:0 12px 10px 0;border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:11px 12px 11px 0;border-bottom:1px solid var(--rule);vertical-align:top}
tr:last-child td{border-bottom:none}
.scroll{overflow-x:auto}
.pill{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:11px;
  font-weight:600;padding:2px 8px;border-radius:20px;white-space:nowrap}
.pill.ok{background:var(--keep-soft);color:var(--keep)}
.pill.no{background:var(--drop-soft);color:var(--drop)}
.pill.warn{background:var(--warn-soft);color:var(--warn)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}
.bar{display:flex;height:26px;border-radius:5px;overflow:hidden;background:var(--sunk);margin:12px 0 8px}
.bar>span{display:block}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--muted)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:baseline}
.stat{display:flex;justify-content:space-between;gap:12px;padding:7px 0;
  border-bottom:1px dotted var(--rule);font-size:13px}
.stat:last-of-type{border-bottom:none}
.stat b{font-family:"IBM Plex Mono",monospace;font-weight:600;font-variant-numeric:tabular-nums}
audio{width:100%;height:34px;margin-top:6px;display:block}
.player{padding:12px 0;border-bottom:1px solid var(--rule)}
.player:last-child{border-bottom:none}
.player .lbl{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
.player .lbl span:first-child{font-weight:600;font-size:13px}
.player .lbl span:last-child{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--muted)}
.note{border-left:3px solid var(--accent);background:var(--accent-soft);
  padding:14px 18px;border-radius:0 8px 8px 0;font-size:14px}
.note.bad{border-left-color:var(--drop);background:var(--drop-soft)}
.dt{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:11px;
  padding:1px 7px;border-radius:4px;background:var(--sunk);color:var(--muted);font-weight:600}
ul{margin:10px 0 0;padding-left:20px;color:var(--muted)}
li{margin-bottom:6px}
li b{color:var(--ink)}
.foot{color:var(--muted);font-size:13px;border-top:1px solid var(--rule);padding-top:20px}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">Podcast pipeline · bước 1 · lọc nhiễu</div>
  <h1>Phát hiện trước, tách sau: sáu tổ hợp trên ba bản ghi</h1>
  <p class="lede">Hai bộ phát hiện sinh ra hai bản đồ nhạc khác nhau; mỗi bản đồ đi qua ba bộ tách.
  Trang này ghi lại đúng những gì đã chạy được, kể cả những chỗ không chạy được.</p>
</header>
<div id="app"></div>
<div class="foot">Sinh tự động từ <code>tools/stage1_detect.py</code> → <code>tools/stage1_separate.py</code> → <code>tools/stage1_report.py</code>.
Âm thanh nhúng thẳng trong trang dưới dạng mp3 48 kbps, mỗi đoạn 12 giây đầu của span dài nhất.</div>
</div>
<script id="payload" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const secs = v => v >= 60 ? `${Math.floor(v/60)}′${String(Math.round(v%60)).padStart(2,'0')}″` : `${v.toFixed(1)}s`;
const DET = {panns:'PANNs', sslam:'SSLAM'};

function availability(){
  const rows = [
    ['Mel-Band RoFormer Denoise','aufr33 · SDR 27.996','ok','Chạy được','Masking — che phổ, không sinh tín hiệu'],
    ['Mel-Band RoFormer Denoise Aggressive','aufr33 · SDR 27.977','ok','Chạy được','Bản mạnh tay hơn của trên'],
    ['BS-RoFormer ep_368','viperx · SDR 12.963','ok','Chạy được','Baseline hiện tại của pipeline'],
    ['SGMSE','sp-uhh · MIT','warn','Tạ ở Google Drive','Diffusion — <b>sinh</b> tín hiệu, không dùng cho corpus'],
    ['Edge-BS-RoFormer','MDPI 2025','no','Không tìm thấy repo/tạ',''],
    ['AP-BSR','—','no','Không tồn tại','Không có trong ZFTurbo, audio-separator, bs-roformer-infer, hay literature'],
    ['BSR-Flow / FlowBSRoformer','lucidrains','no','Chỉ có kiến trúc, chưa ai release tạ','Flow-matching từ nhiễu → target, tức là bịa tín hiệu'],
  ];
  return `<section><div class="shead"><span class="n">01</span><h2>Model nào thật sự chạy được</h2></div>
  <p class="sdesc">Bảy cái tên được nêu ra; ba cái có trọng số tải về chạy ngay, một cái phải tải tay, ba cái không tồn tại dưới dạng dùng được.</p>
  <div class="card scroll"><table><thead><tr><th>Model</th><th>Nguồn</th><th>Tình trạng</th><th>Ghi chú</th></tr></thead><tbody>
  ${rows.map(r=>`<tr><td><code>${esc(r[0])}</code></td><td class="mono" style="color:var(--muted)">${esc(r[1])}</td>
    <td><span class="pill ${r[2]}">${esc(r[3])}</span></td><td style="color:var(--muted)">${r[4]}</td></tr>`).join('')}
  </tbody></table></div></section>`;
}

function detectorSection(){
  const recs = Object.entries(D.recordings);
  return `<section><div class="shead"><span class="n">02</span><h2>Hai bộ phát hiện thấy gì</h2></div>
  <p class="sdesc">Cùng một ngưỡng, cùng một bộ nhãn AudioSet 527, cùng cách gom nhóm. Chỉ khác cách sinh khung.
  Thanh ngang là toàn bộ bản ghi: phần tô màu là phần bị đụng tới. Dòng <b>độ vụn</b> đáng chú ý nhất —
  cùng chừng ấy giây nhạc, nhưng chia thành bao nhiêu mảnh thì quyết định bao nhiêu mối nối phải ghép lại,
  và mảnh dưới một giây thì bị vứt cả mảnh.</p>
  <div class="grid2">${recs.map(([rid,rec])=>{
    return `<div class="card"><h3>${esc(rec.label)}</h3>
    ${Object.entries(rec.detectors).map(([d,e])=>{
      const dur=rec.duration, t=e.totals, mus=t.music||0, sing=t.singing||0, song=t.song||0;
      const pc=v=>(v/dur*100);
      const kept=dur-mus-sing-song;
      return `<div style="margin-top:16px">
        <div style="display:flex;justify-content:space-between;align-items:baseline">
          <span class="dt">${DET[d]||d}</span>
          <span class="mono num" style="font-size:11px;color:var(--muted)">${e.fps} fps · quét ${e.detect_seconds.toFixed(0)}s</span>
        </div>
        <div class="bar">
          <span style="width:${pc(kept)}%;background:var(--keep)"></span>
          <span style="width:${pc(mus)}%;background:var(--accent)"></span>
          <span style="width:${pc(sing)}%;background:var(--drop)"></span>
          <span style="width:${pc(song)}%;background:var(--drop);opacity:.55"></span>
        </div>
        <div class="stat"><span>Giữ nguyên</span><b class="num">${secs(kept)} · ${pc(kept).toFixed(1)}%</b></div>
        <div class="stat"><span>Nhạc nền dưới lời — đưa vào tách</span><b class="num">${secs(mus)} · ${e.n_separate} span</b></div>
        <div class="stat"><span>Có người hát — cắt bỏ</span><b class="num">${secs(sing)}</b></div>
        <div class="stat"><span>Nhạc không lời — cắt bỏ</span><b class="num">${secs(song)}</b></div>
        <div class="stat"><span>Độ vụn</span><b class="num">${e.span_len.n} span · trung vị ${e.span_len.median.toFixed(1)}s · dài nhất ${e.span_len.max.toFixed(1)}s${e.span_len.under_1s?` · <span style="color:var(--drop)">${e.span_len.under_1s} mảnh &lt;1s</span>`:''}</b></div>
      </div>`;}).join('')}
    <div class="legend" style="margin-top:14px">
      <span><i style="background:var(--keep)"></i>giữ</span>
      <span><i style="background:var(--accent)"></i>tách</span>
      <span><i style="background:var(--drop)"></i>hát</span>
      <span><i style="background:var(--drop);opacity:.55"></i>nhạc</span>
    </div></div>`;}).join('')}</div></section>`;
}

function listen(){
  const recs = Object.entries(D.recordings);
  return `<section><div class="shead"><span class="n">03</span><h2>Nghe thử</h2></div>
  <p class="sdesc">Mỗi khối là span nhạc dài nhất mà bộ phát hiện đó tìm ra, 12 giây đầu.
  Dòng trên cùng là nguyên bản; ba dòng dưới là ba bộ tách chạy trên đúng đoạn đó.
  Con số bên phải là phần năng lượng bộ tách quyết định <b>không phải giọng nói</b> —
  trung bình trên toàn bộ span của tổ hợp đó, không riêng đoạn đang nghe.</p>
  ${recs.map(([rid,rec])=>`<div style="margin-bottom:26px"><h3 style="margin-bottom:12px">${esc(rec.label)}</h3>
    <div class="grid2">${Object.entries(rec.detectors).map(([d,e])=>{
      if(!e.pick) return `<div class="card"><span class="dt">${DET[d]||d}</span>
        <p style="color:var(--muted);margin-top:12px;font-size:13px">Không có span nhạc nào để tách.</p></div>`;
      const rows=[['before','Nguyên bản', null]].concat(D.sep_order.map(s=>[s, D.separators[s], e.sep_stats[s]]));
      return `<div class="card"><div style="display:flex;justify-content:space-between;align-items:baseline">
        <span class="dt">${DET[d]||d}</span>
        <span class="mono num" style="font-size:11px;color:var(--muted)">${e.pick.start.toFixed(1)}s → ${e.pick.end.toFixed(1)}s</span></div>
        ${rows.map(([k,label,st])=>{
          const uri=e.clips[k];
          const meta = st ? (st.energy ? `bỏ ${st.energy.dropped_share.toFixed(1)}% năng lượng · ${st.ok} span`
                                        : `${st.ok} span ok${st.fail?` · ${st.fail} lỗi`:''}`) : 'đầu vào';
          return `<div class="player"><div class="lbl"><span>${esc(label)}</span><span>${esc(meta)}</span></div>
            ${uri?`<audio controls preload="none" src="${uri}"></audio>`
                 :`<p style="color:var(--muted);font-size:12px;margin-top:6px">chưa có kết quả</p>`}</div>`;
        }).join('')}</div>`;}).join('')}</div></div>`).join('')}</section>`;
}

function dist(){
  const rows=[];
  for(const [rid,rec] of Object.entries(D.recordings))
    for(const d of Object.keys(rec.detectors)){
      const x=D.dist[`${rid}__${d}`]; if(!x) continue;
      rows.push([rec.label, DET[d]||d, x]);
    }
  const cell=(v,hi)=>`<td class="mono num"${hi?' style="color:var(--accent);font-weight:600"':''}>${v.toFixed(3)}</td>`;
  return `<section><div class="shead"><span class="n">04</span><h2>Vì sao SSLAM cứu được nhiều hơn</h2></div>
  <p class="sdesc">Không phải vì ngưỡng lệch — tỉ lệ khung vượt 0.10 gần như trùng nhau giữa hai mô hình.
  Khác biệt nằm ở chỗ SSLAM tự tin hơn hẳn về <code>Speech</code>. Ngưỡng <code>SPEECH_PRESENT = 0.20</code>
  là thứ phân biệt “nhạc nền dưới lời” (tách được) với “nhạc trơ” (cắt bỏ), nên điểm speech cao hơn
  đẩy khung từ nhóm phải cắt sang nhóm cứu được.</p>
  <div class="card scroll"><table><thead><tr><th>Bản ghi</th><th>Detector</th><th>Nhãn</th>
  <th>p50</th><th>p90</th><th>p99</th><th>max</th><th>≥0.10</th></tr></thead><tbody>
  ${rows.map(([lbl,det,x])=>['speech','music','singing'].filter(k=>x[k]).map((k,i)=>
    `<tr>${i===0?`<td rowspan="3">${esc(lbl)}</td><td rowspan="3"><span class="dt">${esc(det)}</span></td>`:''}
     <td class="mono">${k}</td>${cell(x[k].p50,k==='speech')}${cell(x[k].p90)}${cell(x[k].p99)}${cell(x[k].max)}
     <td class="mono num" style="color:var(--muted)">${x[k].over.toFixed(1)}%</td></tr>`).join('')).join('')}
  </tbody></table></div>
  <div class="note bad" style="margin-top:16px"><b>Hệ quả: cổng phát hiện hát chết trên SSLAM.</b>
  Cổng đòi <code>singing ≥ speech + 0.15</code>. Trên vimeanh, SSLAM thấy tiếng hát lên tới 0.712
  (PANNs chỉ 0.077) nhưng vẫn báo 0 giây hát, vì speech của nó đã ở mức p50 0.89 — không nhãn nào vượt nổi.
  <code>SINGING_MARGIN</code> phải hiệu chuẩn lại riêng cho SSLAM trước khi dùng thật.</div>
  </section>`;
}

function notes(){
  return `<section><div class="shead"><span class="n">05</span><h2>Những chỗ đã thử và hỏng</h2></div>
  <p class="sdesc">Ghi lại để không ai thử lại lần nữa.</p>
  <div class="card"><div class="note bad"><b>SSLAM per-token không định vị được.</b>
  Head của SSLAM là một <code>Linear(768→527)</code> đặt trên đặc trưng đã pooling. Áp nó lên từng token 160 ms
  cho đường cong phẳng lì: mọi nhãn nằm trong 0.45–0.53 suốt cả file, vì logit trên token trần gần 0 nên sigmoid trả về một nửa.
  Không mang thông tin.</div>
  <div class="note" style="margin-top:14px"><b>Cửa sổ trượt ngắn thì được.</b>
  Đưa 1 giây audio zero-pad lên 10.24 s rồi đọc head clip đã hiệu chuẩn: Speech đảo 0.02 ↔ 0.87 và Music 0.22 ↔ 0.87
  ngay trên đoạn intro nhạc của LM8. Đổi lại, độ phân giải là 1 giây chứ không phải 160 ms — thô hơn mức 320 ms mà PANNs thật sự phân giải.</div>
  <ul>
    <li><b>SGMSE</b> tải được mã nguồn trên HuggingFace nhưng checkpoint nằm ở Google Drive, phải tải tay. Dù có tải được thì đầu ra vẫn là tín hiệu <b>sinh ra</b>, không dùng làm corpus.</li>
    <li><b>Edge-BS-RoFormer</b> được bài báo MDPI nói là đã mở mã, nhưng không tìm thấy repo hay trọng số ở đâu.</li>
    <li><b>Ngưỡng đang dùng được hiệu chuẩn cho PANNs.</b> SSLAM có phân bố điểm khác hẳn, nên cùng một ngưỡng <code>0.10</code> không có nghĩa như nhau trên hai mô hình. Đây là việc còn phải làm.</li>
  </ul></div></section>`;
}

document.getElementById('app').innerHTML =
  availability() + detectorSection() + listen() + dist() + notes();
</script>
"""


def main():
    out = sys.argv[1]
    dest = sys.argv[2]
    with open(os.path.join(out, "report_data.json")) as f:
        data = f.read()
    html = TEMPLATE.replace("__DATA__", data.replace("</", "<\\/"))
    with open(dest, "w") as f:
        f.write(html)
    print(f"{dest}: {os.path.getsize(dest)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
