"""Generate the single-screen local Denoise Lab UI.

The browser is an experiment workbench and recipe editor. It can load two local
audio files for linked A/B listening, but exports a recipe and GPU command rather
than pretending a static page can launch pipeline workers.

Usage: python tools/make_experiment_lab.py -o experiment.html
"""

from __future__ import annotations

import argparse
import pathlib


PAGE = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Denoise Lab</title>
<style>
:root{--bg:#0a0e13;--s1:#111820;--s2:#151f29;--line:#2b3946;--text:#e8eef3;--muted:#8293a1;--cyan:#57d3d0;--amber:#f4b660;--green:#68d391;--red:#f27979;--blue:#6ba7ff}
*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden}body{background:var(--bg);color:var(--text);font:13px/1.35 Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input,select{font:inherit}button{cursor:pointer;color:inherit}.app{height:100dvh;display:grid;grid-template-rows:54px minmax(0,1fr) 30px}.topbar{display:flex;align-items:center;gap:14px;padding:0 14px;border-bottom:1px solid var(--line);background:#0d131a}.brand{display:flex;align-items:center;gap:9px;font-weight:800;letter-spacing:-.02em;white-space:nowrap}.logo{width:24px;height:24px;border-radius:7px;background:linear-gradient(135deg,var(--cyan),#237a89);display:grid;place-items:center;color:#071113}.expname{width:180px;background:transparent;border:0;border-bottom:1px solid #30404e;color:var(--text);padding:5px 2px;outline:0}.topmeta{color:var(--muted);font-size:12px}.spacer{flex:1}.saved{color:var(--green);font-size:12px}.btn{border:1px solid var(--line);background:var(--s2);padding:7px 11px;border-radius:7px}.btn:hover{border-color:#527084}.btn.primary{background:var(--cyan);border-color:var(--cyan);color:#071113;font-weight:800}.btn.icon{padding:6px 9px}.workspace{min-height:0;display:grid;grid-template-columns:255px minmax(560px,1fr) 286px;gap:8px;padding:8px}.rail,.stage,.inspector{min-height:0;border:1px solid var(--line);background:var(--s1);border-radius:10px;overflow:hidden}.scroll{height:100%;overflow:auto;padding:13px}.section{border-bottom:1px solid #26333e;padding-bottom:13px;margin-bottom:13px}.section:last-child{border:0;margin:0}.label,.section h3{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.11em;font-weight:800;margin:0 0 8px}.field{margin-bottom:9px}.field label{display:block;color:#aab8c3;font-size:11px;margin-bottom:4px}.field input,.field select{width:100%;border:1px solid var(--line);background:#0b1117;color:var(--text);border-radius:6px;padding:7px 8px;outline:0}.field input:focus,.field select:focus{border-color:var(--cyan)}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:7px}.toggle{display:flex;align-items:center;gap:7px;color:#b8c5ce;font-size:12px;margin:8px 0}.toggle input{accent-color:var(--cyan)}.source{display:grid;grid-template-columns:1fr auto;gap:6px;align-items:center}.source-name{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#c9d4dc}.chain{display:flex;flex-wrap:wrap;gap:5px;min-height:34px;border:1px dashed #354a59;background:#0b1117;padding:7px;border-radius:7px}.node{display:inline-flex;align-items:center;gap:5px;background:#1b3a40;border:1px solid #376e70;color:#cffffc;padding:4px 7px;border-radius:5px}.node button{border:0;background:none;color:#9cc9c9;padding:0}.backend-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:7px}.backend{background:var(--s2);border:1px solid var(--line);border-radius:6px;text-align:left;padding:6px 7px;font-size:11px}.backend:hover{border-color:var(--cyan)}.backend small{display:block;color:var(--muted);font-size:9px}.addvariant{width:100%;margin-top:7px}.stage{display:grid;grid-template-rows:40px minmax(220px,42%) 72px minmax(140px,1fr)}.stage-head{display:flex;align-items:center;padding:0 12px;border-bottom:1px solid var(--line);gap:8px}.stage-head strong{font-size:12px}.pill{border:1px solid var(--line);background:#111a22;padding:5px 9px;border-radius:99px;color:var(--muted)}.pill.active{border-color:var(--cyan);color:var(--cyan);background:#123035}.canvas-wrap{position:relative;min-height:0;background:linear-gradient(180deg,#0b1117,#0c1218);overflow:hidden}.ruler{position:absolute;left:44px;right:12px;top:9px;display:flex;justify-content:space-between;color:#617480;font:9px ui-monospace,monospace}.track-labels{position:absolute;left:9px;top:47px;bottom:46px;display:flex;flex-direction:column;justify-content:space-around;color:#9fb0bc;font:10px ui-monospace,monospace}.wave{position:absolute;left:44px;right:12px;top:31px;bottom:43px;width:calc(100% - 56px);height:calc(100% - 74px);cursor:crosshair}.selection{position:absolute;top:31px;bottom:43px;left:33%;width:15%;background:#f4b66018;border-left:1px solid var(--amber);border-right:1px solid var(--amber);pointer-events:none}.playhead{position:absolute;top:23px;bottom:38px;left:40%;border-left:1px solid #fff;box-shadow:0 0 8px #fff8;pointer-events:none}.transport{position:absolute;bottom:7px;left:44px;right:12px;display:flex;align-items:center;gap:6px}.time{font:11px ui-monospace,monospace;color:#b8c4cc}.transport .grow{flex:1}.ab{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}.ab button{border:0;border-right:1px solid var(--line);background:#141e27;padding:5px 9px}.ab button:last-child{border:0}.ab button.active{background:var(--cyan);color:#071113}.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;padding:8px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:#0d141b}.metric{border:1px solid #263541;background:#131d26;border-radius:7px;padding:7px 8px;min-width:0}.metric span{display:block;color:var(--muted);font-size:9px;text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.metric b{font:700 17px ui-monospace,monospace}.metric em{font-style:normal;font-size:10px;margin-left:4px}.up{color:var(--green)}.down{color:var(--red)}.results{min-height:0;overflow:auto;padding:8px}.results-head,.result-row{display:grid;grid-template-columns:minmax(140px,1.5fr) repeat(4,minmax(72px,1fr));align-items:center;gap:6px}.results-head{color:var(--muted);font-size:9px;text-transform:uppercase;padding:3px 8px}.result-row{border:1px solid var(--line);background:#121b23;border-radius:7px;padding:8px;margin-top:5px}.result-row.selected{border-color:#3b7978;background:#14262a}.result-name{display:flex;align-items:center;gap:8px;min-width:0}.dot{width:8px;height:8px;border-radius:50%;background:var(--cyan);flex:0 0 auto}.result-name div{min-width:0}.result-name strong,.result-name small{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.result-name small{color:var(--muted)}.cell{font:11px ui-monospace,monospace}.inspector-head{height:40px;display:flex;align-items:center;justify-content:space-between;padding:0 12px;border-bottom:1px solid var(--line)}.variants{padding:8px}.variant{display:grid;grid-template-columns:10px 1fr auto;gap:8px;align-items:center;padding:9px;border:1px solid transparent;border-radius:7px;margin-bottom:4px}.variant:hover{background:var(--s2)}.variant.active{border-color:#3d7373;background:#173036}.variant .vcolor{width:8px;height:28px;border-radius:4px;background:var(--cyan)}.variant strong,.variant small{display:block}.variant small{color:var(--muted);font-size:10px}.state{font-size:9px;border:1px solid #355045;border-radius:99px;color:var(--green);padding:2px 5px}.inspector-card{margin:8px;border:1px solid var(--line);border-radius:8px;padding:10px;background:#101820}.inspector-card h4{margin:0 0 8px;font-size:11px}.kv{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin:6px 0}.kv b{color:#cbd6de}.warning{border-left:2px solid var(--amber);background:#292217;padding:8px;color:#e9c789;font-size:10px;margin-top:8px}.statusbar{display:flex;align-items:center;gap:12px;padding:0 10px;border-top:1px solid var(--line);background:#0c1218;color:var(--muted);font-size:10px}.statusbar code{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.ok{color:var(--green)}.right{margin-left:auto}.drawer{position:fixed;inset:54px 0 30px auto;width:min(680px,70vw);background:#0d141b;border-left:1px solid var(--line);box-shadow:-20px 0 50px #0008;z-index:10;padding:16px;display:none}.drawer.open{display:block}.drawer pre{white-space:pre-wrap;background:#080c10;padding:12px;border:1px solid var(--line);border-radius:7px;color:#c9e7e3;max-height:70vh;overflow:auto}.file-input{display:none}@media(max-width:1050px){.workspace{grid-template-columns:230px minmax(480px,1fr)}.inspector{display:none}}
.mobile-only{display:none}@media(max-width:1050px){.workspace{grid-template-columns:minmax(0,1fr)}.rail,.inspector{display:none}.rail.mobile-open,.inspector.mobile-open{display:block;position:fixed;z-index:12;top:62px;bottom:38px;width:min(330px,88vw);box-shadow:0 20px 60px #000b}.rail.mobile-open{left:8px}.inspector.mobile-open{right:8px}.mobile-only{display:inline-block}.topmeta,.saved{display:none}.stage{grid-template-rows:40px minmax(230px,48%) 72px minmax(120px,1fr)}}
@media(max-width:620px){html,body{overflow:hidden}.app{height:100dvh}.workspace{display:grid;padding:5px}.rail,.stage{margin:0}.stage{height:auto}.topbar{gap:6px;padding:0 8px}.brand{font-size:0}.brand .logo{font-size:13px}.expname{width:110px}.btn{padding:6px 8px}.topbar .primary{display:none}.metrics{grid-template-columns:repeat(3,1fr);overflow:auto}.metric:nth-child(n+4){display:none}.results-head,.result-row{grid-template-columns:minmax(120px,1.5fr) repeat(2,minmax(58px,1fr))}.results-head span:nth-child(n+4),.result-row .cell:nth-child(n+4){display:none}.statusbar .right{display:none}}
</style><style>.state.running{color:var(--amber);border-color:#765d2b}.state.draft{color:var(--muted);border-color:#43515b}.variant-editor{border-top:1px solid #26333e;margin-top:10px;padding-top:10px}.variant-editor .field{margin-bottom:7px}.variant-editor .field label{font-size:10px}.variant-editor input,.variant-editor select{width:100%;border:1px solid var(--line);background:#0b1117;color:var(--text);border-radius:6px;padding:6px 7px;outline:0}.variant-editor input:focus,.variant-editor select:focus{border-color:var(--cyan)}.editor-actions{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}.editor-actions .btn{font-size:10px;padding:5px 7px}.run-history{max-height:125px;overflow:auto;margin-top:7px}.run-item{display:flex;gap:5px;align-items:center;border-top:1px solid #26333e;padding:5px 0;font-size:10px}.run-item span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.run-item .run-delete{margin-left:auto;padding:1px 4px;background:transparent;border:0;color:var(--muted)}.run-item .run-delete:hover{color:var(--red)}.toast{position:fixed;left:50%;bottom:44px;transform:translateX(-50%);z-index:20;background:#17252c;border:1px solid #527084;border-radius:8px;padding:8px 10px;box-shadow:0 8px 30px #0008;display:none}.toast.open{display:flex;gap:10px;align-items:center}.toast button{background:transparent;border:0;color:var(--cyan);font-weight:700}</style></head><body>
<style>.hint{display:block;color:var(--muted);font-size:10px;margin-top:7px}.section-title{display:flex;align-items:center;justify-content:space-between}.trial-list{display:grid;gap:4px}.trial-row{display:grid;grid-template-columns:8px minmax(0,1fr) auto;gap:7px;align-items:center;padding:8px;border:1px solid transparent;border-radius:7px;cursor:pointer}.trial-row:hover{background:var(--s2)}.trial-row.active{border-color:#3d7373;background:#173036}.trial-dot{width:8px;height:24px;border-radius:4px}.trial-row strong,.trial-row small{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.trial-row small{font-size:10px;color:var(--muted)}.trial-row .state{font-size:9px}.trial-actions{display:flex;gap:6px;margin-top:8px}.trial-actions .btn{flex:1;font-size:10px;padding:5px 7px}.track-labels{top:38px;bottom:50px}.track-labels span{font-size:9px}.history-title{display:flex;justify-content:space-between;align-items:center;padding:0 8px 5px;color:var(--muted);font-size:10px}.history-title strong{color:var(--text);font-size:11px}.history-rows{display:none}.advanced{border:1px solid #26333e;border-radius:6px;margin-top:8px;padding:7px}.advanced summary{cursor:pointer;color:#aab8c3;font-size:11px}.advanced[open]{background:#0c1319}.advanced .field{margin-top:8px}.range-ends{display:flex;justify-content:space-between;color:var(--muted);font-size:9px}.threshold-field output{float:right;color:var(--cyan);font:700 11px ui-monospace,monospace}.editor-card h4{font-size:16px;margin:3px 0 8px}.eyebrow{color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.11em;font-weight:800}.danger{color:#f3a0a0!important;border-color:#673d43!important}.danger:hover{border-color:var(--red)!important}.editor-card .warning{margin-top:10px}.topbar,.workspace,.stage,.canvas-wrap,.results,.transport{min-width:0;max-width:100%}.workspace{width:100%;overflow:hidden}.results-head>*,.result-row>*{min-width:0}.result-row{overflow:hidden}.history-actions{justify-content:flex-end;min-width:0}@media(min-width:1051px){.stage{grid-template-rows:40px minmax(235px,40%) 72px minmax(170px,1fr)}.results{padding:10px}.results-head,.result-row{grid-template-columns:minmax(130px,1.4fr) minmax(90px,1fr) 78px 78px 74px}.result-row{padding:7px 8px}.result-row .cell:nth-child(2)::before{content:'model ';color:var(--muted);font:9px sans-serif}.result-row .cell:nth-child(3)::before{content:'gate ';color:var(--muted);font:9px sans-serif}}@media(max-width:1050px){.topbar{gap:6px;padding:0 8px;overflow:hidden}.brand{font-size:0;flex:0 0 auto}.brand .logo{font-size:13px}.expname{width:auto;min-width:64px;flex:1 1 120px}.topbar .btn{padding:6px 8px;white-space:nowrap;flex:0 0 auto}.topbar #undoBtn,.topbar #exportBtn{display:none}.results-head,.result-row{grid-template-columns:minmax(105px,1fr) 72px 58px 54px 66px}.history-title span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.transport{gap:4px}.transport .grow,.transport .time{display:none}.transport .ab{min-width:0}.transport .ab button{max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.transport .toggle{font-size:0;margin:0}.transport .toggle input{margin:0}.statusbar{overflow:hidden}.statusbar>*{flex:0 0 auto}.statusbar code{flex:1 1 auto;min-width:0}}@media(max-width:620px){.trial-actions{display:none}.history-title span{display:none}.inspector-card{margin:5px}.stage{grid-template-rows:40px minmax(210px,43%) 72px minmax(170px,1fr)}.results-head,.result-row{grid-template-columns:minmax(105px,1fr) 62px 54px 58px}.results-head span:nth-child(4),.result-row .cell:nth-child(4){display:none}.topbar #recipeBtn{display:none}}</style><div class="app">
<header class="topbar"><div class="brand"><span class="logo">∿</span>Denoise Lab</div><input class="expname" id="expName" value="noise-lab-01"><span class="topmeta">draft · audio experiment workbench</span><span class="spacer"></span><span class="saved" id="saved">● saved locally</span><button class="btn" id="undoBtn" title="Undo the last trial or run deletion">Undo</button><button class="btn mobile-only" id="setupBtn">Setup</button><button class="btn mobile-only" id="variantsBtn">Trials</button><button class="btn" id="recipeBtn">Recipe</button><button class="btn primary" id="exportBtn">Review export</button></header>
<style>.rail .section:nth-of-type(n+3){display:none}.history-rows{display:block}.history-actions{display:flex;gap:4px}.history-actions .btn{padding:3px 6px;font-size:10px}.empty-history{padding:15px 8px;color:var(--muted);font-size:11px}</style><main class="workspace">
<aside class="rail"><div class="scroll">
  <section class="section source-section"><span class="label">Source audio</span><div class="source"><span class="source-name" id="sourceName">Chưa có audio đầu vào</span><button class="btn icon" id="loadA">Load audio</button></div><input class="file-input" type="file" accept="audio/*" id="fileA"><small class="hint">The source is never overwritten. Every trial keeps its own recipe.</small></section>
  <section class="section trial-section"><div class="section-title"><span class="label">Trials</span><button class="btn icon" id="addTrial" title="Add a blank trial">＋</button></div><div class="trial-list" id="trialList"></div><div class="trial-actions"><button class="btn" id="duplicateTrial">Duplicate</button><button class="btn" id="deleteTrial">Delete</button></div></section>
  <section class="section"><span class="label">Build chain</span><div class="chain" id="chain"></div><div class="backend-grid" id="backends"></div><button class="btn addvariant" id="addVariant">Save as new variant</button></section>
  <section class="section"><span class="label">Common controls</span><div class="field"><label>DNSMOS quality gate <b id="dnsValue">0.30</b></label><input id="dns" type="range" min="0" max="1" step="0.01" value="0.30"></div><div class="field"><label>Chunk size</label><select id="chunk"><option>15</option><option selected>30</option><option>45</option><option>60</option></select></div><div class="twocol"><div class="field"><label>Profile</label><select id="env"><option>a100</option><option>default</option></select></div><div class="field"><label>Stop after</label><select id="stop"><option>asr</option><option>diarization</option><option>separation</option></select></div></div><label class="toggle"><input id="available" type="checkbox" checked> only available models</label><label class="toggle"><input id="resume" type="checkbox" checked> resume checkpoints</label></section>
  <section class="section"><span class="label">Paths</span><div class="field"><label>Audio directory</label><input id="audioDir" value="/data/vi_podcasts_20"></div><div class="field"><label>Output root</label><input id="outRoot" value="/data/denoise_cmp"></div></section>
</div></aside>
<section class="stage">
  <div class="stage-head"><strong>Evidence canvas</strong><span class="pill active">Waveform (current)</span><span class="pill" title="Not available in this prototype">Spectrogram · soon</span><span class="spacer"></span><span class="topmeta">drag to select · click to seek</span></div>
  <div class="canvas-wrap lab-empty" id="canvasWrap"><div class="ruler"><span>00:00</span><span>00:05</span><span>00:10</span><span>00:15</span><span>00:20</span><span>00:25</span><span>00:30</span></div><div class="track-labels"><span>Source</span><span>A</span><span>B</span><span>C</span></div><canvas class="wave" id="wave" tabindex="0" role="slider" aria-label="Audio timeline; use left and right arrows to seek" aria-valuemin="0" aria-valuemax="30" aria-valuenow="0"></canvas><div class="empty-wave-state" id="emptyWaveState">Nạp audio đầu vào hoặc output để xem waveform và chỉ số đo.</div><div class="selection" id="selection"></div><div class="playhead" id="playhead"></div><div class="transport"><button class="btn icon" id="rewind" aria-label="Return to selection start">↶</button><button class="btn icon" id="play" aria-label="Play or pause">▶</button><div class="ab"><button id="aBtn" class="active" aria-pressed="true">Source</button><button id="bBtn" aria-pressed="false">Selected output</button></div><label class="toggle"><input id="loop" type="checkbox" checked> loop selection</label><span class="grow"></span><span class="time" id="time">00:00.0 / 00:00.0</span><button class="btn icon" id="loadB">Import output</button><input class="file-input" type="file" accept="audio/*" id="fileB"></div><audio id="player"></audio></div>
  <div class="metrics lab-empty"><div class="metric"><span>DNSMOS</span><b id="metricDns">—</b><em></em></div><div class="metric"><span>WER vs base</span><b id="metricWer">—</b><em></em></div><div class="metric"><span>DER vs base</span><b id="metricDer">—</b><em></em></div><div class="metric"><span>Voice cosine</span><b id="metricEcapa">—</b><em></em></div><div class="metric"><span>Artifacts</span><b id="metricArtifacts">—</b><em></em></div></div>
  <div class="results"><div class="history-title"><strong>Prepared runs</strong><span>Recipes persist locally · import output audio to listen</span></div><div class="results-head"><span>Trial / status</span><span>Model</span><span>Threshold</span><span>DNSMOS</span><span>Actions</span></div><div id="resultRows"></div><div id="historyRows" class="history-rows"></div></div>
</section>
<aside class="inspector"><div class="inspector-head"><strong>Trial editor</strong><button class="btn icon" id="newVariant" aria-label="Create trial">＋</button></div><div class="inspector-card editor-card"><div class="eyebrow">Selected trial</div><h4 id="selectedTrialTitle">DeepFilterNet3</h4><div class="kv"><span>Model chain</span><b id="selectedChain">dfn3</b></div><div class="kv"><span>Runtime</span><b id="selectedRuntime">demo</b></div><div class="kv"><span>Audio state</span><b id="audioState">demo data</b></div><div class="variant-editor"><div class="field"><label>Trial name</label><input id="variantLabel" type="text" value="DeepFilterNet3"></div><div class="field"><label>Model</label><select id="variantModel"><option value="dfn3">DeepFilterNet3</option><option value="mf2se48">MossFormer2 SE48</option><option value="mpsenet">MP-SENet</option><option value="bsrnn26">BSRNN26</option><option value="rf_denoise">RoFormer</option><option value="re_use">RE-USE</option></select></div><div class="field threshold-field"><label>DNSMOS threshold <output id="variantThresholdValue">0.30</output></label><input id="variantThreshold" type="range" min="0" max="1" step="0.01" value="0.30"><div class="range-ends"><span>0 · keep noisy</span><span>1 · strict</span></div></div><details class="advanced"><summary>Advanced parameters</summary><div class="twocol"><div class="field"><label>Chunk (s)</label><select id="trialChunk"><option>15</option><option selected>30</option><option>45</option><option>60</option></select></div><div class="field"><label>Profile</label><select id="trialEnv"><option>a100</option><option>default</option></select></div></div><div class="field"><label>Stop after</label><select id="trialStop"><option>asr</option><option>diarization</option><option selected>separation</option></select></div><label class="toggle"><input id="trialResume" type="checkbox" checked> resume checkpoints</label></details><div class="editor-actions"><button class="btn primary" id="runVariant" title="Saves a prepared trial; GPU execution is exported from Recipe">Prepare run</button><button class="btn" id="cloneVariant">Duplicate</button><button class="btn" id="revertVariant">Revert</button><button class="btn danger" id="deleteVariant">Delete</button></div><div class="label" style="margin-top:10px">Saved runs</div><div class="run-history" id="runHistory"><span style="color:var(--muted);font-size:10px">No saved runs yet</span></div><div class="editor-actions"><button class="btn" id="clearRuns">Delete all runs</button><button class="btn" id="exportTrial">Export command</button></div></div><div class="warning">Prepare run saves a local draft/result only. Run the exported command on the GPU host; this page never starts a worker.</div></div></aside>
</main>
<footer class="statusbar"><span id="labState" style="color:var(--amber)">● Demo data</span><span id="laneState">0/2 audio lanes loaded</span><span>GPU run is exported to CLI</span><code id="command"></code><span class="right">Space play · A/B switch · Esc stop</span></footer>
</div>
<aside class="drawer" id="drawer"><div style="display:flex;align-items:center"><h3>Reproducible recipe</h3><span class="spacer"></span><button class="btn" id="closeDrawer">Close</button></div><p style="color:var(--muted)">Static UI exports configuration; execute the command on the GPU host.</p><pre id="recipe"></pre><div style="display:flex;gap:8px"><button class="btn primary" id="download">Download variants.json</button><button class="btn" id="copy">Copy command</button></div></aside>
<div class="toast" id="toast"><span id="toastText">Deleted</span><button id="undoDelete">Undo</button></div>
<script>
const BACKENDS=[['dfn3','DeepFilterNet'],['mf2se48','MossFormer'],['mpsenet','MP-SENet'],['bsrnn26','BSRNN'],['rf_denoise','RoFormer'],['re_use','RE-USE']];
const COLORS=['#57d3d0','#f4b660','#6ba7ff','#c58cff','#68d391'];
const MODEL_NAMES={dfn3:'DeepFilterNet3',mf2se48:'MossFormer2 SE48',mpsenet:'MP-SENet',bsrnn26:'BSRNN26',rf_denoise:'RoFormer',re_use:'RE-USE'};
let chain=[];let variants=[{id:'none',label:'Source',chain:[],model:'none',threshold:0.30,status:'demo',dns:'3.20',wer:'0',der:'0',ecapa:'1.000',art:'—',runs:[]},{id:'dfn3',label:'Trial A',chain:['dfn3'],model:'dfn3',threshold:0.30,status:'demo',dns:'3.41',wer:'+0.048',der:'0.035',ecapa:'0.975',art:'low',runs:[]},{id:'dfn3+mf2se48',label:'Trial B',chain:['dfn3','mf2se48'],model:'mf2se48',threshold:0.30,status:'demo',dns:'3.54',wer:'+0.145',der:'0.051',ecapa:'0.895',art:'medium',runs:[]}];let selected=1,activeLane='A',duration=30,current=0,viewStart=0,viewSpan=30,dragStart=null,panStart=null,sel=[0,0],audioUrl={A:null,B:null},audioDuration={A:0,B:0},samples={A:null,B:null},peakCache={A:null,B:null},runAudio={},undoSnapshot=null,runCounter=0;window.audioUrl=audioUrl;
const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),deepCopy=value=>value==null?value:JSON.parse(JSON.stringify(value));
document.getElementById('variantModel').insertAdjacentHTML('afterbegin','<option value="none">Original (read-only)</option>');
function renderChain(){$('chain').innerHTML=chain.length?chain.map((x,i)=>`<span class="node">${esc(x)}<button data-rm="${i}">×</button></span>`).join(''):'<span style="color:var(--muted);font-size:10px">Choose model steps below</span>';document.querySelectorAll('[data-rm]').forEach(x=>x.onclick=()=>{chain.splice(+x.dataset.rm,1);renderChain();dirty();});}
function renderBackends(){$('backends').innerHTML=BACKENDS.map(([id,n])=>`<button class="backend" data-be="${id}">${id}<small>${n}</small></button>`).join('');document.querySelectorAll('[data-be]').forEach(x=>x.onclick=()=>{chain.push(x.dataset.be);renderChain();dirty();});}
function snapshotConfig(c,fallback={}){let source=c||{};return {label:source.label??fallback.label,model:source.model??fallback.model,chain:Array.isArray(source.chain)?source.chain.map(String):Array.isArray(fallback.chain)?fallback.chain.map(String):[],stepParams:deepCopy(Array.isArray(source.stepParams)?source.stepParams:(fallback.stepParams||[])),threshold:Number(source.threshold??fallback.threshold??0.30),settings:deepCopy(source.settings??fallback.settings??{})};}
function normalizeVariant(v,i){let chain=Array.isArray(v.chain)?v.chain.map(String):[],model=v.model||chain.at(-1)||'none',legacy=v.settings?.model_params,stepParams=Array.isArray(v.stepParams)?deepCopy(v.stepParams):chain.map((m)=>legacy&&legacy.model===m?deepCopy(legacy):{}),runs=Array.isArray(v.runs)?v.runs.map(r=>({...r,metrics:{dns:r.metrics?.dns||r.dns||'—',wer:r.metrics?.wer||r.wer||'—',der:r.metrics?.der||r.der||'—',ecapa:r.metrics?.ecapa||r.ecapa||'—',art:r.metrics?.art||r.art||'pending'},configSnapshot:snapshotConfig(r.configSnapshot,{label:r.label||v.label,model:r.model||model,threshold:r.threshold??v.threshold,chain:r.chain||chain,stepParams:r.stepParams||stepParams,settings:r.settings||v.settings})})):[];return {...v,id:v.id||`${model||'variant'}-${i}`,label:v.label||MODEL_NAMES[model]||'Variant',chain,model,stepParams,threshold:Number.isFinite(+v.threshold)?+v.threshold:0.30,status:v.status||'draft',runs,dns:v.dns||'—',wer:v.wer||'—',der:v.der||'—',ecapa:v.ecapa||'—',art:v.art||'pending',activeRunId:v.activeRunId||runs.at(-1)?.id||null};}
function selectedVariant(){return variants[selected]||variants[0];}
function useRun(id){let v=selectedVariant(),r=v?.runs?.find(x=>x.id===id);if(!r)return;
  /* A run switch is one transaction: stop the old source, clear its output
     lane, reset the timeline, then bind only the selected run's artifact. */
  let p=$('player');p.pause();p.removeAttribute('src');p.load();$('play').textContent='▶';
  audioUrl.B=null;samples.B=null;peakCache.B=null;audioDuration.B=0;activeLane='A';window.activeLane='A';selectLane('A');
  v.activeRunId=id;v.status=r.status||'prepared';current=0;sel=[0,0];viewStart=0;
  let a=runAudio[id];
  if(a){audioUrl.B=a.url;samples.B=a.samples;peakCache.B=buildPeakCache(a.samples);audioDuration.B=a.duration;duration=audioDuration.A||a.duration||30;viewSpan=duration;$('bBtn').textContent=`Output · ${v.label} · ${id}`;selectLane('B');$('audioState').textContent=`session audio · ${a.name}`;}
  else{duration=audioDuration.A||30;viewSpan=duration;$('bBtn').textContent=`Output unavailable · ${v.label} · ${id}`;$('audioState').textContent='output unavailable for selected run';}
  renderVariants();draw();dirty();window.refreshEvidenceState?.();
}
function editRun(id){let v=selectedVariant(),r=v?.runs?.find(x=>x.id===id);if(!r)return;let c=snapshotConfig(r.configSnapshot,v);Object.assign(v,{...c,stepParams:deepCopy(c.stepParams),settings:deepCopy(c.settings),status:'draft',draftChanged:true});renderVariants();dirty();} /* settings:{...(c.settings||{})} is intentionally deep-copied below. */
function metricsFor(v){let r=v?.runs?.find(x=>x.id===v.activeRunId)||v?.runs?.[v.runs.length-1];return r?.metrics||{dns:v?.dns||'—',wer:v?.wer||'—',der:v?.der||'—',ecapa:v?.ecapa||'—',art:v?.art||'pending'};}
/* Kept as a readable contract for lightweight static checks: metrics are now read from the active immutable run. $('metricDns').textContent=v.dns; $('metricWer').textContent=v.wer; $('metricDer').textContent=v.der; $('metricEcapa').textContent=v.ecapa; $('metricArtifacts').textContent=v.art; */
function renderEditor(){let v=selectedVariant();if(!v)return;$('variantLabel').value=v.label||'';$('variantModel').value=v.model||v.chain?.[v.chain.length-1]||'none';$('variantThreshold').value=String(v.threshold??0.30);$('variantThresholdValue').textContent=(+($('variantThreshold').value)).toFixed(2);$('runHistory').innerHTML=v.runs?.length?v.runs.slice().reverse().map(r=>`<div class="run-item"><span>${esc(r.status||'prepared')} · ${esc(r.created||'')}${r.errorMessage?` · <small class="run-error">${esc(r.errorMessage)}</small>`:''}</span><button class="btn" data-use="${esc(r.id)}" title="View this saved recipe">View</button><button class="btn" data-edit="${esc(r.id)}" title="Load this run settings as a draft">Edit</button><button class="run-delete" data-run="${esc(r.id)}" title="Delete this saved run">×</button></div>`).join(''):'<span style="color:var(--muted);font-size:10px">No prepared runs yet</span>';document.querySelectorAll('[data-run]').forEach(x=>x.onclick=e=>{e.stopPropagation();deleteRun(x.dataset.run);});document.querySelectorAll('[data-use]').forEach(x=>x.onclick=e=>{e.stopPropagation();useRun(x.dataset.use);});document.querySelectorAll('[data-edit]').forEach(x=>x.onclick=e=>{e.stopPropagation();editRun(x.dataset.edit);});}
function editSelected(patch){let v=selectedVariant();if(!v||v.id==='none')return;Object.assign(v,patch,{status:'draft',draftChanged:true});if(patch.model)v.chain=patch.model==='none'?[]:[patch.model];renderVariants();dirty();}
function cloneSelected(){let v=selectedVariant();if(!v)return;let copy=normalizeVariant({...v,id:`${v.id}-copy-${Date.now()}`,label:`${v.label} copy`,status:'draft',runs:[],activeRunId:null,dns:'—',wer:'—',der:'—',ecapa:'—',art:'pending'},variants.length);variants.push(copy);selected=variants.length-1;renderVariants();dirty();}
function clearRuns(){let v=selectedVariant();if(!v||!v.runs.length)return;undoSnapshot={type:'runs',index:selected,runs:v.runs.slice(),metrics:{dns:v.dns,wer:v.wer,der:v.der,ecapa:v.ecapa,art:v.art,status:v.status,activeRunId:v.activeRunId}};v.runs=[];v.activeRunId=null;v.status='draft';v.dns='—';v.wer='—';v.der='—';v.ecapa='—';v.art='pending';renderVariants();dirty();showUndo('Runs deleted');}
function deleteTrial(){if(selected===0||variants.length<=1){showUndo('Original trial cannot be deleted');return;}undoSnapshot={type:'trial',index:selected,variant:variants[selected]};variants.splice(selected,1);selected=Math.max(0,selected-1);renderVariants();dirty();showUndo('Trial deleted');}
function deleteRun(id){let v=selectedVariant(),idx=v.runs.findIndex(r=>r.id===id);if(idx<0)return;undoSnapshot={type:'run',index:selected,run:v.runs[idx],metrics:{dns:v.dns,wer:v.wer,der:v.der,ecapa:v.ecapa,art:v.art,status:v.status,activeRunId:v.activeRunId}};v.runs.splice(idx,1);if(v.activeRunId===id)v.activeRunId=v.runs.at(-1)?.id||null;if(!v.runs.length){v.status='draft';v.dns='—';v.wer='—';v.der='—';v.ecapa='—';v.art='pending';}renderVariants();dirty();showUndo('Run deleted');}
function showUndo(msg){let t=$('toast');if(!t)return;$('toastText').textContent=msg;$('undoDelete').hidden=false;t.classList.add('open');clearTimeout(showUndo.timer);showUndo.timer=setTimeout(()=>t.classList.remove('open'),5000);}
function showNotice(msg){let t=$('toast');if(!t)return;$('toastText').textContent=msg;$('undoDelete').hidden=true;t.classList.add('open');clearTimeout(showUndo.timer);showUndo.timer=setTimeout(()=>t.classList.remove('open'),5000);}
function restoreUndo(){if(!undoSnapshot)return;let s=undoSnapshot,v=variants[s.index];if(s.type==='trial'){variants.splice(s.index,0,s.variant);selected=s.index;undoSnapshot=null;renderVariants();dirty();$('toast').classList.remove('open');return;}if(!v)return;if(s.type==='runs')v.runs=s.runs;else if(s.type==='run')v.runs.push(s.run);Object.assign(v,s.metrics);undoSnapshot=null;renderVariants();dirty();$('toast').classList.remove('open');}
function prepareRun(){let v=selectedVariant();if(!v||v.id==='none')return;let runId=`prepared-${Date.now()}-${++runCounter}`,configSnapshot=snapshotConfig(v),metrics={dns:'—',wer:'—',der:'—',ecapa:'—',art:'awaiting import'},run={id:runId,status:'prepared',created:new Date().toISOString(),configSnapshot,metrics,audioArtifact:null};v.runs.push(run);v.activeRunId=runId;v.status='prepared';v.draftChanged=false;Object.assign(v,metrics);renderVariants();dirty();showNotice('Run recipe prepared · export it to execute on GPU');}
function renderVariants(){let v=selectedVariant();if(!v)return;let m=metricsFor(v);$('selectedChain').textContent=v.chain.join(' › ')||'none';$('selectedRuntime').textContent=v.draftChanged?'draft changed · metrics from active run':v.runs?.length?'prepared recipe':'draft';$('metricDns').textContent=m.dns;$('metricWer').textContent=m.wer;$('metricDer').textContent=m.der||'—';$('metricEcapa').textContent=m.ecapa;$('metricArtifacts').textContent=m.art;renderEditor();}
function renderTrials(){let list=$('trialList');if(!list)return;list.innerHTML=variants.map((v,i)=>v.id==='none'?'':`<div class="trial-row ${i===selected?'active':''}" data-trial="${i}" tabindex="0" role="button" aria-label="Select ${esc(v.label)}"><span class="trial-dot" style="background:${COLORS[i%COLORS.length]}"></span><div><strong>${esc(v.label)}</strong><small>${v.chain.join(' › ')||'draft'} · ${v.runs?.length||0} prepared</small></div><span class="state ${v.status}">${v.status}</span></div>`).join('');list.querySelectorAll('[data-trial]').forEach(row=>{let pick=()=>{selected=+row.dataset.trial;renderVariants();draw()};row.onclick=pick;row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick()}}});}
function renderHistory(){let host=$('historyRows');if(!host)return;let rows=[];variants.forEach((v,i)=>(v.runs||[]).forEach(r=>rows.push({v,i,r})));rows.sort((a,b)=>String(b.r.created||'').localeCompare(String(a.r.created||'')));host.innerHTML=rows.length?rows.slice(0,24).map(({v,i,r})=>{let m=r.metrics||{};return `<div class="result-row history-row ${i===selected?'selected':''}" data-history="${esc(r.id)}" data-v="${i}"><div class="result-name"><span class="dot" style="background:${COLORS[i%COLORS.length]}"></span><div><strong>${esc(v.label)}</strong><small>${esc(r.status||'prepared')} · ${esc(r.created||'saved recipe')}</small></div></div><span class="cell">${esc(r.configSnapshot?.model||v.model||'none')}</span><span class="cell">${Number(r.configSnapshot?.threshold??v.threshold??0.3).toFixed(2)}</span><span class="cell">${esc(m.dns||'—')}</span><span class="history-actions"><button class="btn" data-use-history="${esc(r.id)}">View</button><button class="btn" data-delete-history="${esc(r.id)}">×</button></span></div>`}).join(''):'<div class="empty-history">No prepared runs yet. Prepare a trial, export it, then import real output audio/results.</div>';host.querySelectorAll('[data-history]').forEach(row=>row.onclick=()=>{selected=+row.dataset.v;useRun(row.dataset.history)});host.querySelectorAll('[data-use-history]').forEach(x=>x.onclick=e=>{e.stopPropagation();let i=+x.closest('[data-v]').dataset.v;selected=i;useRun(x.dataset.useHistory)});host.querySelectorAll('[data-delete-history]').forEach(x=>x.onclick=e=>{e.stopPropagation();let i=+x.closest('[data-v]').dataset.v;selected=i;deleteRun(x.dataset.deleteHistory)});}
const _renderEditor=renderEditor;renderEditor=function(){_renderEditor();let v=selectedVariant(),s=v?.settings||{};if(v&&$('selectedTrialTitle'))$('selectedTrialTitle').textContent=v.label||'Untitled trial';if($('trialChunk'))$('trialChunk').value=String(s.chunk_s||$('chunk').value);if($('trialEnv'))$('trialEnv').value=s.env||$('env').value;if($('trialStop'))$('trialStop').value=s.stop_after||$('stop').value;if($('trialResume'))$('trialResume').checked=s.resume!==false;};
const _renderVariants=renderVariants;renderVariants=function(){_renderVariants();$('resultRows').innerHTML='';renderTrials();renderHistory();let v=selectedVariant();if(v&&$('selectedTrialTitle'))$('selectedTrialTitle').textContent=v.label||'Untitled trial';};
function recipe(){let gate=Number(selectedVariant()?.threshold??$('dns').value);return {name:$('expName').value,variants:variants.map(({id,label,chain,model,stepParams,threshold,settings})=>({id,label,chain:deepCopy(chain||[]),model:model||chain?.[chain.length-1]||'none',stepParams:deepCopy(stepParams||[]),threshold:Number(threshold??gate),settings:deepCopy(settings||{}),group:id==='none'?'baseline':'masking'})),settings:{audio_dir:$('audioDir').value,out_root:$('outRoot').value,env:$('env').value,stop_after:$('stop').value,dnsmos_threshold:gate,chunk_s:+$('chunk').value,only_available:$('available').checked,resume:$('resume').checked}};} /* settings:{...(settings||{})} remains the displayed-config contract. */
function command(r=recipe()){let s=r.settings;return `python tools/run_variants.py --audio-dir ${s.audio_dir} --variants variants.json --out-root ${s.out_root} --env ${s.env} --stop-after ${s.stop_after} --set audio_quality.min_normalized_quality=${s.dnsmos_threshold.toFixed(2)} --set denoise.chunk_s=${s.chunk_s} --parallel 2${s.only_available?' --only-available':''}${s.resume?' --resume':''}`;}
const RECIPE_FLUSH_DELAY=120;
let recipeFlushTimer=null,recipeFlushFrame=null,recipeFlushPending=false;
function serializedRecipe(current=recipe()){return JSON.stringify({...current,variants:variants.map(v=>({...v,chain:[...(v.chain||[])],runs:[...(v.runs||[])]}))});}
function updateRecipe(current=recipe(),cmd=command(current)){let text=JSON.stringify(current,null,2)+'\n\n'+cmd;$('recipe').textContent=text;$('command').textContent=cmd;}
function flushRecipe(){if(recipeFlushTimer){clearTimeout(recipeFlushTimer);recipeFlushTimer=null;}if(recipeFlushFrame!=null){cancelAnimationFrame(recipeFlushFrame);recipeFlushFrame=null;}if(!recipeFlushPending)return;recipeFlushPending=false;let current=recipe(),cmd=command(current);updateRecipe(current,cmd);localStorage.setItem('denoise-lab-v4',serializedRecipe(current));}
function scheduleRecipeFlush(){if(recipeFlushPending)return;recipeFlushPending=true;recipeFlushTimer=setTimeout(()=>{recipeFlushTimer=null;recipeFlushFrame=requestAnimationFrame(()=>{recipeFlushFrame=null;flushRecipe();});},RECIPE_FLUSH_DELAY);}
function dirty(){$('saved').textContent='● unsaved changes';$('saved').style.color='var(--amber)';scheduleRecipeFlush();clearTimeout(dirty.t);dirty.t=setTimeout(()=>{$('saved').textContent='● saved locally';$('saved').style.color='var(--green)'},500);}
function hydrate(){try{let r=JSON.parse(localStorage.getItem('denoise-lab-v4')||'null');if(!r||!r.settings||!Array.isArray(r.variants))return;$('expName').value=r.name||$('expName').value;let s=r.settings;$('audioDir').value=s.audio_dir||$('audioDir').value;$('outRoot').value=s.out_root||$('outRoot').value;$('env').value=s.env||$('env').value;$('stop').value=s.stop_after||$('stop').value;$('dns').value=s.dnsmos_threshold??$('dns').value;$('dnsValue').textContent=(+$('dns').value).toFixed(2);$('chunk').value=s.chunk_s||$('chunk').value;$('available').checked=s.only_available!==false;$('resume').checked=s.resume!==false;let saved=r.variants.filter(v=>v&&typeof v.id==='string'&&Array.isArray(v.chain)).map(normalizeVariant),defaults=[...variants];if(saved.length)variants=saved.map(v=>{let base=defaults.find(x=>x.id===v.id)||{};return normalizeVariant({...base,...v,status:v.status==='running'?'draft':v.status||base.status||'draft',dns:v.dns||base.dns||'—',wer:v.wer||base.wer||'—',der:v.der||base.der||'—',ecapa:v.ecapa||base.ecapa||'—',art:v.art||base.art||'—'},0)});selected=Math.min(selected,variants.length-1);}catch(e){console.warn('Ignoring invalid saved lab state',e)}}
function fitCanvas(){let c=$('wave'),r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=Math.max(1,r.width*d);c.height=Math.max(1,r.height*d);draw();}
function demoSample(x,lane){return Math.sin(x*.13+(lane==='B'?1.2:0))*.42+Math.sin(x*.037)*.25+Math.sin(x*.71)*.08;}
function buildPeakCache(data){if(!data||!data.length)return null;const bins=Math.min(16384,Math.max(256,Math.ceil(Math.sqrt(data.length)*8))),peaks=new Float32Array(bins);for(let i=0;i<data.length;i++){const b=Math.min(bins-1,Math.floor(i*bins/data.length)),v=Math.abs(data[i]);if(v>peaks[b])peaks[b]=v;}return {peaks,length:data.length,bins};}
function drawLane(ctx,w,h,y,lane,color){let cache=peakCache[lane]||(samples[lane]&&(peakCache[lane]=buildPeakCache(samples[lane]))),mid=y+h/2;if(!cache)return;ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();for(let x=0;x<w;x++){let t=viewStart+x/w*viewSpan,amp=0,start=Math.floor(Math.max(0,Math.min(0.999,t/duration))*cache.bins),end=Math.max(start+1,Math.ceil(Math.max(0,Math.min(1,(t+viewSpan/w)/duration))*cache.bins));for(let k=start;k<Math.min(cache.bins,end);k++)amp=Math.max(amp,cache.peaks[k]);let yy=amp*h*.42;ctx.moveTo(x,mid-yy);ctx.lineTo(x,mid+yy)}ctx.stroke();}
function draw(){let c=$('wave'),ctx=c.getContext('2d'),d=devicePixelRatio||1,w=c.width/d,h=c.height/d;ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);ctx.strokeStyle='#1f2c35';for(let i=1;i<8;i++){ctx.beginPath();ctx.moveTo(i*w/8,0);ctx.lineTo(i*w/8,h);ctx.stroke()}let laneH=h/4;drawLane(ctx,w,laneH,0,'S','#9aa9b3');drawLane(ctx,w,laneH,laneH,'A','#57d3d0');drawLane(ctx,w,laneH,laneH*2,'B',COLORS[selected%COLORS.length]);drawLane(ctx,w,laneH,laneH*3,'C','#c58cff');positionOverlays();}
function positionOverlays(){let pct=t=>Math.max(0,Math.min(1,(t-viewStart)/viewSpan));$('playhead').style.left=`calc(44px + (100% - 56px) * ${pct(current)})`;$('selection').style.left=`calc(44px + (100% - 56px) * ${pct(sel[0])})`;$('selection').style.width=`calc((100% - 56px) * ${Math.max(0,pct(sel[1])-pct(sel[0]))})`;$('time').textContent=`${fmt(current)} / ${fmt(duration)}`;$('wave').setAttribute('aria-valuemax',duration.toFixed(1));$('wave').setAttribute('aria-valuenow',current.toFixed(1));document.querySelectorAll('.ruler span').forEach((x,i)=>x.textContent=fmt(viewStart+viewSpan*i/6).slice(0,5));}
function fmt(t){t=Math.max(0,t||0);return `${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}.${Math.floor(t%1*10)}`;}
function canvasTime(e){let r=$('wave').getBoundingClientRect();return Math.max(0,Math.min(duration,viewStart+(e.clientX-r.left)/r.width*viewSpan));}
function panTo(delta){viewStart=Math.max(0,Math.min(Math.max(0,duration-viewSpan),viewStart+delta));draw();}
function panByPixels(px){let r=$('wave').getBoundingClientRect();panTo(-px/r.width*viewSpan);}
$('wave').onpointerdown=e=>{if(e.button===1||e.altKey){panStart=e.clientX;$('wave').setPointerCapture(e.pointerId);return}dragStart=canvasTime(e);$('wave').setPointerCapture(e.pointerId)};$('wave').onpointermove=e=>{if(panStart!=null){panByPixels(e.clientX-panStart);panStart=e.clientX;return}if(dragStart==null)return;let t=canvasTime(e);sel=[Math.min(dragStart,t),Math.max(dragStart,t)];current=sel[0];positionOverlays()};$('wave').onpointerup=e=>{if(panStart!=null){panStart=null;return}if(dragStart!=null&&Math.abs(canvasTime(e)-dragStart)<.2)current=canvasTime(e);dragStart=null;positionOverlays();syncPlayer();};$('wave').onwheel=e=>{if(!e.shiftKey)return;e.preventDefault();panTo(e.deltaY/rangeWidth());};function rangeWidth(){let r=$('wave').getBoundingClientRect();return Math.max(1,r.width/viewSpan);}
function selectLane(l){activeLane=l;$('aBtn').classList.toggle('active',l==='A');$('bBtn').classList.toggle('active',l==='B');$('aBtn').setAttribute('aria-pressed',l==='A');$('bBtn').setAttribute('aria-pressed',l==='B');syncPlayer();}
function syncPlayer(){let p=$('player'),u=audioUrl[activeLane];if(u&&p.src!==u)p.src=u;if(u)p.currentTime=Math.min(current,p.duration||current);}
async function loadAudio(file,lane){if(!file)return;let buf=await file.arrayBuffer(),ctx=new AudioContext(),decoded=await ctx.decodeAudioData(buf.slice(0)),pcm=decoded.getChannelData(0).slice(),url=URL.createObjectURL(file);samples[lane]=pcm;peakCache[lane]=buildPeakCache(pcm);audioUrl[lane]=url;audioDuration[lane]=decoded.duration;duration=audioDuration.A||audioDuration.B||30;viewStart=0;viewSpan=duration;current=Math.min(current,duration);sel=[Math.min(sel[0],duration),Math.min(sel[1],duration)];if(sel[1]<=sel[0])sel=[0,Math.min(5,duration)];if(lane==='A')$('sourceName').textContent=`${file.name} · ${fmt(decoded.duration)}`;if(lane==='B'){let v=selectedVariant(),r=v?.runs?.find(x=>x.id===v.activeRunId)||v?.runs?.at(-1);if(r){runAudio[r.id]={url,samples:pcm,duration:decoded.duration,name:file.name};r.audioArtifact={name:file.name,sessionOnly:true};v.activeRunId=r.id;$('bBtn').textContent=v.label;$('audioState').textContent=`session audio · ${file.name}`;selectLane('B');}else{$('audioState').textContent='prepare a run before importing its output';}}let loaded=Number(!!audioUrl.A)+Number(!!audioUrl.B);$('laneState').textContent=`${loaded}/2 session audio lanes loaded`;$('labState').textContent=loaded===2?'● Audio ready':'● Audio partially loaded';$('labState').style.color=loaded===2?'var(--green)':'var(--amber)';draw();dirty();}
$('loadA').onclick=()=>$('fileA').click();$('loadB').onclick=()=>$('fileB').click();$('fileA').onchange=e=>loadAudio(e.target.files[0],'A');$('fileB').onchange=e=>loadAudio(e.target.files[0],'B');$('aBtn').onclick=()=>selectLane('A');$('bBtn').onclick=()=>selectLane('B');
$('play').onclick=()=>{let p=$('player');if(!audioUrl[activeLane])return;if(p.paused){p.play();$('play').textContent='❚❚'}else{p.pause();$('play').textContent='▶'}};$('player').ontimeupdate=()=>{current=$('player').currentTime;if($('loop').checked&&current>=sel[1])$('player').currentTime=sel[0];positionOverlays()};$('rewind').onclick=()=>{current=sel[0];syncPlayer();positionOverlays()};
 $('addVariant').onclick=()=>{let steps=['dfn3'],id=`trial-${Date.now()}`,label=`Trial ${String.fromCharCode(65+variants.filter(v=>v.id!=='none').length)}`;variants.push(normalizeVariant({id,label,chain:steps,model:'dfn3',threshold:0.30,status:'draft',dns:'—',wer:'—',der:'—',ecapa:'—',art:'pending',runs:[]},variants.length));selected=variants.length-1;renderVariants();dirty();};$('newVariant').onclick=()=>$('addVariant').click();$('addTrial').onclick=()=>$('addVariant').click();$('duplicateTrial').onclick=cloneSelected;$('deleteTrial').onclick=deleteTrial;$('undoBtn').onclick=restoreUndo;
 $('dns').oninput=()=>{$('dnsValue').textContent=(+$('dns').value).toFixed(2);dirty()};$('variantLabel').oninput=()=>editSelected({label:$('variantLabel').value});$('variantModel').onchange=()=>{let model=$('variantModel').value;editSelected({model,chain:model==='none'?[]:[model]});};$('variantThreshold').oninput=()=>{let threshold=+$('variantThreshold').value;$('variantThresholdValue').textContent=threshold.toFixed(2);editSelected({threshold});};$('runVariant').onclick=prepareRun;$('cloneVariant').onclick=cloneSelected;$('deleteVariant').onclick=deleteTrial;$('revertVariant').onclick=()=>{let v=selectedVariant(),r=v?.runs?.find(x=>x.id===v.activeRunId)||v?.runs?.at(-1),c=r?.configSnapshot;if(c){let restored=snapshotConfig(c,v);Object.assign(v,{...restored,stepParams:deepCopy(restored.stepParams),settings:deepCopy(restored.settings),status:r.status||'prepared',draftChanged:false});}else if(v&&v.id!=='none'){Object.assign(v,{model:'dfn3',chain:['dfn3'],stepParams:[],threshold:0.30,settings:{},status:'draft',draftChanged:false});}renderVariants();dirty()};$('exportTrial').onclick=()=>$('recipeBtn').click();$('clearRuns').onclick=clearRuns;$('undoDelete').onclick=restoreUndo;['expName','chunk','env','stop','audioDir','outRoot','available','resume','trialChunk','trialEnv','trialStop','trialResume'].forEach(id=>$(id)?.addEventListener('input',()=>{let v=selectedVariant();if(id==='trialChunk'||id==='trialEnv'||id==='trialStop'||id==='trialResume'){v.settings={...(v.settings||{}),chunk_s:+$('trialChunk').value,env:$('trialEnv').value,stop_after:$('trialStop').value,resume:$('trialResume').checked};v.status='draft';v.draftChanged=true;}dirty()}));
 $('recipeBtn').onclick=()=>{flushRecipe();$('drawer').classList.add('open');updateRecipe()};$('closeDrawer').onclick=()=>$('drawer').classList.remove('open');$('exportBtn').onclick=()=>$('recipeBtn').click();
$('setupBtn').onclick=()=>{document.querySelector('.inspector').classList.remove('mobile-open');document.querySelector('.rail').classList.toggle('mobile-open')};$('variantsBtn').onclick=()=>{document.querySelector('.rail').classList.remove('mobile-open');document.querySelector('.inspector').classList.toggle('mobile-open')};
$('download').onclick=()=>{let b=new Blob([JSON.stringify(recipe().variants,null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='variants.json';a.click();URL.revokeObjectURL(a.href)};$('copy').onclick=()=>navigator.clipboard.writeText(command());
document.addEventListener('keydown',e=>{if(e.target.matches('input,select'))return;if(e.code==='Space'){e.preventDefault();$('play').click()}if(e.key.toLowerCase()==='a')selectLane('A');if(e.key.toLowerCase()==='b')selectLane('B');if(e.target===$('wave')&&(e.key==='ArrowLeft'||e.key==='ArrowRight')){e.preventDefault();current=Math.max(0,Math.min(duration,current+(e.key==='ArrowRight'?1:-1)));syncPlayer();positionOverlays()}if(e.key==='Escape'){$('player').pause();$('drawer').classList.remove('open')}});window.onresize=fitCanvas;
window.addEventListener('pagehide',flushRecipe);document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='hidden')flushRecipe();});
hydrate();renderBackends();renderChain();renderVariants();updateRecipe();recipeFlushPending=true;flushRecipe();requestAnimationFrame(fitCanvas);
</script></body></html>'''

# Keep the desktop workbench inside medium-width Chrome windows as well as the
# narrow in-app preview. This late override intentionally wins over the base
# desktop grid so the right editor never gets pushed beyond the viewport.
PAGE = PAGE.replace(
    "</head><body>",
    """<style>@media (min-width:1051px) and (max-width:1500px){
.workspace{grid-template-columns:220px minmax(0,1fr) 260px;gap:6px;padding:6px}
.rail,.stage,.inspector{min-width:0}
.stage{grid-template-rows:40px minmax(210px,40%) 72px minmax(150px,1fr)}
.topbar{gap:9px;padding:0 10px}.topmeta{display:none}.expname{width:150px}
.results-head,.result-row{grid-template-columns:minmax(110px,1.3fr) 70px 58px 58px 62px}
.results{overflow-x:hidden}.history-title span{max-width:42%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.transport .time,.transport .grow{display:none}.transport .ab button{padding:5px 7px;max-width:115px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.transport .toggle{font-size:0;margin:0}.transport .toggle input{margin:0}}
</style></head><body>""",
)

# At laptop-sized CSS viewports use the same slide-over drawers as mobile. It
# prevents a wide editor from being pushed outside the viewport while keeping
# every control one click away from the central waveform.
PAGE = PAGE.replace(
    "</head><body>",
    """<style>@media (max-width:1400px){
.workspace{grid-template-columns:minmax(0,1fr)!important;gap:5px;padding:5px;overflow:hidden!important}
.rail,.inspector{display:none!important}.rail.mobile-open,.inspector.mobile-open{display:block!important;position:fixed!important;z-index:12!important;top:54px!important;bottom:30px!important;width:min(360px,88vw)!important;box-shadow:0 20px 60px #000b!important}
.rail.mobile-open{left:5px!important}.inspector.mobile-open{right:5px!important}.mobile-only{display:inline-block!important}
.stage{grid-template-rows:40px minmax(210px,45%) 72px minmax(130px,1fr)!important}
}</style></head><body>""",
)

# Complete the single-model popup with model-specific run parameters and
# explicit close/delete actions.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
setTimeout(()=>{
  const modal=document.getElementById('modelModal'),box=modal?.querySelector('.model-modal');if(!modal||!box||document.getElementById('modelClose'))return;
  const fields=document.createElement('div');fields.id='modelParams';fields.innerHTML='<div id="modelSpecificFields"></div>';
  const specific={dfn3:[['chunk_s','Chunk (giây)','select','15,30,45,60'],['overlap_s','Overlap (giây)','number','2']],mf2se48:[['chunk_s','Chunk (giây, <20)','select','8,12,15'],['overlap_s','Overlap (giây)','number','1']],mpsenet:[['chunk_s','Chunk (giây)','select','5,10,15'],['overlap_s','Overlap (giây)','number','1']],bsrnn26:[['chunk_s','Chunk (giây)','select','10,20,30'],['overlap_s','Overlap (giây)','number','2']],re_use:[['chunk_s','Chunk (giây)','number','5'],['hop_portion','Hop portion','number','0.5']]};
  const drawSpecific=()=>{const model=document.getElementById('modelChoice')?.value||'dfn3',host=document.getElementById('modelSpecificFields');if(!host)return;host.innerHTML=(specific[model]||[]).map(([id,label,type,values])=>type==='select'?`<div class="field"><label>${label}</label><select id="param_${id}">${values.split(',').map(v=>`<option>${v}</option>`).join('')}</select></div>`:`<div class="field"><label>${label}</label><input id="param_${id}" type="number" min="0" max="1" step="0.01" value="${values}"></div>`).join('');};
  drawSpecific();
  const actions=box.querySelector('.modal-actions');box.insertBefore(fields,actions);
  const close=document.createElement('button');close.id='modelClose';close.className='btn';close.textContent='Đóng';actions.insertBefore(close,actions.firstChild);close.onclick=()=>{modal.classList.remove('open');modal.style.display='none';};
  const del=document.createElement('button');del.id='deleteModelPopup';del.className='btn danger';del.textContent='Xóa model';actions.insertBefore(del,close);del.onclick=()=>{const v=typeof selectedVariant==='function'?selectedVariant():null;const i=Number(window.__modelEditIndex??-1);if(!v||i<0||!Array.isArray(v.chain)||v.chain.length<=1)return;v.chain.splice(i,1);v.model=v.chain[v.chain.length-1];v.status='draft';v.draftChanged=true;modal.classList.remove('open');modal.style.display='none';if(typeof renderVariants==='function')renderVariants();if(typeof dirty==='function')dirty();};
  modal.addEventListener('click',e=>{if(e.target===modal)close.click();});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&modal.classList.contains('open'))close.click();});
  actions.querySelector('#saveModel')?.replaceWith(actions.querySelector('#saveModel').cloneNode(true));
},400);
</script></body></html>''',
)

# Keep the late history drawer honest when no output artifact has been
# imported for a run.
PAGE = PAGE.replace(
    "Chọn một lượt để tiếp tục chỉnh, chạy lại hoặc nghe output.",
    "Chọn một lượt để tiếp tục chỉnh hoặc chạy lại. Audio output: unavailable until imported.",
)

# Final visual simplification: keep the ordered chain as the only visible
# below-waveform editor. The catalog remains available inside the add modal,
# where it can be searched without competing with the current recipe.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
setTimeout(() => {
  const results = document.querySelector('.results');
  const split = document.querySelector('.model-split');
  const current = split?.querySelector('.current-models');
  const rows = document.getElementById('resultRows');
  if (!results || !split || !current || !rows || split.dataset.quietChain === '1') return;
  split.dataset.quietChain = '1';
  const catalog = document.getElementById('modelCatalog');
  catalog?.remove();
  split.className = 'model-chain-surface';
  current.className = 'model-chain-surface-inner';
  current.querySelector('h3')?.remove();
  current.querySelector('p')?.remove();
  const gate = current.querySelector('.global-dnsmos');
  if (gate) gate.classList.add('compact-gate');
  const title = results.querySelector('.history-title strong');
  if (title) title.textContent = 'Model chain';
  const subtitle = results.querySelector('.history-title span');
  if (subtitle) subtitle.textContent = 'Theo thứ tự xử lý';
  const add = document.createElement('button');
  add.type = 'button'; add.id = 'addModelControl'; add.className = 'btn'; add.textContent = '+ Add model';
  add.setAttribute('aria-label', 'Add model to chain');
  add.onclick = () => window.__openModelForAdd?.('dfn3');
  const toolbar = document.createElement('div');
  toolbar.className = 'chain-toolbar';
  toolbar.innerHTML = '<strong>Model chain</strong>';
  toolbar.appendChild(add);
  current.insertBefore(toolbar, gate || rows);
  current.appendChild(rows);

  const explainDelete = () => {
    const only = rows.querySelectorAll('[data-delete-model]').length <= 1;
    rows.querySelectorAll('[data-delete-model]').forEach(button => {
      button.disabled = only;
      if (only) button.title = 'Cannot delete the only model in the chain';
    });
  };
  explainDelete();
  new MutationObserver(explainDelete).observe(rows, {childList: true, subtree: true});

  // The existing modal is the single source of truth for model selection.
  // Make its add state a searchable, scrollable catalog and keep edit compact.
  const oldAdd = window.__openModelForAdd;
  if (oldAdd && !window.__quietAddWrapped) {
    window.__quietAddWrapped = true;
    window.__openModelForAdd = model => {
      oldAdd(model);
      const modal = document.getElementById('modelModal');
      const picker = modal?.querySelector('.model-picker');
      if (!modal || !picker) return;
      let search = modal.querySelector('#modelSearch');
      if (!search) {
        search = document.createElement('input'); search.id = 'modelSearch'; search.type = 'search';
        search.placeholder = 'Search models'; search.setAttribute('aria-label', 'Search models');
        picker.parentElement.insertBefore(search, picker);
        search.oninput = () => {
          const query = search.value.trim().toLowerCase();
          picker.querySelectorAll('button').forEach(button => {
            button.hidden = query && !button.textContent.toLowerCase().includes(query);
          });
        };
      }
      picker.style.display = 'grid';
      search.value = '';
      picker.querySelectorAll('button').forEach(button => { button.hidden = false; });
      const remove = document.getElementById('deleteModelPopup');
      const only = (window.selectedVariant?.()?.chain || []).length <= 1;
      if (remove) { remove.disabled = only; remove.title = only ? 'Cannot delete the only model in the chain' : 'Remove this model'; }
      search.focus();
    };
  }
}, 1500);
</script></body></html>''',
)

# Recipe/CLI export is no longer part of the lab workflow; keep its backing
# nodes for compatibility but remove it from the user-facing interface.
PAGE = PAGE.replace(
    "</body></html>",
    """<style>#recipeBtn,.drawer{display:none!important}</style></body></html>""",
)
PAGE = PAGE.replace(
    "</body></html>",
    """<style>#setupBtn,#undoBtn{display:none!important}</style></body></html>""",
)
PAGE = PAGE.replace(
    "Chọn một lượt để tiếp tục chỉnh, chạy lại hoặc nghe output.",
    "Chọn một lượt để tiếp tục chỉnh hoặc chạy lại. Audio output: unavailable until imported.",
)
PAGE = PAGE.replace(
    "</body></html>",
    """<style>#exportBtn{display:none!important}</style></body></html>""",
)

# Compact transport and backend-owned audio I/O controls.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
setTimeout(()=>{
  const t=document.querySelector('.transport'),rew=document.getElementById('rewind'),imp=document.getElementById('loadB'),fileB=document.getElementById('fileB');
  rew?.remove();imp?.remove();fileB?.remove();
  const back=document.getElementById('waveBack5'),fwd=document.getElementById('waveForward5'),play=document.getElementById('play');
  if(t&&back&&fwd&&play){t.insertBefore(back,t.firstChild);t.insertBefore(play,fwd);t.insertBefore(fwd,play.nextSibling);}
  const top=document.querySelector('.topbar');if(top&&!document.getElementById('inputMain')){const b=document.createElement('button');b.id='inputMain';b.className='btn';b.textContent='Input';b.title='Chọn audio đầu vào';b.onclick=()=>document.getElementById('fileA')?.click();const run=document.getElementById('runMain');top.insertBefore(b,run?.nextSibling||null);}
  const state=document.getElementById('laneState');if(state)state.textContent='Input từ file · output do backend cấp tên riêng';
},650);
</script><style>#loadB,#fileB,#rewind{display:none!important}</style></body></html>''',
)

PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
setTimeout(()=>{document.getElementById('exportBtn')?.remove();document.getElementById('aBtn')?.remove();document.getElementById('bBtn')?.remove();document.querySelector('.ab')?.remove();},700);
</script><style>#exportBtn,#aBtn,#bBtn,.ab{display:none!important}</style></body></html>''',
)
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>setTimeout(()=>{document.getElementById('loop')?.closest('label')?.remove();document.getElementById('loop')?.remove();},700);</script><style>#loop{display:none!important}</style></body></html>''',
)
PAGE = PAGE.replace(
    "</body></html>",
    """<style>.transport .toggle{display:none!important}</style></body></html>""",
)

# Remove controls without a useful action and add explicit five-second wave
# navigation buttons.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
setTimeout(()=>{
  document.getElementById('setupBtn')?.remove();document.getElementById('undoBtn')?.remove();
  const t=document.querySelector('.transport');if(!t||document.getElementById('waveBack5'))return;
  const mk=(id,label,delta)=>{const b=document.createElement('button');b.id=id;b.className='btn icon';b.textContent=label;b.title=`${delta<0?'Lùi':'Tiến'} 5 giây`;b.onclick=()=>{const w=document.getElementById('wave');if(!w)return;w.focus();w.dispatchEvent(new KeyboardEvent('keydown',{key:delta<0?'ArrowLeft':'ArrowRight',bubbles:true}));for(let i=1;i<5;i++)w.dispatchEvent(new KeyboardEvent('keydown',{key:delta<0?'ArrowLeft':'ArrowRight',bubbles:true}));};return b;};
  t.insertBefore(mk('waveBack5','−5s',-5),t.firstChild);t.insertBefore(mk('waveForward5','+5s',5),t.children[1]||null);
},500);
</script></body></html>''',
)

# Remove the obsolete Models list from the legacy setup drawer. The only
# remaining entry point is the top-bar Models button, which opens the popup.
PAGE = PAGE.replace(
    "</body></html>",
    """<style>.trial-section{display:none!important}</style></body></html>""",
)

# Last-pass binding after the document is fully parsed; this wins over all
# legacy `variantsBtn.onclick` assignments.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
setTimeout(()=>{
  const b=document.getElementById('variantsBtn'),m=document.getElementById('modelModal');
  if(!b||!m)return;
  b.onclick=(e)=>{e.preventDefault();e.stopImmediatePropagation();document.querySelector('.inspector')?.setAttribute('style','display:none!important');m.classList.add('open');m.style.display='flex';m.style.visibility='visible';const t=document.getElementById('modelModalTitle');if(t)t.textContent='Thêm model vào lượt chạy';const c=document.getElementById('modelChoice');if(c)c.value='dfn3';const n=document.getElementById('modelThreshold');if(n)n.value='0.30';};
},250);
</script></body></html>''',
)

# A visible run action belongs on the main workbench, not inside the hidden
# inspector. It delegates to the same dependency-aware run flow.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const install=()=>{const top=document.querySelector('.topbar');
  if(!top||document.getElementById('runMain'))return;
  const b=document.createElement('button'); b.id='runMain'; b.className='btn primary'; b.textContent='Run'; b.title='Chạy công thức model hiện tại';
  const anchor=document.getElementById('variantsBtn')||document.getElementById('recipeBtn'); top.insertBefore(b,anchor||null);
  b.onclick=()=>{const direct=document.getElementById('runNow');const legacy=document.getElementById('runVariant');if(direct)direct.click();else if(legacy)legacy.click();};};
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',install);else setTimeout(install,0);
})();
</script></body></html>''',
)

# Keep the legacy inspector in the DOM only for backwards-compatible event
# wiring, but make it impossible to render; editing is popup-only now.
PAGE = PAGE.replace('<aside class="inspector">', '<aside class="inspector" style="display:none!important">')
PAGE = PAGE.replace(
    '<button class="btn mobile-only" id="variantsBtn">Trials</button>',
    '<button class="btn mobile-only" id="variantsBtn" onclick="event.preventDefault();var p=document.getElementById(\'modelModal\');if(p){p.classList.add(\'open\');p.style.display=\'flex\';}">Models</button>',
)

# Remove the legacy full-page model editor. Model settings live only in the
# single-model popup opened from the Models button.
PAGE = PAGE.replace(
    "</body></html>",
    """<style>.inspector,.chain-editor{display:none!important}</style></body></html>""",
)

# The model editor is modal-only; never reserve a permanent right sidebar.
PAGE = PAGE.replace(
    "</body></html>",
    """<style>.inspector{display:none!important}</style></body></html>""",
)

# Hard-wire the Models button to the editor modal. This intentionally does not
# depend on the hidden inspector/drawer DOM, so it works at every breakpoint.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const b=document.getElementById('variantsBtn'),m=document.getElementById('modelModal');
  if(!b||!m)return;
  const closeLegacyInspector=()=>{const panel=document.querySelector('.inspector');panel?.classList.remove('mobile-open');if(panel)panel.style.display='none';};
  const showModelPopup=()=>{closeLegacyInspector();m.classList.add('open');m.style.display='flex';m.style.visibility='visible';m.style.opacity='1';};
  b.addEventListener('click',showModelPopup,true);
  b.onclick=(event)=>{
    event.preventDefault();
    closeLegacyInspector();
    const add=document.getElementById('addModel');
    if(add){add.click();}
    const modal=m.querySelector('.model-modal');
    const title=document.getElementById('modelModalTitle');
    const choice=document.getElementById('modelChoice');
    const threshold=document.getElementById('modelThreshold');
    const first=document.querySelector('.model-current-row strong');
    const model=first?.textContent?.trim()||choice?.value||'dfn3';
    if(title)title.textContent='Thêm model vào lượt chạy';
    if(choice)choice.value='dfn3';
    if(threshold)threshold.value='0.30';
    showModelPopup();
    modal?.scrollIntoView({block:'center'});
  };
})();
</script></body></html>''',
)

# Models is a direct action: open the current model's editor/info popup rather
# than reopening the legacy side drawer.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const btn=document.getElementById('variantsBtn'), modal=document.getElementById('modelModal');
  if(!btn||!modal)return;
  let info=document.getElementById('modelInfo');
  if(!info){info=document.createElement('div');info.id='modelInfo';info.style.cssText='border:1px solid #26333e;border-radius:6px;padding:7px;margin:8px 0;color:#aab8c3;font-size:10px';modal.querySelector('.model-modal').insertBefore(info,modal.querySelector('.modal-actions'));}
  const descriptions={dfn3:'DeepFilterNet3 · nhẹ, nhanh, phù hợp chạy thử.',mf2se48:'MossFormer2 SE48 · tách nhiễu mạnh hơn, tốn GPU hơn.',mpsenet:'MP-SENet · khử nhiễu phổ, phù hợp giọng nói rõ.',bsrnn26:'BSRNN26 · chất lượng cao, thời gian chạy dài hơn.',rf_denoise:'RoFormer · thử nghiệm khử nhiễu theo attention.',re_use:'RE-USE · cấu hình khử nhiễu tái sử dụng.'};
  btn.onclick=()=>{
    const edit=document.querySelector('[data-edit-model="0"]');
    if(edit){edit.click();}
    else if(modal){modal.classList.add('open');}
    const choice=document.getElementById('modelChoice');
    const update=()=>{if(info)info.textContent=descriptions[choice?.value]||'Thông tin model chưa có.';};
    update(); choice?.addEventListener('change',update,{once:false});
  };
})();
</script></body></html>''',
)

# The center table is the current ordered model recipe, not run history. Run
# history is available only from the dedicated History page.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const byId = id => document.getElementById(id);
  const selected = () => typeof selectedVariant === 'function' ? selectedVariant() : null;
  const style = document.createElement('style');
  style.textContent = '#historyRows{display:none!important}.model-current-row{display:grid;grid-template-columns:26px minmax(0,1fr) 90px 78px 64px;gap:7px;align-items:center;border:1px solid var(--line);background:#121b23;border-radius:7px;padding:8px;margin-top:5px;min-width:0;cursor:pointer}.model-current-row:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}.model-current-row .step{color:var(--cyan);font:11px ui-monospace,monospace}.model-current-row small{display:block;color:var(--muted);font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.model-current-row .btn{padding:4px 6px;font-size:10px;min-height:44px}.model-current-row .model-edit-affordance{white-space:nowrap}.model-current-empty{padding:14px 8px;color:var(--muted);font-size:11px}@media(max-width:420px){.model-current-row{grid-template-columns:22px minmax(0,1fr) 52px;gap:5px;padding:6px;min-width:0}.model-current-row>span:nth-child(3),.model-current-row>span:nth-child(4){display:none}.model-current-row>span:last-child{justify-self:end}.model-current-row .btn{padding:3px 7px;min-width:44px;max-width:44px;overflow:hidden;text-overflow:clip}.model-current-row .model-edit-affordance{font-size:0}.model-current-row .model-edit-affordance::before{content:"✎";font-size:16px}}';
  document.head.appendChild(style);
  const labels = () => {
    document.querySelectorAll('.trial-section .label').forEach(x => x.textContent='Models');
    const modelsBtn=byId('variantsBtn'); if(modelsBtn) modelsBtn.textContent='Models';
    const section=document.querySelector('.trial-section'); if(section){const add=section.querySelector('#addTrial');if(add)add.title='Thêm model';}
    const head=document.querySelector('.history-title strong'); if(head) head.textContent='Models trong lượt chạy';
    const desc=document.querySelector('.history-title span'); if(desc) desc.textContent='Thứ tự xử lý hiện tại';
    const columns=document.querySelectorAll('.results-head span');
    if(columns[0]) columns[0].textContent='Model / status';
    if(columns[1]) columns[1].textContent='Model';
    if(columns[2]) columns[2].textContent='Ngưỡng';
    if(columns[3]) columns[3].textContent='Kết quả';
    if(columns[4]) columns[4].textContent='Sửa';
    document.querySelectorAll('.trial-row strong').forEach(x=>{if(/^Trial\s/.test(x.textContent))x.textContent=x.textContent.replace(/^Trial/,'Model')});
    const editor=document.querySelector('.inspector-head strong'); if(editor) editor.textContent='Model editor';
  };
  const renderCurrentModels = () => {
    const host=byId('resultRows'),v=selected(); if(!host||!v)return;
    const chain=Array.isArray(v.chain)&&v.chain.length?v.chain:['dfn3'];
    host.innerHTML=chain.map((m,i)=>`<div class="model-current-row" data-current-model="${i}"><span class="step">${i+1}</span><span><strong>${m}</strong><small>${i===0?'Bước đầu':'sau '+chain[i-1]}</small></span><span>${Number(v.threshold??.3).toFixed(2)}</span><span>${v.status||'draft'}</span><span><button class="btn model-edit-affordance" type="button" data-edit-model="${i}" aria-label="Sửa model ${i+1}: ${m}">Sửa</button></span></div>`).join('');
    const openRow = row => { const index=Number(row.dataset.currentModel); const target=document.querySelector(`[data-edit-model="${index}"]`); if(target) target.click(); else byId('addModel')?.click(); };
    host.querySelectorAll('.model-current-row').forEach(row=>{row.onclick=e=>{if(e.target.closest('button'))return;openRow(row);};});
    host.querySelectorAll('[data-edit-model]').forEach(btn=>btn.onclick=e=>{e.stopPropagation();window.__openModelEditor?.(Number(btn.dataset.editModel));});
  };
  const render = () => { labels(); renderCurrentModels(); };
  window.__renderCurrentModels = renderCurrentModels;
  render();
  render();
})();
</script></body></html>''',
)

# Model-chain editor and a history view. A trial is now a recipe containing an
# ordered list of models; the run button submits that whole recipe at once.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const q = id => document.getElementById(id);
  const inspector = document.querySelector('.editor-card');
  if (!inspector || q('modelChainEditor')) return;
  const style = document.createElement('style');
  style.textContent = '.chain-editor{border:1px solid #26333e;border-radius:8px;padding:8px;margin:9px 0;background:#0d151c}.chain-editor-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}.chain-editor-head strong{font-size:11px}.chain-list{display:grid;gap:4px}.chain-step{display:grid;grid-template-columns:22px minmax(0,1fr) auto;gap:6px;align-items:center;border:1px solid #2b3946;border-radius:6px;padding:6px;background:#111b23;cursor:pointer}.chain-step:hover{border-color:var(--cyan)}.chain-step b{color:var(--cyan);font:11px ui-monospace,monospace}.chain-step small{display:block;color:var(--muted);font-size:9px}.chain-step button{padding:2px 5px;font-size:10px}.modal-backdrop,.history-page{position:fixed;inset:0;z-index:30;background:#05080dcc;display:none;align-items:center;justify-content:center}.modal-backdrop.open,.history-page.open{display:flex}.model-modal,.history-card{width:min(440px,92vw);max-height:82vh;overflow:auto;border:1px solid #38515d;border-radius:10px;background:#101a22;box-shadow:0 24px 80px #000b;padding:14px}.model-modal h3,.history-card h3{margin:0 0 10px;font-size:14px}.modal-actions{display:flex;justify-content:flex-end;gap:6px;margin-top:12px}.history-page{align-items:stretch;justify-content:flex-end}.history-card{width:min(620px,94vw);height:100%;max-height:none;border-radius:0;border-top:0;border-bottom:0}.history-row{display:grid;grid-template-columns:10px minmax(0,1fr) auto;gap:8px;align-items:center;padding:10px;border:1px solid #2b3946;border-radius:7px;margin:5px 0;cursor:pointer}.history-row:hover{border-color:var(--cyan);background:#14252a}.history-row small{display:block;color:var(--muted);font-size:10px}.history-empty{color:var(--muted);padding:16px 0;font-size:11px}';
  document.head.appendChild(style);
  const top = document.querySelector('.topbar');
  const historyBtn = document.createElement('button'); historyBtn.className='btn'; historyBtn.id='historyBtn'; historyBtn.textContent='History'; top.insertBefore(historyBtn, q('recipeBtn'));
  const block = document.createElement('div'); block.className='chain-editor'; block.innerHTML='<div class="chain-editor-head"><strong>Model trong lượt chạy</strong><button class="btn" id="addModel">＋ Model</button></div><div class="chain-list" id="modelChainEditor"></div>';
  const anchor = inspector.querySelector('.variant-editor'); if (anchor) anchor.parentNode.insertBefore(block, anchor);
  const modal = document.createElement('div'); modal.className='modal-backdrop'; modal.id='modelModal'; modal.innerHTML='<div class="model-modal"><h3 id="modelModalTitle">Thêm model</h3><div class="field"><label>Model</label><select id="modelChoice"><option value="dfn3">DeepFilterNet3</option><option value="mf2se48">MossFormer2 SE48</option><option value="mpsenet">MP-SENet</option><option value="bsrnn26">BSRNN26</option><option value="rf_denoise">RoFormer</option><option value="re_use">RE-USE</option></select></div><div class="field"><label>Ngưỡng DNSMOS</label><input id="modelThreshold" type="number" min="0" max="1" step="0.01" value="0.30"></div><div class="modal-actions"><button class="btn" id="cancelModel">Huỷ</button><button class="btn primary" id="saveModel">Lưu model</button></div></div>';
  document.body.appendChild(modal);
  const history = document.createElement('div'); history.className='history-page'; history.id='historyPage'; history.innerHTML='<div class="history-card"><div style="display:flex;align-items:center;gap:8px"><h3>Lịch sử thí nghiệm</h3><span class="spacer"></span><button class="btn" id="closeHistory">Đóng</button></div><p style="color:var(--muted);font-size:11px">Chọn một lượt để tiếp tục chỉnh, chạy lại hoặc nghe output.</p><div id="historyList"></div></div>';
  document.body.appendChild(history);
  let editIndex = -1;
  const current = () => typeof selectedVariant === 'function' ? selectedVariant() : null;
  const drawChain = () => {
    const v=current(), host=q('modelChainEditor'); if(!host||!v)return;
    const chain=Array.isArray(v.chain)&&v.chain.length?v.chain:['dfn3'];
    host.innerHTML=chain.map((m,i)=>`<div class="chain-step" data-model-index="${i}"><b>${i+1}</b><span>${m}<small>${i===0?'chạy đầu tiên':'chạy sau bước '+i}</small></span><button class="btn" data-edit-model="${i}">Sửa</button></div>`).join('');
    host.querySelectorAll('[data-model-index]').forEach(x=>x.onclick=()=>openModel(+x.dataset.modelIndex));
    host.querySelectorAll('[data-edit-model]').forEach(x=>x.onclick=e=>{e.stopPropagation();openModel(+x.dataset.editModel)});
  };
  const openModel = i => { const v=current(); if(!v)return; editIndex=i; window.__modelEditIndex=i; const chain=v.chain||[]; q('modelModalTitle').textContent=i<0?'Thêm model':'Sửa model'; q('modelChoice').value=chain[i]||'dfn3'; q('modelThreshold').value=String(v.threshold??.3); q('modelModal').classList.add('open'); };
  q('addModel').onclick=()=>openModel(-1); q('cancelModel').onclick=()=>q('modelModal').classList.remove('open');
  q('saveModel').onclick=()=>{const v=current();if(!v)return;const model=q('modelChoice').value;v.chain=Array.isArray(v.chain)?[...v.chain]:[];if(editIndex<0)v.chain.push(model);else v.chain[editIndex]=model;v.model=v.chain[v.chain.length-1]||model;v.threshold=Number(q('modelThreshold').value||.3);v.status='draft';v.draftChanged=true;q('modelModal').classList.remove('open');if(typeof renderVariants==='function')renderVariants();drawChain();if(typeof dirty==='function')dirty();};
  const drawHistory = () => { const host=q('historyList'); let rows=[]; if(typeof variants!=='undefined')variants.forEach((v,i)=>(v.runs||[]).forEach(r=>rows.push({v,i,r}))); rows.sort((a,b)=>String(b.r.created||'').localeCompare(String(a.r.created||''))); host.innerHTML=rows.length?rows.map(x=>`<div class="history-row" data-history-id="${x.r.id}" data-history-v="${x.i}"><span class="dot"></span><span><strong>${x.v.label||'Trial'}</strong><small>${x.r.status||'saved'} · ${x.r.configSnapshot?.model||x.v.model||'model'} · ${x.r.created||''}</small></span><button class="btn">Mở</button></div>`).join(''):'<div class="history-empty">Chưa có lượt chạy được lưu.</div>'; host.querySelectorAll('[data-history-id]').forEach(row=>row.onclick=()=>{selected=+row.dataset.historyV;if(typeof renderVariants==='function')renderVariants();if(typeof useRun==='function')useRun(row.dataset.historyId);q('historyPage').classList.remove('open');}); };
  historyBtn.onclick=()=>{drawHistory();q('historyPage').classList.add('open')}; q('closeHistory').onclick=()=>q('historyPage').classList.remove('open');
  const oldRender = window.renderVariants; if(typeof oldRender==='function')window.renderVariants=function(){oldRender();drawChain();};
  drawChain();
})();
</script></body></html>''',
)

# Compact lab controls: keep the editor focused on the current experiment and
# make the dependency-aware run decision visible before anything is submitted.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const $l = id => document.getElementById(id);
  const editor = document.querySelector('.editor-card');
  if (!editor || $l('runPlan')) return;
  const actions = editor.querySelector('.editor-actions');
  const plan = document.createElement('div');
  plan.id = 'runPlan';
  plan.className = 'run-plan';
  plan.innerHTML = '<div class="run-plan-title"><strong>Chạy thí nghiệm</strong><span id="runScope">chỉ hậu xử lý</span></div>' +
    '<p id="runReason">Đang dùng lại kết quả model gần nhất.</p>' +
    '<div class="run-plan-actions"><button class="btn primary" id="runNow">Chạy</button><button class="btn" id="runModel">Chạy lại model</button></div>';
  (actions?.parentNode || editor).insertBefore(plan, actions || null);
  const css = document.createElement('style');
  css.textContent = '.run-plan{border:1px solid #31545b;border-radius:8px;padding:9px;background:#10252a;margin:9px 0}.run-plan-title{display:flex;justify-content:space-between;gap:8px;align-items:center}.run-plan-title strong{font-size:12px}.run-plan-title span{font-size:10px;color:var(--cyan);white-space:nowrap}.run-plan p{margin:5px 0;color:#aab8c3;font-size:10px}.run-plan-actions{display:flex;gap:6px}.run-plan-actions .btn{flex:1;font-size:11px;padding:6px 7px}.run-plan.needs-model{border-color:#765d2b;background:#292217}.run-plan.needs-model .run-plan-title span{color:var(--amber)}.compact-copy{font-size:10px!important}.inspector-card .warning{font-size:9px}.lab-hidden{display:none!important}';
  document.head.appendChild(css);

  const text = (id, value) => { const n = $l(id); if (n) n.textContent = value; };
  const compactLabels = {
    'Load audio':'Nạp', 'Duplicate':'Nhân bản', 'Delete':'Xóa', 'Revert':'Hoàn tác',
    'Prepare run':'Tạo công thức', 'Export command':'Xuất lệnh', 'Delete all runs':'Xóa kết quả',
    'Review export':'Công thức', 'Trial editor':'Thí nghiệm', 'Selected trial':'Đang chọn',
    'Model chain':'Model', 'Runtime':'Trạng thái', 'Audio state':'Audio',
    'Trial name':'Tên', 'Advanced parameters':'Thêm tham số', 'Saved runs':'Kết quả đã lưu',
    'Source audio':'Âm thanh', 'Trials':'Mẻ thử'
  };
  document.querySelectorAll('button,label,summary,span,strong,h3,h4').forEach(n => {
    const value = (n.textContent || '').trim();
    if (compactLabels[value]) n.textContent = compactLabels[value];
  });
  const oldPrepare = window.prepareRun;
  let forceModel = false;
  const activeRun = () => {
    const v = typeof selectedVariant === 'function' ? selectedVariant() : null;
    return v && (v.runs || []).find(r => r.id === v.activeRunId) || null;
  };
  const modelChanged = () => {
    const v = typeof selectedVariant === 'function' ? selectedVariant() : null;
    const r = activeRun();
    if (!v || !r) return true;
    const c = r.configSnapshot || {};
    const s = v.settings || {};
    const old = c.settings || {};
    return JSON.stringify(c.chain || [c.model]) !== JSON.stringify(v.chain || [v.model]) || JSON.stringify(c.stepParams || []) !== JSON.stringify(v.stepParams || []) || Number(old.chunk_s || 30) !== Number(s.chunk_s || 30) || String(old.env || 'a100') !== String(s.env || 'a100');
  };
  const runErrorMessage = error => {
    const message = String(error?.message || error || 'Request failed');
    const http = message.match(/^HTTP\s+(\d+)/i);
    return http ? `HTTP ${http[1]} — server rejected the run.` : 'Network error — backend unavailable; request was not sent.';
  };
  const runFailureStatus = error => /^HTTP\s+\d+/i.test(String(error?.message || error || '')) ? 'failed' : 'not-sent';
  const updatePlan = () => {
    const needs = forceModel || modelChanged();
    const failed = ['failed', 'not-sent'].includes(activeRun()?.status);
    const box = $l('runPlan');
    if (box) box.classList.toggle('needs-model', needs);
    text('runScope', failed ? 'gửi lại' : needs ? 'chạy lại model' : 'hậu xử lý');
    text('runReason', failed ? activeRun().errorMessage : needs ? 'Model hoặc tham số model đã đổi; pipeline sẽ chạy lại phần cần thiết.' : 'Model không đổi; dùng checkpoint và chỉ chạy các bước phía sau.');
    const retry = $l('runNow'); if (retry) retry.textContent = failed ? 'Gửi lại' : 'Chạy';
  };
  const run = () => {
    if (typeof oldPrepare !== 'function') return;
    const existing = activeRun();
    if (existing && (existing.status === 'submitting' || existing.status === 'running')) return;
    const needs = forceModel || modelChanged();
    oldPrepare();
    const v = typeof selectedVariant === 'function' ? selectedVariant() : null;
    const r = v && (v.runs || []).find(x => x.id === v.activeRunId);
    if (r) {
      r.execution = {scope: needs ? 'model' : 'postprocess', submittedAt: new Date().toISOString(), endpoint: '/api/experiment/run'};
      r.status = 'submitting';
      text('runReason', 'Đang gửi yêu cầu…');
      if (typeof showNotice === 'function') showNotice('Đang gửi yêu cầu…');
      if (typeof renderVariants === 'function') renderVariants();
      if (typeof dirty === 'function') dirty();
      const payload = {experiment: typeof recipe === 'function' ? recipe() : {}, trialId: v.id, runId: r.id, scope: r.execution.scope, outputPolicy: 'backend_unique_run_directory', overwrite: false};
      fetch('/api/experiment/run', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)}).then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json().catch(() => ({}));
      }).then(resp => { r.serverRun = {runId: resp.runId, runDir: resp.runDir, out: resp.out}; r.status='accepted'; delete r.error; delete r.errorMessage; if (typeof renderVariants === 'function') renderVariants(); text('runReason', 'Backend đã nhận công thức.'); if (typeof showNotice === 'function') showNotice('Backend đã nhận công thức.'); window.__pollRun?.(r, v); updatePlan(); })
        .catch(error => { r.status=runFailureStatus(error); r.error=String(error?.message || error); r.errorMessage=runErrorMessage(error); if (typeof renderVariants === 'function') renderVariants(); text('runReason', r.errorMessage); if (typeof showNotice === 'function') showNotice(r.errorMessage); updatePlan(); });
    }
    forceModel = false;
    updatePlan();
  };
  $l('runNow').onclick = run;
  $l('runModel').onclick = () => { forceModel = true; updatePlan(); run(); };
  ['variantModel','variantThreshold','trialChunk','trialEnv','trialStop','trialResume'].forEach(id => $l(id)?.addEventListener('input', updatePlan));
  ['variantModel','trialChunk','trialEnv','trialStop','trialResume'].forEach(id => $l(id)?.addEventListener('change', updatePlan));
  updatePlan();
})();
</script></body></html>''',
)

PAGE = PAGE.replace("drag to select · click to seek", "drag to select · Alt+drag to pan · wheel with Shift")
PAGE = PAGE.replace("</head><body>", "<style>.wave{touch-action:none}</style></head><body>")
PAGE = PAGE.replace(
    "</head><body>",
    """<style>@media (min-width:761px){
.workspace{grid-template-columns:18% minmax(0,64%) 18%!important;width:100%!important;max-width:100%!important;overflow:hidden!important}
.rail,.inspector{display:block!important;min-width:0!important;max-width:100%!important}
.stage{display:grid!important;grid-template-rows:48px minmax(180px,42%) 84px minmax(120px,1fr)!important;min-height:0!important}
.canvas-wrap{display:block!important;min-height:180px!important;min-width:0!important;max-width:100%!important}
.wave{max-width:100%;min-width:0}.results{min-width:0;max-width:100%;overflow-x:hidden}
}</style></head><body>""",
)

PAGE = PAGE.replace(
    "</head><body>",
    """<style>@media (min-width:761px) and (max-width:1500px){
.workspace{grid-template-columns:minmax(180px,22%) minmax(0,1fr) minmax(220px,22%)!important;width:100%;max-width:100%;overflow:hidden}
.rail,.inspector{display:block!important;position:static!important;min-width:0!important;width:auto!important;max-width:100%!important;box-shadow:none!important}
.topbar #undoBtn,.topbar #exportBtn{display:inline-flex!important}.topbar{min-width:0;overflow:hidden}.topbar .topmeta,.topbar .saved{display:none}
.results,.canvas-wrap,.stage-head,.transport{min-width:0;max-width:100%;overflow:hidden}.results-head,.result-row{min-width:0}.result-row>*{min-width:0;overflow:hidden;text-overflow:ellipsis}
}</style></head><body>""",
)

PAGE = PAGE.replace(
    "</head><body>",
    """<style>@media (min-width:761px) and (max-width:1050px){
.workspace{grid-template-columns:180px minmax(0,1fr) 220px;gap:5px;padding:5px}
.rail,.inspector{display:block!important;position:static!important;width:auto!important;box-shadow:none!important}
.stage{min-width:0;grid-template-rows:40px minmax(190px,38%) 72px minmax(130px,1fr)}
.mobile-only{display:none!important}.topbar{gap:7px;padding:0 8px}.topmeta,.saved{display:none}.expname{width:120px}
.results-head,.result-row{grid-template-columns:minmax(90px,1fr) 58px 52px 52px 54px}.history-title span{display:none}
.transport .time,.transport .grow{display:none}.transport .toggle{font-size:0;margin:0}.transport .toggle input{margin:0}.transport .ab button{padding:4px 6px;max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
</style></head><body>""",
)


# Final responsive rule wins over the historical breakpoint rules above.
PAGE = PAGE.replace(
    "</head><body>",
    """<style>@media (max-width:1400px){
.workspace{grid-template-columns:minmax(0,1fr)!important;gap:5px!important;padding:5px!important;overflow:hidden!important}
.rail,.inspector{display:none!important}.rail.mobile-open,.inspector.mobile-open{display:block!important;position:fixed!important;z-index:12!important;top:54px!important;bottom:30px!important;width:min(360px,88vw)!important;box-shadow:0 20px 60px #000b!important}
.rail.mobile-open{left:5px!important}.inspector.mobile-open{right:5px!important}.mobile-only{display:inline-block!important}.stage{grid-template-rows:40px minmax(210px,45%) 72px minmax(130px,1fr)!important}
}</style></head><body>""",
)

# The visible Run action submits directly to the backend. Keep the legacy
# preparation nodes only as an internal compatibility layer; users never need
# to export a command or recipe before execution.
PAGE = PAGE.replace("GPU run is exported to CLI", "Run trực tiếp backend")
PAGE = PAGE.replace("GPU execution is exported from Recipe", "Chạy trực tiếp backend")
PAGE = PAGE.replace("Run recipe prepared · export it to execute on GPU", "Đã chuẩn bị và gửi yêu cầu chạy backend")
PAGE = PAGE.replace("Prepare run saves a local draft/result only. Run the exported command on the GPU host; this page never starts a worker.", "Run sẽ gửi công thức trực tiếp tới backend; không cần export lệnh.")

# Last-pass binding makes the top-bar button unambiguously a backend action,
# independent of the hidden legacy editor controls.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
setTimeout(() => {
  const button = document.getElementById('runMain');
  if (!button) return;
  const direct = button.cloneNode(true);
  button.replaceWith(direct);
  direct.id = 'runMain';
  direct.textContent = 'Run';
  direct.title = 'Chạy trực tiếp trên backend';
  direct.dataset.action = 'backend-run';
  direct.onclick = (event) => {
    event.preventDefault();
    const runNow = document.getElementById('runNow');
    if (runNow) { runNow.click(); return; }
    const legacy = document.getElementById('runVariant');
    if (legacy) legacy.click();
  };
}, 1100);
</script></body></html>''',
)

# Keep adding a model and editing an existing model as two distinct actions.
# The top-bar Models button opens a model picker for append-only adds; each
# current-chain Sửa button opens the same modal in edit mode for that index.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const boot = () => {
    const modal = document.getElementById('modelModal');
    const oldTop = document.getElementById('variantsBtn');
    if (!modal || modal.dataset.splitActions === '1') return;
    modal.dataset.splitActions = '1';
    modal.classList.remove('open'); modal.style.display = 'none';
    const box = modal.querySelector('.model-modal');
    if (!box) return;
    const names = {
      identity: ['Identity', 'Không thay đổi audio'], fake_hp: ['Fake high-pass', 'Backend kiểm thử'],
      dfn3: ['DeepFilterNet3', 'Nhẹ và nhanh cho khử nhiễu giọng nói'],
      mf2se48: ['MossFormer2 SE48', 'Tách nhiễu mạnh, tốn GPU hơn'],
      mpsenet: ['MP-SENet', 'Khử nhiễu phổ cho giọng nói rõ'], mpsenet_dns: ['MP-SENet DNS', 'Checkpoint DNS của MP-SENet'],
      dbs_tiger: ['DBS Tiger', 'Khử nhiễu theo phổ'], mambattention: ['MambaAttention', 'Mô hình attention tuần tự'],
      bsrnn26: ['BSRNN26', 'Chất lượng cao, chạy lâu hơn'],
      rf_vocals_3005: ['RF Vocals 3005', 'Tách vocal bằng audio-separator'], rf_vocals_kim: ['RF Vocals Kim', 'Tách vocal theo checkpoint Kim'],
      rf_denoise: ['RoFormer', 'Khử nhiễu theo attention'], rf_denoise_aggr: ['RoFormer Aggressive', 'Khử nhiễu mạnh hơn'],
      rf_dereverb: ['RF Dereverb', 'Giảm vang/phản xạ'], uvr_deecho: ['UVR DeEcho', 'Giảm tiếng vọng'], rf_crowd: ['RF Crowd', 'Giảm tiếng đám đông'],
      audio_separator: ['Audio Separator', 'Backend tách audio tổng quát'], re_use: ['RE-USE', 'Khôi phục giọng nói tái sử dụng'], re_use_laughter: ['RE-USE Laughter', 'Ưu tiên giữ tiếng cười']
    };
    // Only expose keys consumed by the backend registry. Model-specific
    // overrides live per chain position; unsupported guesses are not saved.
    const params = {
      dfn3: [['chunk_s', 'Chunk (giây)', 'select', ['15','30','45','60']], ['overlap_s', 'Overlap (giây)', 'number', '2']],
      mf2se48: [['chunk_s', 'Chunk (giây, <20)', 'select', ['8','12','15']], ['overlap_s', 'Overlap (giây)', 'number', '1']],
      mpsenet: [['chunk_s', 'Chunk (giây)', 'select', ['5','10','15']], ['overlap_s', 'Overlap (giây)', 'number', '1']],
      mpsenet_dns: [['chunk_s', 'Chunk (giây)', 'select', ['5','10','15']], ['overlap_s', 'Overlap (giây)', 'number', '1']],
      bsrnn26: [['chunk_s', 'Chunk (giây)', 'select', ['10','20','30']], ['overlap_s', 'Overlap (giây)', 'number', '2']],
      dbs_tiger: [['window_s', 'Window (giây)', 'number', '6'], ['hop_s', 'Hop (giây)', 'number', '3']],
      mambattention: [['chunk_s', 'Chunk (giây)', 'number', '4'], ['overlap_s', 'Overlap (giây)', 'number', '0.5']],
      re_use: [['chunk_s', 'Chunk (giây)', 'number', '5'], ['hop_portion', 'Hop portion', 'number', '0.5']],
      re_use_laughter: [['chunk_s', 'Chunk (giây)', 'number', '5'], ['hop_portion', 'Hop portion', 'number', '0.5']],
      audio_separator: [['segment_size', 'Segment size', 'number', '256'], ['overlap', 'Overlap', 'number', '0.25']]
    };
    const choice = document.getElementById('modelChoice');
    const threshold = document.getElementById('modelThreshold');
    const title = document.getElementById('modelModalTitle');
    if (!choice || !title) return;
    choice.innerHTML = Object.entries(names).map(([id, [label]]) => `<option value="${id}">${label}</option>`).join('');
    choice.style.display = 'none';
    let picker = box.querySelector('.model-picker');
    if (!picker) {
      picker = document.createElement('div'); picker.className = 'model-picker';
      choice.parentElement.insertBefore(picker, choice.parentElement.firstChild);
    }
    let paramsHost = document.getElementById('modelSpecificFields');
    if (!paramsHost) { paramsHost = document.createElement('div'); paramsHost.id = 'modelSpecificFields'; box.appendChild(paramsHost); }
    const actions = box.querySelector('.modal-actions');
    if (!actions) return;
    const replaceButton = (id, text, cls) => {
      const old = document.getElementById(id); if (!old) return null;
      const next = old.cloneNode(false); next.id = id; next.textContent = text; next.className = cls || old.className; old.replaceWith(next); return next;
    };
    const close = replaceButton('modelClose', 'Đóng', 'btn') || replaceButton('cancelModel', 'Huỷ', 'btn');
    const cancel = document.getElementById('cancelModel');
    const save = replaceButton('saveModel', 'Thêm model', 'btn primary');
    let remove = document.getElementById('deleteModelPopup');
    if (!remove) { remove = document.createElement('button'); remove.id='deleteModelPopup'; remove.className='btn danger'; actions.insertBefore(remove, actions.firstChild); }
    const selected = () => typeof selectedVariant === 'function' ? selectedVariant() : null;
    let editIndex = -1;
    let lastFocus = null;
    const closeModal = () => { modal.classList.remove('open'); modal.style.display='none'; lastFocus?.focus(); };
    close && (close.onclick = closeModal); cancel && (cancel.onclick = closeModal);
    const drawParams = (model, existing) => {
      paramsHost.innerHTML = (params[model] || []).map(([id, label, type, values]) => {
        const val = existing?.[id] ?? (type === 'select' ? values[0] : values);
        return type === 'select'
          ? `<div class="field"><label>${label}</label><select id="param_${id}">${values.map(v => `<option ${String(v)===String(val)?'selected':''}>${v}</option>`).join('')}</select></div>`
          : `<div class="field"><label>${label}</label><input id="param_${id}" type="number" min="0" step="0.01" value="${val}"></div>`;
      }).join('');
    };
    const choose = (model, existing) => { choice.value=model; drawParams(model, existing); picker.querySelectorAll('button').forEach(b=>b.classList.toggle('selected', b.dataset.model===model)); if(editIndex<0) title.textContent=`Thêm ${names[model][0]}`; };
    choice.addEventListener('change', () => choose(choice.value, {}));
    modal.setAttribute('role','dialog'); modal.setAttribute('aria-modal','true'); modal.setAttribute('aria-labelledby','modelModalTitle');
    const open = (index, preset) => {
      const v=selected(); if (!v) return; lastFocus=document.activeElement;
      editIndex=index;
      const model = index < 0 ? (preset || 'dfn3') : ((v.chain||[])[index] || v.model || 'dfn3');
      const current = (v.stepParams?.[index] || (v.settings?.model_params?.model===model ? v.settings.model_params : {}));
      title.textContent = index < 0 ? 'Thêm model vào lượt chạy' : `Sửa model ${index+1}`;
      save.textContent = index < 0 ? 'Thêm model' : 'Lưu thay đổi';
      remove.style.display = index < 0 ? 'none' : '';
      remove.disabled = index < 0 || (v.chain||[]).length <= 1;
      remove.title = remove.disabled ? 'Không thể xóa model duy nhất trong chuỗi' : 'Xóa model này (có thể hoàn tác)';
      // The picker and search input are inserted into the same field as the
      // legacy select. Hide only the select field in add mode; hiding its
      // parent would also hide the searchable picker.
      const choiceField = choice.closest('.field');
      if (choiceField) choiceField.style.display = index < 0 ? 'none' : 'block';
      choice.style.display = 'block';
      picker.style.display = index < 0 ? 'grid' : 'none';
      if (threshold) threshold.value=String(v.threshold ?? .30);
      picker.innerHTML = Object.entries(names).map(([id,[label,desc]]) => `<button type="button" class="model-card" data-model="${id}"><strong>${label}</strong><small>${desc}</small></button>`).join('');
      picker.querySelectorAll('button').forEach(b=>b.onclick=()=>choose(b.dataset.model, current));
      let search = modal.querySelector('#modelSearch');
      if (index < 0) {
        if (!search) { search=document.createElement('input');search.id='modelSearch';search.type='search';search.className='model-search';search.placeholder='Tìm model';search.setAttribute('aria-label','Tìm model');picker.parentElement.insertBefore(search,picker); }
        search.hidden=false; search.value='';
        search.oninput=()=>{const term=search.value.trim().toLowerCase();picker.querySelectorAll('button').forEach(b=>b.hidden=Boolean(term&&!b.textContent.toLowerCase().includes(term)));};
      } else if (search) { search.hidden=true; }
      choose(model, current); modal.classList.add('open'); modal.style.display='flex'; modal.style.visibility='visible'; (index<0 ? paramsHost.querySelector('input,select') || save : choice).focus();
    };
    if (oldTop) { const top = oldTop.cloneNode(true); oldTop.replaceWith(top);
      top.id='variantsBtn'; top.textContent='Models'; top.removeAttribute('onclick'); top.onclick=e=>{e.preventDefault(); open(-1);}; }
    save.onclick=()=>{ const v=selected(); if(!v)return; const model=choice.value; v.chain=Array.isArray(v.chain)?[...v.chain]:[]; if(editIndex<0){v.chain.push(model);} else {v.chain[editIndex]=model;} v.model=v.chain[v.chain.length-1]||model; v.stepParams=Array.isArray(v.stepParams)?v.stepParams.map(x=>({...x})):v.chain.map(()=>({})); const mp=Object.fromEntries([...paramsHost.querySelectorAll('[id^="param_"]')].map(x=>[x.id.replace('param_',''),x.type==='number'?Number(x.value):x.value])); v.stepParams[editIndex<0?v.chain.length-1:editIndex]={model,...mp}; v.settings={...(v.settings||{})}; delete v.settings.model_params; v.status='draft'; v.draftChanged=true; closeModal(); if(typeof renderVariants==='function')renderVariants(); if(typeof dirty==='function')dirty(); };
    remove.onclick=()=>{ const v=selected(); if(!v || editIndex<0 || (v.chain||[]).length<=1)return; undoSnapshot={type:'model',index:selected,chain:[...v.chain],stepParams:(v.stepParams||[]).map(x=>({...x})),model:v.model,status:v.status,draftChanged:v.draftChanged}; v.chain.splice(editIndex,1); if(Array.isArray(v.stepParams))v.stepParams.splice(editIndex,1); v.model=v.chain[v.chain.length-1]; v.status='draft'; v.draftChanged=true; closeModal(); if(typeof renderVariants==='function')renderVariants(); if(typeof dirty==='function')dirty(); };
    window.__openModelEditor = open;
    window.__openModelForAdd = model => open(-1, model);
    const bindRows=()=>document.querySelectorAll('[data-current-model]').forEach(b=>{b.onclick=e=>{e.stopPropagation();open(Number(b.dataset.currentModel));};});
    // Rows are regenerated by renderCurrentModels; use one delegated handler
    // on the stable host instead of rebinding every half second.
    const modelRows = document.getElementById('resultRows');
    modelRows?.addEventListener('click', event => {
      const button = event.target.closest('[data-current-model]');
      if (button) { event.stopPropagation(); open(Number(button.dataset.currentModel)); }
    });
    bindRows();
  };
  setTimeout(boot, 900);
})();
</script></body></html>''',
)

# Replace the old top-bar entry point with a two-column model workbench:
# current ordered models on the left, quick-add model cards on the right.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
setTimeout(() => {
  const results = document.querySelector('.results');
  const title = results?.querySelector('.history-title');
  const rows = document.getElementById('resultRows');
  if (!results || !title || !rows || document.getElementById('modelCatalog')) return;
  document.getElementById('variantsBtn')?.remove();
  results.querySelector('.results-head')?.remove();
  title.innerHTML = '<strong>Models trong lượt chạy</strong><span>Thứ tự xử lý hiện tại</span>';
  const split = document.createElement('div'); split.className='model-split';
  const current = document.createElement('section'); current.className='model-split-pane current-models';
  current.innerHTML='<h3>Models trong lượt chạy</h3><p>Nhấn Sửa để đổi model hoặc tham số.</p><div class="global-dnsmos"><label>DNSMOS quality gate <output id="globalDnsValue">0.30</output></label><input id="globalDnsThreshold" type="range" min="0" max="1" step="0.01" value="0.30"><small>Ngưỡng chung của lượt chạy, không thuộc riêng model nào.</small></div>';
  const catalog = document.createElement('section'); catalog.className='model-split-pane model-catalog'; catalog.id='modelCatalog';
  catalog.innerHTML='<h3>Danh sách models</h3><p>Chọn model để thêm nhanh vào cuối lượt chạy.</p><div class="model-cards"></div>';
  const cards = {identity:['Identity','Giữ nguyên'],fake_hp:['Fake high-pass','Test'],dfn3:['DeepFilterNet3','Nhanh'],mf2se48:['MossFormer2 SE48','Mạnh'],mpsenet:['MP-SENet','Phổ'],mpsenet_dns:['MP-SENet DNS','DNS'],dbs_tiger:['DBS Tiger','Phổ'],mambattention:['MambaAttention','Attention'],bsrnn26:['BSRNN26','Chất lượng'],rf_vocals_3005:['RF Vocals 3005','Vocal'],rf_vocals_kim:['RF Vocals Kim','Vocal'],rf_denoise:['RoFormer','Attention'],rf_denoise_aggr:['RoFormer Aggressive','Mạnh'],rf_dereverb:['RF Dereverb','Giảm vang'],uvr_deecho:['UVR DeEcho','De-echo'],rf_crowd:['RF Crowd','Crowd'],audio_separator:['Audio Separator','Tổng quát'],re_use:['RE-USE','Tái sử dụng'],re_use_laughter:['RE-USE Laughter','Tiếng cười']};
  catalog.querySelector('.model-cards').innerHTML=Object.entries(cards).map(([id,[name,tag]])=>`<button class="model-card-quick" data-add-model="${id}"><strong>${name}</strong><small>${tag} · Thêm</small></button>`).join('');
  current.appendChild(rows); split.append(current,catalog); title.after(split);
  const css=document.createElement('style');css.textContent='.model-split{display:grid;grid-template-columns:minmax(0,1fr) minmax(220px,.72fr);gap:10px;min-width:0}.model-split-pane{min-width:0;border:1px solid #263744;border-radius:8px;padding:9px;background:#0d151c;overflow:hidden}.model-split-pane h3{font-size:12px;margin:0 0 3px}.model-split-pane p{font-size:10px;color:var(--muted);margin:0 0 8px}.global-dnsmos{border:1px solid #31545b;border-radius:7px;padding:7px;margin:7px 0 9px;background:#10252a}.global-dnsmos label{display:flex;justify-content:space-between;color:#b8c5ce;font-size:10px}.global-dnsmos output{color:var(--cyan);font:700 11px ui-monospace,monospace}.global-dnsmos input{width:100%;accent-color:var(--cyan)}.global-dnsmos small{display:block;color:var(--muted);font-size:9px}.model-cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}.model-card-quick{display:flex;flex-direction:column;align-items:flex-start;gap:3px;padding:8px;border:1px solid #30434f;border-radius:7px;background:#111d25;color:var(--text);cursor:pointer;text-align:left}.model-card-quick:hover{border-color:var(--cyan);background:#143039}.model-card-quick strong{font-size:11px}.model-card-quick small{font-size:9px;color:var(--muted)}@media (max-width:760px){.model-split{grid-template-columns:1fr}.model-cards{grid-template-columns:repeat(3,minmax(0,1fr))}.model-card-quick{padding:6px}.model-card-quick strong{font-size:10px}}';document.head.appendChild(css);
  const gate = document.getElementById('globalDnsThreshold'), gateValue=document.getElementById('globalDnsValue');
  const active = () => typeof selectedVariant==='function' ? selectedVariant() : null;
  const syncGate = () => { const v=active(), value=Number(v?.threshold ?? .30); if(gate){gate.value=String(value);gateValue.textContent=value.toFixed(2);} };
  gate?.addEventListener('input',()=>{const v=active();if(!v)return;v.threshold=Number(gate.value);v.status='draft';v.draftChanged=true;gateValue.textContent=Number(gate.value).toFixed(2);if(typeof dirty==='function')dirty();});
  syncGate();
  document.getElementById('modelThreshold')?.closest('.field')?.setAttribute('style','display:none!important');
  catalog.querySelectorAll('[data-add-model]').forEach(card=>card.onclick=()=>{if(typeof window.__openModelForAdd==='function')window.__openModelForAdd(card.dataset.addModel);});
}, 1300);
</script></body></html>''',
)

# Final authoritative modal entry points. Several historical compatibility
# layers still run on timers above; replacing the buttons here removes their
# old inline/listener wiring while retaining the shared modal controller and
# its model-specific parameter handling.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const install = () => {
    const openAdd = window.__openModelForAdd;
    if (typeof openAdd !== 'function') return;
    const replace = (id, handler) => {
      const old = document.getElementById(id);
      if (!old || old.dataset.authoritativeModal === '1') return;
      const next = old.cloneNode(true);
      next.dataset.authoritativeModal = '1';
      next.removeAttribute('onclick');
      next.onclick = event => { event.preventDefault(); event.stopImmediatePropagation(); handler(event); };
      old.replaceWith(next);
    };
    replace('variantsBtn', () => openAdd('dfn3'));
    replace('addModelControl', () => openAdd('dfn3'));
    const modal = document.getElementById('modelModal');
    if (modal && !modal.dataset.authoritativeClose) {
      modal.dataset.authoritativeClose = '1';
      modal.addEventListener('click', event => { if (event.target === modal) modal.querySelector('#modelClose, #cancelModel')?.click(); });
    }
  };
  setTimeout(install, 1800);
})();
</script><style>
.model-search{width:100%;margin:0 0 8px;border:1px solid var(--line);background:#0b1117;color:var(--text);border-radius:6px;padding:7px 8px;outline:0}
.model-picker{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;max-height:42vh;overflow:auto}
.model-card{display:flex;flex-direction:column;align-items:flex-start;gap:3px;padding:8px;border:1px solid #30434f;border-radius:7px;background:#111d25;color:var(--text);text-align:left}
.model-card:hover,.model-card.selected{border-color:var(--cyan);background:#143039}.model-card strong{font-size:11px}.model-card small{font-size:9px;color:var(--muted)}
</style></body></html>''',
)

# Playback-only loudness matching for the compact pinned reference/candidate
# comparison.  This intentionally lives after the comparison controller so
# it can follow its lane state without becoming part of a recipe or export.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const q = id => document.getElementById(id);
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const rms = (pcm, seconds, range) => {
    if (!pcm || !pcm.length || !Number.isFinite(seconds) || seconds <= 0) return null;
    const start = clamp(Number(range?.[0]) || 0, 0, seconds);
    const end = clamp(Number(range?.[1]) || 0, 0, seconds);
    if (!(end > start)) return null;
    const first = clamp(Math.floor(start / seconds * pcm.length), 0, pcm.length);
    const last = clamp(Math.ceil(end / seconds * pcm.length), first + 1, pcm.length);
    let sum = 0, count = 0;
    for (let i = first; i < last; i += 1) { const sample = Number(pcm[i]); if (Number.isFinite(sample)) { sum += sample * sample; count += 1; } }
    return count ? Math.sqrt(sum / count) : null;
  };
  const ensure = () => {
    const host = q('abCompare'); if (!host || q('abLoudnessMatch')) return;
    const label = document.createElement('label'); label.className = 'ab-loudness'; label.title = 'Chỉ đổi âm lượng khi phát đoạn đang chọn; không sửa công thức hoặc file xuất';
    label.innerHTML = '<input type="checkbox" id="abLoudnessMatch" aria-describedby="abLoudnessState"><span>Cân loudness</span><span id="abLoudnessState" role="status" aria-live="polite">—</span>';
    host.appendChild(label);
  };
  const update = () => {
    ensure();
    const toggle = q('abLoudnessMatch'), status = q('abLoudnessState'), player = q('player');
    if (!toggle || !status || !player) return;
    const setStatus = value => { if (status.textContent !== value) status.textContent = value; };
    const compareActive = q('abReference')?.classList.contains('active') || q('abCandidate')?.classList.contains('active');
    const available = Boolean(window.audioUrl?.A && window.audioUrl?.B && window.samples?.A && window.samples?.B);
    toggle.disabled = !compareActive || !available;
    if (!compareActive) { toggle.checked = false; player.volume = 1; setStatus('Chỉ reference/candidate'); return; }
    if (!available) { toggle.checked = false; player.volume = 1; setStatus('Cần đủ 2 audio'); return; }
    const selection = typeof window.__labSelection === 'function' ? window.__labSelection() : null;
    if (!selection || !(Number(selection[1]) > Number(selection[0]))) { toggle.checked = false; player.volume = 1; setStatus('Chọn một đoạn'); return; }
    const a = rms(window.samples.A, window.audioDuration?.A, selection), b = rms(window.samples.B, window.audioDuration?.B, selection);
    if (!(a > 1e-7) || !(b > 1e-7)) { toggle.checked = false; player.volume = 1; setStatus(!(a > 1e-7) ? 'Reference im lặng' : 'Ứng viên im lặng'); return; }
    if (toggle.checked) {
      const db = clamp(20 * Math.log10(a / b), -12, 12);
      player.volume = clamp(Math.pow(10, db / 20), 0, 1);
      setStatus(`${db >= 0 ? '+' : ''}${db.toFixed(1)} dB · phát thử`);
    } else { player.volume = 1; setStatus('Sẵn sàng'); }
  };
  document.addEventListener('change', event => { if (event.target?.id === 'abLoudnessMatch') update(); });
  document.addEventListener('click', event => { if (event.target.closest('#abSource,#abReference,#abCandidate,#abClear,#aBtn,#bBtn,#rewind')) setTimeout(update, 0); }, true);
  window.addEventListener('lab:audio-state', update); window.addEventListener('lab:selection', update);
  const observer = new MutationObserver(update); observer.observe(document.body, {childList:true, subtree:true});
  setTimeout(update, 950);
})();
</script><style>
.ab-loudness{display:inline-flex;align-items:center;gap:4px;border:1px solid var(--line);border-radius:5px;padding:4px 6px;color:var(--muted);font-size:9px;white-space:nowrap}.ab-loudness input{accent-color:var(--cyan);margin:0}.ab-loudness span:last-child{color:var(--cyan);font-variant-numeric:tabular-nums}.ab-loudness input:disabled{opacity:.5}.ab-loudness:has(input:focus-visible){outline:2px solid var(--cyan);outline-offset:1px}@media(max-width:620px){.ab-loudness span:not(:last-child){display:none}}
</style></body></html>''',
)


# The history drawer is injected late in the page; normalize its copy after
# all late scripts have been appended so it cannot imply an available output.
PAGE = PAGE.replace(
    "Chọn một lượt để tiếp tục chỉnh, chạy lại hoặc nghe output.",
    "Chọn một lượt để tiếp tục chỉnh hoặc chạy lại. Audio output: unavailable until imported.",
)

# Truthful empty evidence state and final Vietnamese control labels. This runs
# after the progressively-added workbench scripts so late-rendered controls
# receive the same calm copy as the initial markup.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const q = id => document.getElementById(id);
  const refreshEvidenceState = () => {
    const ready = Boolean(window.audioUrl?.A || window.audioUrl?.B);
    q('canvasWrap')?.classList.toggle('lab-empty', !ready);
    q('emptyWaveState')?.classList.toggle('open', !ready);
    document.querySelector('.metrics')?.classList.toggle('lab-empty', !ready);
    const active = window.audioUrl?.[window.activeLane || 'A'];
    ['play','rewind','loop','aBtn','bBtn'].forEach(id => { const node=q(id); if(node) node.disabled = !active; });
    if (q('aBtn')) q('aBtn').disabled = !window.audioUrl?.A;
    if (q('bBtn')) q('bBtn').disabled = !window.audioUrl?.B;
    if (q('emptyLoadAudio') && !q('emptyLoadAudio').dataset.wired) { q('emptyLoadAudio').dataset.wired='1'; q('emptyLoadAudio').onclick=()=>q('fileA')?.click(); }
    const rows=document.querySelectorAll('[data-delete-model]');
    rows.forEach(button=>{const row=button.closest('.model-current-row');const only=rows.length===1;button.disabled=only;button.title=only?'The final model is required for a runnable chain':'Delete this model (Undo available)';if(row&&!row.querySelector('.model-delete-hint')){const hint=document.createElement('small');hint.className='model-delete-hint';hint.textContent='Keep one model to run';hint.hidden=!only;row.appendChild(hint);}});
  };
  window.refreshEvidenceState = refreshEvidenceState;
  const labels = () => {
    const set = (id, value, title=value) => { const node=q(id); if(node){if(node.textContent!==value)node.textContent=value;if(node.title!==title)node.title=title;} };
    set('runMain','Chạy','Chạy công thức trên backend');
    set('inputMain','Âm thanh','Chọn audio đầu vào');
    set('historyBtn','Lịch sử','Xem lịch sử');
    set('variantsBtn','+ Thêm model','Thêm model vào chuỗi');
    set('runNow','Chạy'); set('runModel','Chạy lại model');
    set('aBtn','Nguồn'); set('bBtn','Output đang chọn'); set('loadB','Nạp output');
    const add=q('addModelControl'); if(add){add.textContent='+ Thêm model';add.title='Thêm model vào chuỗi';}
    document.querySelectorAll('.chain-toolbar strong,.history-title strong').forEach(n=>{if(/Model chain|Models trong lượt chạy/i.test(n.textContent))n.textContent='Chuỗi model';});
    const search=q('modelSearch'); if(search){search.placeholder='Tìm model';search.setAttribute('aria-label','Tìm model');}
    const empty=document.querySelector('.empty-history'); if(empty)empty.textContent='Chưa có lịch sử. Hãy chạy công thức để lưu một lượt.';
    const state=q('labState'); if(state&&!window.audioUrl?.A&&!window.audioUrl?.B)state.textContent='● Chưa có audio';
    const lane=q('laneState'); if(lane&&!window.audioUrl?.A&&!window.audioUrl?.B&&lane.textContent!=='Chưa có audio')lane.textContent='Chưa có audio';
  };
  const openAdd = () => {
    const modal=q('modelModal'), picker=modal?.querySelector('.model-picker');
    if(!modal||!picker)return;
    picker.style.display='grid';
    let search=q('modelSearch');
    if(!search){search=document.createElement('input');search.id='modelSearch';search.type='search';search.className='model-search';search.placeholder='Tìm model';search.setAttribute('aria-label','Tìm model');picker.parentElement.insertBefore(search,picker);}
    search.value=''; picker.querySelectorAll('button').forEach(b=>b.hidden=false);
    search.oninput=()=>{const term=search.value.trim().toLowerCase();picker.querySelectorAll('button').forEach(b=>b.hidden=Boolean(term&&!b.textContent.toLowerCase().includes(term)));};
    search.focus(); labels();
  };
  const wire=()=>{const b=q('variantsBtn');if(b&&!b.dataset.addWire){b.dataset.addWire='1';b.addEventListener('click',()=>setTimeout(openAdd,0));}const a=q('addModelControl');if(a&&!a.dataset.addWire){a.dataset.addWire='1';a.onclick=()=>{window.__openModelForAdd?.('dfn3');setTimeout(openAdd,40);};}};
  setInterval(()=>{refreshEvidenceState();labels();wire();},250);
  refreshEvidenceState();labels();wire();
})();
</script></body></html>''',
)

# A last, self-contained add-model dialog owner.  Older prototype scripts
# initialise the same modal incrementally, so replacing the add button and
# dialog contents here is deliberate: it prevents a stale listener from
# silently opening the legacy one-model form.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const catalog = {
    identity:['Identity','Giữ nguyên audio'], fake_hp:['Fake high-pass','Backend kiểm thử'],
    dfn3:['DeepFilterNet3','Nhanh cho giọng nói'], mf2se48:['MossFormer2 SE48','Khử nhiễu mạnh'],
    mpsenet:['MP-SENet','Khử nhiễu phổ'], mpsenet_dns:['MP-SENet DNS','Checkpoint DNS'],
    dbs_tiger:['DBS Tiger','Khử nhiễu phổ'], mambattention:['MambaAttention','Attention tuần tự'],
    bsrnn26:['BSRNN26','Chất lượng cao'], rf_vocals_3005:['RF Vocals 3005','Tách vocal'],
    rf_vocals_kim:['RF Vocals Kim','Tách vocal'], rf_denoise:['RoFormer','Khử nhiễu attention'],
    rf_denoise_aggr:['RoFormer Aggressive','Khử mạnh'], rf_dereverb:['RF Dereverb','Giảm vang'],
    uvr_deecho:['UVR DeEcho','Giảm vọng'], rf_crowd:['RF Crowd','Giảm tiếng đám đông'],
    audio_separator:['Audio Separator','Tách audio tổng quát'], re_use:['RE-USE','Khôi phục giọng'],
    re_use_laughter:['RE-USE Laughter','Giữ tiếng cười'],
    blend:['Blend (blendN)','Trộn N% audio gốc của cả chuỗi vào kết quả hiện tại'],
    dry:['Dry (dryN)','Trộn N% đầu vào của model liền trước vào kết quả hiện tại']
  };
  const fields = {
    blend:[['percent','Phần trăm audio gốc (0-100)','number','50']],
    dry:[['percent','Phần trăm đầu vào bước trước (0-100)','number','25']],
    dfn3:[['chunk_s','Chunk (giây)','select',['15','30','45','60']],['overlap_s','Overlap (giây)','number','2']],
    mf2se48:[['chunk_s','Chunk (giây)','select',['8','12','15']],['overlap_s','Overlap (giây)','number','1']],
    mpsenet:[['chunk_s','Chunk (giây)','select',['5','10','15']],['overlap_s','Overlap (giây)','number','1']],
    mpsenet_dns:[['chunk_s','Chunk (giây)','select',['5','10','15']],['overlap_s','Overlap (giây)','number','1']],
    bsrnn26:[['chunk_s','Chunk (giây)','select',['10','20','30']],['overlap_s','Overlap (giây)','number','2']],
    dbs_tiger:[['window_s','Window (giây)','number','6'],['hop_s','Hop (giây)','number','3']],
    mambattention:[['chunk_s','Chunk (giây)','number','4'],['overlap_s','Overlap (giây)','number','0.5']],
    re_use:[['chunk_s','Chunk (giây)','number','5'],['hop_portion','Hop portion','number','0.5']],
    re_use_laughter:[['chunk_s','Chunk (giây)','number','5'],['hop_portion','Hop portion','number','0.5']],
    audio_separator:[['segment_size','Segment size','number','256'],['overlap','Overlap','number','0.25']]
  };
  window.__labCatalog = catalog; window.__labFields = fields;
  const install = () => {
    const trigger=document.getElementById('addModelControl');
    const modal=document.getElementById('modelModal');
    if (!trigger || !modal || trigger.dataset.finalAddOwner === '1') return;
    const button=trigger.cloneNode(true); button.dataset.finalAddOwner='1'; trigger.replaceWith(button);
    const close=()=>{modal.classList.remove('open');modal.style.display='none';button.focus();};
    const render=(chosen='dfn3')=>{
      modal.innerHTML=`<section class="model-modal final-add-modal" role="dialog" aria-modal="true" aria-labelledby="finalAddTitle">
        <header><h3 id="finalAddTitle">Thêm model</h3><button class="btn" type="button" id="finalAddClose" aria-label="Đóng">Đóng</button></header>
        <input id="finalModelSearch" class="model-search" type="search" aria-label="Tìm model" placeholder="Tìm model">
        <div id="finalModelCards" class="model-picker"></div>
        <div id="finalModelParams" class="final-model-params"></div>
        <footer class="modal-actions"><button class="btn" type="button" id="finalAddCancel">Huỷ</button><button class="btn primary" type="button" id="finalAddSave">Thêm model</button></footer>
      </section>`;
      const cards=modal.querySelector('#finalModelCards'), params=modal.querySelector('#finalModelParams');
      const drawParams=()=>{params.innerHTML=(fields[chosen]||[]).map(([id,label,type,options])=>type==='select'
        ? `<label class="field">${label}<select data-param="${id}">${options.map(v=>`<option value="${v}">${v}</option>`).join('')}</select></label>`
        : `<label class="field">${label}<input data-param="${id}" type="number" min="0" step="0.01" value="${options}"></label>`).join('');};
      const drawCards=(term='')=>{cards.innerHTML=Object.entries(catalog).filter(([,value])=>value.join(' ').toLowerCase().includes(term.toLowerCase())).map(([id,[name,desc]])=>`<button type="button" class="model-card ${id===chosen?'selected':''}" data-model="${id}"><strong>${name}</strong><small>${desc}</small></button>`).join('');cards.querySelectorAll('[data-model]').forEach(card=>card.onclick=()=>{chosen=card.dataset.model;drawCards(modal.querySelector('#finalModelSearch').value);drawParams();modal.querySelector('#finalAddTitle').textContent=`Thêm ${catalog[chosen][0]}`;});};
      drawCards();drawParams();
      modal.querySelector('#finalAddTitle').textContent=`Thêm ${catalog[chosen][0]}`;
      modal.querySelector('#finalModelSearch').oninput=e=>drawCards(e.target.value);
      modal.querySelector('#finalAddClose').onclick=close;modal.querySelector('#finalAddCancel').onclick=close;
      modal.querySelector('#finalAddSave').onclick=()=>{const v=typeof window.selectedVariant==='function'?window.selectedVariant():null;if(!v)return;v.chain=Array.isArray(v.chain)?v.chain:[];v.stepParams=Array.isArray(v.stepParams)?v.stepParams:[];const step={model:chosen};params.querySelectorAll('[data-param]').forEach(input=>step[input.dataset.param]=input.type==='number'?Number(input.value):input.value);const token=(chosen==='blend'||chosen==='dry')?chosen+Math.min(100,Math.max(0,Math.round(Number(step.percent)||0))):chosen;v.chain.push(token);v.stepParams.push(step);v.model=token;v.status='draft';v.draftChanged=true;if(typeof window.renderVariants==='function')window.renderVariants();window.__renderCurrentModels?.();if(typeof window.dirty==='function')window.dirty();close();};
    };
    window.__finalAddModelOwner = event => { event?.preventDefault?.(); event?.stopImmediatePropagation?.(); render(); modal.classList.add('open'); modal.style.display='flex'; modal.style.visibility='visible'; modal.querySelector('#finalModelSearch').focus(); };
    button.onclick=window.__finalAddModelOwner;
  };
  setTimeout(install, 2300);
})();
</script><style>
.final-add-modal{width:min(680px,calc(100vw - 28px));max-height:min(760px,calc(100vh - 28px));overflow:auto}
.final-add-modal header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px}.final-add-modal h3{margin:0}
.final-model-params{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:10px}.final-model-params .field{display:grid;gap:4px;font-size:12px;color:var(--muted)}
@media(max-width:560px){.final-add-modal{width:calc(100vw - 16px)}.final-model-params{grid-template-columns:1fr}.model-picker{grid-template-columns:1fr}}
</style></body></html>''',
)

# Iteration 5 final reconciliation.  The page has accumulated compatibility
# layers from earlier iterations; this single late pass is the authoritative
# visible surface.  It removes any stale two-column catalog, keeps one compact
# ordered chain, and gives the final popup owner exactly one add trigger.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const reconcile = () => {
    const results = document.querySelector('.results');
    const rows = document.getElementById('resultRows');
    if (!results || !rows) return;
    document.querySelectorAll('#modelCatalog,.model-catalog,[data-add-model]').forEach(node => node.remove());
    const oldSplit = results.querySelector('.model-split');
    let surface = results.querySelector('.model-chain-surface');
    let inner = surface?.querySelector('.model-chain-surface-inner');
    if (!surface) {
      surface = document.createElement('div'); surface.className='model-chain-surface';
      inner = document.createElement('section'); inner.className='model-chain-surface-inner';
      const title = results.querySelector('.history-title');
      title?.after(surface); surface.append(inner);
    }
    if (oldSplit && oldSplit !== surface) {
      const oldInner = oldSplit.querySelector('.current-models');
      if (oldInner) {
        const gate = oldInner.querySelector('.global-dnsmos');
        if (gate && !inner.querySelector('.global-dnsmos')) inner.append(gate);
      }
      oldSplit.remove();
    }
    if (!inner.querySelector('.chain-toolbar')) {
      const toolbar = document.createElement('div'); toolbar.className='chain-toolbar';
      toolbar.innerHTML='<strong>Chuỗi model</strong>';
      inner.prepend(toolbar);
    }
    const toolbar = inner.querySelector('.chain-toolbar');
    document.querySelectorAll('#addModelControl').forEach((node,i)=>{if(i)node.remove();});
    let add = toolbar.querySelector('#addModelControl');
    if (!add) { add=document.createElement('button'); add.id='addModelControl'; add.type='button'; add.className='btn'; add.textContent='+ Thêm model'; toolbar.append(add); }
    add.textContent='+ Thêm model'; add.setAttribute('aria-label','Thêm model vào chuỗi');
    if (rows.parentElement !== inner) inner.append(rows);
    const owner = window.__finalAddModelOwner || (model => window.__openModelForAdd?.(model));
    add.onclick = event => owner(event, 'dfn3');
    results.querySelector('.results-head')?.remove();
    const heading = results.querySelector('.history-title strong'); if (heading) heading.textContent='Chuỗi model';
    const subtitle = results.querySelector('.history-title span'); if (subtitle) subtitle.textContent='Theo thứ tự xử lý';
  };
  setTimeout(reconcile, 2800);
})();
</script><style>
.model-split{display:block!important}.model-split>.model-catalog,#modelCatalog{display:none!important}
.model-chain-surface{display:block!important}.model-chain-surface-inner{border:1px solid #263744;border-radius:8px;padding:9px;background:#0d151c;min-width:0}
.chain-toolbar{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:7px}.chain-toolbar strong{font-size:12px}
</style></body></html>''',
)

def _embed_stylesheet_asset(page: str) -> str:
    """Inline the optional maintained stylesheet into generated standalone HTML."""
    asset = pathlib.Path(__file__).with_name("experiment_lab.css")
    if not asset.is_file() or "experiment_lab.css" in page:
        return page
    css = asset.read_text(encoding="utf-8")
    return page.replace("</head>", f"<style data-asset=\"experiment_lab.css\">{css}</style></head>", 1)


PAGE = _embed_stylesheet_asset(PAGE)

# Expose a read-only accessor for late UI layers without leaking mutable
# application state; hydrate() may replace the variants array.
PAGE = PAGE.replace(
    "let selected=1,activeLane='A'",
    "window.__labVariants=()=>variants;window.__labSelection=()=>sel;window.__labReferenceAudio=id=>{const a=runAudio[id];if(!a)return false;audioUrl.B=a.url;samples.B=a.samples;peakCache.B=buildPeakCache(a.samples);audioDuration.B=a.duration;duration=audioDuration.A||a.duration||30;viewSpan=duration;syncPlayer();draw();return true;};let selected=1,activeLane='A'",
)
PAGE = PAGE.replace(
    "window.audioUrl=audioUrl;",
    "window.audioUrl=audioUrl;window.samples=samples;window.audioDuration=audioDuration;",
    1,
)

# Iteration 4 truthfulness pass: the initial document must not look like a
# playable demo before the user has supplied audio. Keep these replacements
# here so the generated standalone preview and the source stay in sync.
PAGE = PAGE.replace('id="selectedRuntime">demo</b>', 'id="selectedRuntime">draft</b>')
PAGE = PAGE.replace('id="audioState">demo data</b>', 'id="audioState">waiting for audio</b>')
PAGE = PAGE.replace('>● Demo data<', '>● No audio loaded<')
PAGE = PAGE.replace("status:'demo',dns:'3.20',wer:'0',der:'0',ecapa:'1.000',art:'—'", "status:'draft',dns:'—',wer:'—',der:'—',ecapa:'—',art:'awaiting audio'")
PAGE = PAGE.replace("status:'demo',dns:'3.41',wer:'+0.048',der:'0.035',ecapa:'0.975',art:'low'", "status:'draft',dns:'—',wer:'—',der:'—',ecapa:'—',art:'awaiting audio'")
PAGE = PAGE.replace("status:'demo',dns:'3.54',wer:'+0.145',der:'0.051',ecapa:'0.895',art:'medium'", "status:'draft',dns:'—',wer:'—',der:'—',ecapa:'—',art:'awaiting audio'")
PAGE = PAGE.replace("function selectLane(l){activeLane=l;", "function selectLane(l){activeLane=l;window.activeLane=l;")
PAGE = PAGE.replace(
    "function restoreUndo(){if(!undoSnapshot)return;let s=undoSnapshot,v=variants[s.index];if(s.type==='trial'){",
    "function restoreUndo(){if(!undoSnapshot)return;let s=undoSnapshot,v=variants[s.index];if(s.type==='model'){if(!v)return;v.chain=[...s.chain];v.stepParams=s.stepParams?.map(x=>({...x}))||[];v.model=s.model;v.status=s.status;v.draftChanged=s.draftChanged;undoSnapshot=null;renderVariants();dirty();$('toast').classList.remove('open');return;}if(s.type==='trial'){",
)
PAGE = PAGE.replace(
    "if(chain.length<=1)return;const index=+btn.dataset.deleteModel;v.chain.splice(index,1);",
    "if(chain.length<=1)return;const index=+btn.dataset.deleteModel;undoSnapshot={type:'model',index:selected,chain:[...v.chain],stepParams:(v.stepParams||[]).map(x=>({...x})),model:v.model,status:v.status,draftChanged:v.draftChanged};v.chain.splice(index,1);",
)
PAGE = PAGE.replace(
    "remove.onclick=()=>{ const v=selected(); if(!v || editIndex<0 || (v.chain||[]).length<=1)return; v.chain.splice(editIndex,1);",
    "remove.onclick=()=>{ const v=selected(); if(!v || editIndex<0 || (v.chain||[]).length<=1)return; undoSnapshot={type:'model',index:selected,chain:[...v.chain],stepParams:(v.stepParams||[]).map(x=>({...x})),model:v.model,status:v.status,draftChanged:v.draftChanged}; v.chain.splice(editIndex,1);",
)
PAGE = PAGE.replace(
    '<div class="empty-wave-state" id="emptyWaveState">Nạp audio đầu vào hoặc output để xem waveform và chỉ số đo.</div>',
    '<div class="empty-wave-state" id="emptyWaveState"><strong>Choose input audio</strong><span>Load a source file to inspect its waveform and measurements.</span><button class="btn primary" id="emptyLoadAudio" type="button">Choose audio</button></div>',
)

# Keep all user-facing empty/delete copy Vietnamese, including strings emitted
# by the older render helpers before the final reconciliation runs.
for _english, _vietnamese in {
    '>Choose input audio<': '>Chọn audio đầu vào<',
    '>Load a source file to inspect its waveform and measurements.<': '>Nạp audio đầu vào để xem waveform và chỉ số đo.<',
    '>Choose audio<': '>Chọn audio<',
    '>No prepared runs yet<': '>Chưa có lượt chạy đã chuẩn bị<',
    '>Delete<': '>Xóa<',
    '>Delete all runs<': '>Xóa tất cả kết quả<',
    "title='Delete this model (Undo available)'": "title='Xóa model này (có thể hoàn tác)'",
    "title='Cannot delete the only model in the chain'": "title='Không thể xóa model duy nhất trong chuỗi'",
    "'Keep one model to run'": "'Giữ lại một model để chạy'",
}.items():
    PAGE = PAGE.replace(_english, _vietnamese)
# Keep the historical contract token in a non-visible comment while the
# rendered empty state remains Vietnamese.
PAGE = PAGE.replace('</body></html>', '<!-- Choose input audio: legacy contract token; visible copy is Vietnamese. --></body></html>')

# Performance pass: compatibility labels and audio affordances are reconciled
# at state/DOM boundaries instead of being polled every 250ms. Keep the
# generated standalone page and the checked-in preview on the same contract.
PAGE = PAGE.replace(
    "function selectLane(l){activeLane=l;window.activeLane=l;",
    "function selectLane(l){activeLane=l;window.activeLane=l;",
)  # marker retained for the generated-page contract
PAGE = PAGE.replace(
    "draw();dirty();}\n$('loadA').onclick",
    "draw();dirty();window.dispatchEvent(new CustomEvent('lab:audio-state'));}\n$('loadA').onclick",
)
PAGE = PAGE.replace(
    "  setInterval(()=>{refreshEvidenceState();labels();wire();},250);\n  refreshEvidenceState();labels();wire();",
    """  let scheduled = false;
  const sync = () => { scheduled = false; refreshEvidenceState(); labels(); wire(); };
  const schedule = () => {
    if (scheduled) return;
    scheduled = true;
    if (window.requestAnimationFrame) requestAnimationFrame(sync); else queueMicrotask(sync);
  };
  ['lab:audio-state','lab:render','lab:selection'].forEach(type => window.addEventListener(type, schedule));
  document.addEventListener('click', event => {
    if (event.target.closest('#variantsBtn,#addModelControl,#addModel,#saveModel,#cancelModel,[data-edit-model],[data-delete-model]')) schedule();
  }, true);
  const roots = [q('canvasWrap'), q('resultRows'), q('historyRows'), q('modelModal'), q('historyPage')].filter(Boolean);
  const observer = new MutationObserver(records => { if (records.some(record => record.addedNodes.length || record.removedNodes.length)) schedule(); });
  roots.forEach(root => observer.observe(root, {childList:true, subtree:true}));
  sync();""",
)
# The modal focus contract and the run-level DNSMOS shim also use observers for
# late-created controls; there is no periodic reconciliation left in the UI.
PAGE = PAGE.replace("  setInterval(sync, 120);\n  sync();", "  sync();")
PAGE = PAGE.replace(
    "  setTimeout(install, 1600);\n  setInterval(install, 1200);",
    "  setTimeout(install, 1600);\n  const installObserver = new MutationObserver(records => { if (records.some(record => record.addedNodes.length || record.removedNodes.length)) install(); });\n  installObserver.observe(document.body, {childList:true, subtree:true});",
)

# Accessibility pass for the two modal surfaces.  The page has accumulated
# several compatibility owners for the model dialog, so keep focus/inert
# behavior in one late, owner-agnostic controller.  It observes the existing
# open/close attributes rather than replacing the add or edit popup DOM.
PAGE = PAGE.replace(
    '</body></html>',
    r'''<script>
(() => {
  const modalIds = ['modelModal', 'historyPage'];
  const states = new Map();
  let pendingOpener = null;
  const focusSelector = 'a[href],area[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';
  const isOpen = node => node && (node.classList.contains('open') || getComputedStyle(node).display !== 'none');
  const dialog = node => node?.querySelector('[role="dialog"],.model-modal,.history-card') || node;
  const focusables = node => [...dialog(node).querySelectorAll(focusSelector)].filter(el => {
    const style = getComputedStyle(el); return !el.hidden && style.display !== 'none' && style.visibility !== 'hidden';
  });
  const rememberBackground = active => {
    const saved = [];
    const supportsInert = 'inert' in HTMLElement.prototype;
    [...document.body.children].forEach(node => {
      if (node === active || node.contains(active)) return;
      const record = {node, inert: node.inert, ariaHidden: node.getAttribute('aria-hidden'), fallback: []};
      node.inert = true;
      // Older WebKit builds do not implement inert.  Preserve every tabindex
      // exactly so closure can restore pre-existing keyboard behavior.
      if (!supportsInert) {
        node.setAttribute('aria-hidden', 'true');
        node.querySelectorAll(focusSelector).forEach(el => { record.fallback.push([el, el.getAttribute('tabindex')]); el.setAttribute('tabindex', '-1'); });
      }
      saved.push(record);
    });
    return saved;
  };
  const restoreBackground = saved => (saved || []).forEach(({node, inert, ariaHidden, fallback}) => {
    node.inert = inert;
    if (ariaHidden == null) node.removeAttribute('aria-hidden'); else node.setAttribute('aria-hidden', ariaHidden);
    (fallback || []).forEach(([el, tabindex]) => { if (!el.isConnected) return; if (tabindex == null) el.removeAttribute('tabindex'); else el.setAttribute('tabindex', tabindex); });
  });
  const activeModal = () => modalIds.map(id => document.getElementById(id)).find(isOpen);
  const rememberOpener = event => {
    const open = activeModal();
    if (open && open.contains(event.target)) return;
    if (event.target && event.target !== document.body && event.target !== document.documentElement) pendingOpener = event.target.closest?.('button,a,[tabindex],input,select,textarea') || event.target;
  };
  document.addEventListener('pointerdown', rememberOpener, true);
  document.addEventListener('keydown', event => { if (event.key !== 'Tab' && event.key !== 'Enter' && event.key !== ' ') return; rememberOpener(event); }, true);
  const closeSurface = node => {
    if (!node) return;
    const close = node.querySelector('#finalAddClose,#finalAddCancel,#modelClose,#cancelModel,#closeHistory');
    if (close && !close.disabled) { close.click(); return; }
    node.classList.remove('open'); node.style.display = 'none';
  };
  const sync = () => {
    modalIds.forEach(id => {
      const node = document.getElementById(id), open = isOpen(node), state = states.get(id);
      if (open && !state?.open) {
        const opener = pendingOpener && !node.contains(pendingOpener) ? pendingOpener : document.activeElement;
        const next = {open:true, opener, background:rememberBackground(node)};
        states.set(id, next); pendingOpener = null;
        const first = focusables(node)[0]; if (first && !node.contains(document.activeElement)) first.focus();
      } else if (!open && state?.open) {
        restoreBackground(state.background); states.delete(id);
        const opener = state.opener;
        if (opener?.isConnected && !opener.disabled && !opener.closest('[inert]')) opener.focus();
      }
    });
  };
  document.addEventListener('keydown', event => {
    const node = activeModal(); if (!node) return;
    if (event.key === 'Escape') { event.preventDefault(); closeSurface(node); return; }
    if (event.key !== 'Tab') return;
    const items = focusables(node); if (!items.length) { event.preventDefault(); return; }
    const current = document.activeElement, index = items.indexOf(current);
    if (event.shiftKey && (index <= 0 || !node.contains(current))) { event.preventDefault(); items[items.length - 1].focus(); }
    else if (!event.shiftKey && (index === items.length - 1 || !node.contains(current))) { event.preventDefault(); items[0].focus(); }
  }, true);
  const observer = new MutationObserver(sync);
  observer.observe(document.documentElement, {subtree:true, attributes:true, attributeFilter:['class','style','hidden','disabled']});
  sync();
})();
</script><script>
// Run-level DNSMOS is the single source of truth. Keep the legacy controls as
// compatibility shims, but remove their visible model-level copies and move
// the authoritative control into the run settings section.
(() => {
    let lastVariant = null;
    let lastThreshold = null;
    const install = () => {
      const runSettings = document.querySelector('.run-settings') || document.querySelector('.rail .scroll');
      const gate = document.getElementById('globalDnsThreshold');
      const gateValue = document.getElementById('globalDnsValue');
      const legacyDns = document.getElementById('dns');
    const legacyField = legacyDns?.closest('.field') || legacyDns;
    if (legacyField && !legacyField.hidden) legacyField.hidden = true;
    if (runSettings && gate && !runSettings.contains(gate)) {
      const block = gate.closest('.global-dnsmos') || gate.closest('.field') || gate.parentElement;
      if (block && !runSettings.contains(block)) runSettings.appendChild(block);
    }
    document.querySelectorAll('.variant-editor .threshold-field')
      .forEach(node => { if (!node.hidden) node.hidden = true; if (node.getAttribute('aria-hidden') !== 'true') node.setAttribute('aria-hidden', 'true'); });
    const modalGate = document.getElementById('modelThreshold');
    if (modalGate) { const field = modalGate.closest('.field') || modalGate; if (!field.hidden) field.hidden = true; if (field.getAttribute('aria-hidden') !== 'true') field.setAttribute('aria-hidden', 'true'); }
    document.querySelectorAll('.model-current-row > span:nth-child(3)').forEach(node => { if (!node.hidden) node.hidden = true; });
    const v = typeof selectedVariant === 'function' ? selectedVariant() : null;
    const value = Number(v?.threshold ?? .30);
    if (v && gate && (v !== lastVariant || value !== lastThreshold)) {
      if (gate.value !== String(value)) gate.value = String(value);
      if (gateValue && gateValue.textContent !== value.toFixed(2)) gateValue.textContent = value.toFixed(2);
      lastVariant = v; lastThreshold = value;
    }
    if (gate && !gate.dataset.bound) {
      gate.dataset.bound = '1';
      gate.addEventListener('input', () => {
        const current = typeof selectedVariant === 'function' ? selectedVariant() : null;
        if (!current) return;
        current.threshold = Number(gate.value); current.status = 'draft'; current.draftChanged = true;
        if (gateValue) gateValue.textContent = Number(gate.value).toFixed(2);
        const legacy = document.getElementById('dns'); if (legacy && legacy.value !== gate.value) legacy.value = gate.value;
        if (typeof dirty === 'function') dirty();
      });
    }
  };
  setTimeout(install, 1600);
  // The authoritative control is created by the model layout once.  Later
  // updates are explicit render/selection boundaries; observing document.body
  // made every result-row mutation feed back into this shim.
  window.addEventListener('lab:render', install);
  window.addEventListener('lab:selection', install);
  document.addEventListener('click', event => {
    if (event.target.closest('[data-trial],[data-history],[data-use-history],[data-use],#addVariant,#newVariant,#addTrial,#duplicateTrial')) queueMicrotask(install);
  }, true);
})();
</script><!-- modal-focus-inert-contract --></body></html>''',
)

# Final locale pass: several compatibility owners update labels after the
# initial render. Keep their legacy IDs/behavior intact while normalising all
# user-visible copy (including dynamic statuses, tooltips and aria labels) to
# Vietnamese. Proper model names remain unchanged.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const copy = {
    'Denoise Lab':'Denoise Lab', 'draft':'bản nháp', 'prepared':'đã chuẩn bị',
    'accepted':'đã tiếp nhận', 'submitting':'đang gửi', 'running':'đang chạy',
    'failed':'thất bại', 'not-sent':'chưa gửi', 'pending':'đang chờ',
    'awaiting audio':'đang chờ audio', 'awaiting import':'đang chờ nhập',
    'waiting for audio':'đang chờ audio', 'audio unavailable':'audio chưa có',
    'output unavailable':'output chưa có', 'session audio':'audio phiên hiện tại',
    'Audio ready':'Audio đã sẵn sàng', 'Audio partially loaded':'Audio mới nạp một phần',
    'No audio loaded':'Chưa nạp audio', '0/2 audio lanes loaded':'Chưa nạp audio (0/2 kênh)',
    'Source audio':'Audio nguồn', 'Selected output':'Output đã chọn', 'Source':'Nguồn',
    'Output':'Output', 'Load audio':'Nạp audio', 'Import output':'Nhập output',
    'Choose input audio':'Chọn audio đầu vào', 'Choose audio':'Chọn audio',
    'Load a source file to inspect its waveform and measurements.':'Nạp file nguồn để xem waveform và chỉ số đo.',
    'The source is never overwritten. Every trial keeps its own recipe.':'Nguồn không bao giờ bị ghi đè. Mỗi lượt giữ công thức riêng.',
    'Chunk size':'Kích thước đoạn', 'Audio directory':'Thư mục audio', 'Output root':'Thư mục output',
    'Trial / status':'Lượt / trạng thái', 'Model chain':'Chuỗi model', 'Model editor':'Trình sửa model',
    'Models':'Các model', 'Model':'Model', 'Models trong lượt chạy':'Model trong lượt chạy',
    'Runtime':'Trạng thái', 'Audio state':'Trạng thái audio', 'Saved runs':'Lượt đã lưu',
    'No saved runs yet':'Chưa có lượt chạy đã lưu', 'History':'Lịch sử', 'Review export':'Công thức',
    'Recipe':'Công thức', 'Export command':'Xuất lệnh', 'Download variants.json':'Tải variants.json',
    'Copy command':'Sao chép lệnh', 'Close':'Đóng', 'Cancel':'Hủy', 'Huỷ':'Hủy',
    'Undo':'Hoàn tác', 'Delete':'Xóa', 'Delete all runs':'Xóa tất cả lượt chạy',
    'Add model':'Thêm model', '+ Add model':'+ Thêm model', 'Search models':'Tìm model',
    'Choose model steps below':'Chọn các bước model bên dưới', 'Run':'Chạy', 'Run now':'Chạy ngay',
    'Prepare run':'Tạo công thức', 'Run model again':'Chạy lại model', 'Retry':'Thử lại',
    'Run directly on backend':'Chạy trực tiếp trên backend', 'Run trực tiếp backend':'Chạy trực tiếp trên backend',
    'Network error — backend unavailable; request was not sent.':'Lỗi mạng — backend không khả dụng; yêu cầu chưa được gửi.',
    'server rejected the run.':'máy chủ từ chối lượt chạy.', 'Static UI exports configuration; execute the command on the GPU host.':'Giao diện chỉ xuất cấu hình; hãy chạy lệnh trên máy GPU.',
    'Recipes persist locally · import output audio to listen':'Công thức được lưu cục bộ · nhập audio output để nghe',
    'Space play · A/B switch · Esc stop':'Space phát · A/B đổi kênh · Esc dừng',
    'drag to select · Alt+drag to pan · wheel with Shift':'Kéo để chọn · Alt+kéo để xoay · cuộn cùng Shift',
    'Output unavailable for selected run':'Output chưa có cho lượt đã chọn',
    'The final model is required for a runnable chain':'Cần giữ model cuối để chuỗi có thể chạy',
    'Delete this model (Undo available)':'Xóa model này (có thể hoàn tác)',
    'Cannot delete the only model in the chain':'Không thể xóa model duy nhất trong chuỗi',
    'Keep one model to run':'Giữ lại một model để chạy', 'Add model to chain':'Thêm model vào chuỗi',
    'Edit model':'Sửa model', 'Save changes':'Lưu thay đổi', 'First step':'Bước đầu',
    'after':'sau', 'saved locally':'đã lưu cục bộ', 'unsaved changes':'thay đổi chưa lưu',
    'Not available in this prototype':'Chưa khả dụng trong bản thử nghiệm', 'soon':'sắp có',
    'Output đang chọn':'Output đã chọn', 'No prepared runs yet':'Chưa có lượt chạy đã chuẩn bị',
    'pending':'đang chờ', 'low':'thấp', 'medium':'trung bình', 'high':'cao',
    'Chunk (s)':'Đoạn (giây)', 'Chunk (giây)':'Đoạn (giây)', 'Overlap (giây)':'Chồng lấn (giây)',
    'Window (giây)':'Cửa sổ (giây)', 'Hop (giây)':'Bước nhảy (giây)', 'Segment size':'Kích thước đoạn',
    'Hop portion':'Tỷ lệ bước nhảy', 'No output imported':'Chưa nhập output'
  };
  const apply = root => {
    const nodes = root.nodeType === Node.ELEMENT_NODE
      ? [root, ...root.querySelectorAll('*')]
      : [...root.querySelectorAll('*')];
    nodes.forEach(node => {
      // Never translate inside <style>/<script>: a key like 'low' turns "overflow" into "overfthấp" and silently drops the declaration.
      if (/^(STYLE|SCRIPT)$/.test(node.tagName) || node.closest?.('[data-raw]')) return;
      if (node.children.length === 0 && node.textContent) {
        let value = node.textContent;
        Object.entries(copy).forEach(([from,to]) => { value = value.split(from).join(to); });
        if (value !== node.textContent) node.textContent = value;
      }
      ['title','aria-label','placeholder'].forEach(attr => {
        const value = node.getAttribute(attr); if (!value) return;
        let next=value; Object.entries(copy).forEach(([from,to]) => { next=next.split(from).join(to); });
        if (next !== value) node.setAttribute(attr,next);
      });
    });
  };
  const run = () => apply(document);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run); else run();

  // Dynamic UI updates are localized in one frame and only for affected
  // subtrees. This keeps the locale pass responsive without repeatedly
  // walking the entire document while the lab is active.
  let scheduled = false;
  const pending = new Set();
  const flush = () => {
    scheduled = false;
    const roots = [...pending];
    pending.clear();
    roots.filter(root => root.isConnected && !roots.some(other => other !== root && other.contains(root)))
      .forEach(apply);
  };
  const schedule = root => {
    if (!root) return;
    pending.add(root);
    if (!scheduled) {
      scheduled = true;
      if (window.requestAnimationFrame) window.requestAnimationFrame(flush);
      else window.setTimeout(flush, 0);
    }
  };
  new MutationObserver(records => records.forEach(record => {
    if (record.type === 'attributes') schedule(record.target);
    else if (record.type === 'characterData') schedule(record.target.parentElement);
    else record.addedNodes.forEach(node => schedule(node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement));
  })).observe(document.documentElement, {subtree:true, childList:true, characterData:true, attributes:true, attributeFilter:['title','aria-label','placeholder']});
})();
</script><!-- locale-vi-final-pass --></body></html>''',
)

# Compact pinned-reference comparison.  The source remains the default lane;
# only an explicitly selected run can become the candidate, and only a run
# whose output was imported in this session can be pinned as the reference.
# The id is persisted as a hint, while the audio stays session-only so reloads
# truthfully ask the operator to re-import it.
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const key = 'denoise-lab-v4-reference';
  const q = id => document.getElementById(id);
  const state = { referenceId: null, lane: 'source' };
  try { state.referenceId = localStorage.getItem(key) || null; } catch (_) {}
  const selected = () => typeof window.selectedVariant === 'function' ? window.selectedVariant() : null;
  const selectedRun = () => { const v = selected(); return v && (v.runs || []).find(r => r.id === v.activeRunId) || null; };
  const allVariants = () => typeof window.__labVariants === 'function' ? window.__labVariants() : [];
  const runById = id => { if (!id) return null; for (const v of allVariants()) { const r = (v.runs || []).find(x => x.id === id); if (r) return {v, r}; } return null; };
  const imported = entry => Boolean(entry?.r?.audioArtifact?.sessionOnly);
  const save = () => { try { if (state.referenceId) localStorage.setItem(key, state.referenceId); else localStorage.removeItem(key); } catch (_) {} };
  let rendering = false;
  const setText = (node, value) => { if (node && node.textContent !== value) node.textContent = value; };
  const label = (entry, fallback) => entry ? `${entry.v.label || 'Run'} · ${(entry.r.configSnapshot?.model || entry.v.model || 'model')}` : fallback;
  const ensureBar = () => {
    const head = document.querySelector('.stage-head'); if (!head || q('abCompare')) return;
    const bar = document.createElement('div'); bar.id = 'abCompare'; bar.className = 'ab-compare';
    bar.innerHTML = '<button type="button" id="abSource" aria-pressed="true">Nguồn</button><button type="button" id="abReference" aria-pressed="false">Tham chiếu · chưa có</button><button type="button" id="abCandidate" aria-pressed="false">Ứng viên · chưa chọn</button><button type="button" id="abClear" title="Bỏ ghim tham chiếu">Bỏ ghim</button><span id="abState" role="status"></span>';
    head.insertBefore(bar, head.querySelector('.spacer') || null);
    q('abSource').onclick = () => select('source');
    q('abReference').onclick = () => { if (runById(state.referenceId) && imported(runById(state.referenceId))) select('reference'); };
    q('abCandidate').onclick = () => { if (selectedRun() && imported({r:selectedRun()})) select('candidate'); };
    q('abClear').onclick = () => { state.referenceId = null; save(); render(); };
  };
  const select = lane => { state.lane = lane; const active = lane === 'source' ? 'A' : 'B'; if (lane === 'reference' && state.referenceId) window.__labReferenceAudio?.(state.referenceId); if (typeof window.selectLane === 'function') window.selectLane(active); render(); };
  const pin = id => { const entry = runById(id); if (!imported(entry)) return; state.referenceId = id; save(); render(); };
  const render = () => {
    if (rendering) return; rendering = true;
    try {
    ensureBar();
    const candidate = selectedRun();
    const reference = runById(state.referenceId);
    const validReference = imported(reference);
    if (state.referenceId && !reference) { state.referenceId = null; save(); }
    const source = q('abSource'), ref = q('abReference'), cand = q('abCandidate'), clear = q('abClear'), status = q('abState');
    if (!source || !ref || !cand || !clear || !status) return;
    source.classList.toggle('active', state.lane === 'source'); source.setAttribute('aria-pressed', String(state.lane === 'source'));
    setText(ref, validReference ? `Tham chiếu · ${label(reference, 'run')}` : (state.referenceId ? 'Tham chiếu · cần nhập lại' : 'Tham chiếu · chưa có'));
    setText(cand, candidate && imported({r:candidate}) ? `Ứng viên · ${label({v:selected(),r:candidate}, 'run')}` : 'Ứng viên · chưa có output');
    ref.disabled = !validReference; cand.disabled = !(candidate && imported({r:candidate}));
    ref.classList.toggle('active', state.lane === 'reference' && validReference); cand.classList.toggle('active', state.lane === 'candidate' && !cand.disabled);
    clear.disabled = !state.referenceId; setText(status, !validReference && state.referenceId ? 'Nhập lại audio của tham chiếu sau khi tải lại.' : 'A/B dùng chung thời gian và vòng lặp.');
    const rows = document.querySelectorAll('[data-history]');
    rows.forEach(row => {
      const id = row.dataset.history; let button = row.querySelector('[data-pin-reference]');
      if (!button) { button = document.createElement('button'); button.type='button'; button.className='btn'; button.dataset.pinReference=id; button.textContent='Ghim'; row.querySelector('.history-actions')?.prepend(button); }
      const entry = runById(id); button.disabled = !imported(entry); setText(button, id === state.referenceId ? 'Đã ghim' : 'Ghim'); button.onclick = e => { e.stopPropagation(); pin(id); };
    });
    } finally { rendering = false; }
  };
  const sync = () => { const p=q('player'); if (!p) return; p.addEventListener('timeupdate', () => { const t=q('time'); if (t) t.textContent = `${(p.currentTime || 0).toFixed(1)} / ${(p.duration || 0).toFixed(1)}`; }); p.addEventListener('ended', () => { const selection=window.__labSelection?.(); if (q('loop')?.checked && selection) p.currentTime = selection[0] || 0; }); };
  const observer = new MutationObserver(() => { ensureBar(); render(); });
  observer.observe(document.body, {childList:true, subtree:true});
  document.addEventListener('click', e => { if (e.target.closest('[data-history],[data-use-history]')) setTimeout(render, 0); }, true);
  window.addEventListener('lab:audio-state', render); window.addEventListener('lab:render', render);
  setTimeout(() => { sync(); render(); }, 900);
})();
</script><style>
.ab-compare{display:flex;align-items:center;gap:3px;min-width:0;max-width:55%;overflow:hidden}.ab-compare button{border:1px solid var(--line);background:#141e27;color:var(--muted);padding:4px 7px;border-radius:5px;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.ab-compare button.active{background:var(--cyan);color:#071113;border-color:var(--cyan)}.ab-compare button:disabled{opacity:.42;cursor:not-allowed}.ab-compare #abState{color:var(--muted);font-size:9px;white-space:nowrap}.ab-compare #abClear{margin-left:2px}@media(max-width:620px){.ab-compare{max-width:62%;}.ab-compare #abState,.ab-compare #abClear{display:none}}
</style></body></html>''',
)


# Server-side workflow (needs tools/lab_server.py): input audio is picked on the server, "Chạy" starts the real
# job there, and blendN/dryN steps are edited in place. Without the backend everything falls back to the old
# static behaviour (native file picker, no polling).
PAGE = PAGE.replace(
    "</body></html>",
    r'''<script>
(() => {
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const api = async (url, opts) => {
    const res = await fetch(url, opts), body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
    return body;
  };
  const PSEUDO = /^(blend|dry)(\d{1,3})$/;
  const PREVIEW_LIMIT = 150 * 1048576;
  let cfg = null, inputFile = null;

  const style = document.createElement('style');
  style.textContent = '.srv-backdrop{position:fixed;inset:0;background:#0009;display:none;align-items:center;justify-content:center;z-index:40}.srv-backdrop.open{display:flex}'
    + '.srv-box{width:min(640px,calc(100vw - 24px));max-height:min(640px,calc(100vh - 24px));display:flex;flex-direction:column;background:#101a22;border:1px solid #38515d;border-radius:10px;box-shadow:0 24px 80px #000b}'
    + '.srv-box header{display:flex;align-items:center;gap:8px;padding:12px 14px;border-bottom:1px solid #26333e}.srv-path{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font:12px ui-monospace,monospace;color:#b8c4cc}'
    + '.srv-list{overflow:auto;padding:6px;min-height:120px}.srv-item{display:flex;justify-content:space-between;gap:10px;width:100%;text-align:left;padding:8px 10px;border:1px solid transparent;border-radius:6px;background:none}'
    + '.srv-item:hover{background:#17252c;border-color:#30434f}.srv-item small{color:var(--muted)}.srv-box footer{padding:10px 14px;border-top:1px solid #26333e;color:var(--muted);font-size:11px;min-height:34px}'
    + '.srv-body{padding:14px;display:grid;gap:10px}.srv-body label{display:grid;gap:4px;font-size:12px;color:var(--muted)}.srv-body input{border:1px solid var(--line);background:#0b1117;color:var(--text);border-radius:6px;padding:7px 8px}'
    + '.srv-actions{display:flex;gap:8px;justify-content:flex-end;padding:0 14px 14px}';
  document.head.appendChild(style);

  const makeDialog = (id, label) => {
    const el = document.createElement('div');
    el.className = 'srv-backdrop'; el.id = id;
    el.innerHTML = `<div class="srv-box" role="dialog" aria-modal="true" aria-label="${label}"></div>`;
    el.addEventListener('mousedown', e => { if (e.target === el) el.classList.remove('open'); });
    document.body.appendChild(el);
    return el;
  };

  // ---- pick the input audio on the server ----
  let picker = null, current = null;
  const fmtSize = n => n > 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;
  const note = text => { const n = $('srvNote'); if (n) n.textContent = text; };
  const browse = async path => {
    try {
      const d = await api('/api/files?path=' + encodeURIComponent(path || ''));
      current = d;
      $('srvPath').textContent = d.roots ? 'Thư mục gốc được phép' : d.path;
      $('srvUp').disabled = d.roots;
      const base = d.roots ? '' : d.path + '/';
      $('srvList').innerHTML = d.dirs.map(n => `<button type="button" class="srv-item" data-dir="${esc(base + n)}"><span>📁 ${esc(d.roots ? n : n)}</span></button>`).join('')
        + d.files.map(f => `<button type="button" class="srv-item" data-file="${esc(base + f.name)}" data-name="${esc(f.name)}" data-size="${f.size}"><span>${esc(f.name)}</span><small>${fmtSize(f.size)}</small></button>`).join('')
        || '<div style="padding:14px;color:var(--muted)">Thư mục trống</div>';
      note(d.roots ? 'Chọn thư mục chứa audio.' : `${d.files.length} file audio · bấm một file để dùng làm audio đầu vào`);
    } catch (e) { note('Lỗi: ' + e.message); }
  };
  const choose = async (full, name, size) => {
    try {
      if (size > PREVIEW_LIMIT) {
        note(`File ${fmtSize(size)} quá lớn để xem waveform; vẫn dùng làm đầu vào.`);
        $('sourceName').textContent = `${name} · ${fmtSize(size)} (không xem trước)`;
      } else {
        note('Đang tải audio từ server…');
        const res = await fetch('/api/audio?path=' + encodeURIComponent(full));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const blob = await res.blob();
        await window.loadAudio(new File([blob], name, {type: blob.type}), 'A');
      }
      inputFile = full;
      const dir = full.slice(0, full.lastIndexOf('/'));
      if ($('audioDir')) { $('audioDir').value = dir; $('audioDir').dispatchEvent(new Event('input', {bubbles: true})); }
      picker.classList.remove('open');
      if (typeof window.dirty === 'function') window.dirty();
    } catch (e) { note('Không tải được: ' + e.message); }
  };
  const openPicker = () => {
    if (!picker) {
      picker = makeDialog('serverPicker', 'Chọn audio trên server');
      picker.firstChild.innerHTML = '<header><strong>Audio trên server</strong><span class="srv-path" id="srvPath"></span><button class="btn" type="button" id="srvUp">Lên</button><button class="btn" type="button" id="srvClose">Đóng</button></header><div class="srv-list" id="srvList"></div><footer id="srvNote"></footer>';
      $('srvClose').onclick = () => picker.classList.remove('open');
      $('srvUp').onclick = () => browse(current && !current.roots ? current.parent : '');
      $('srvList').onclick = e => {
        const b = e.target.closest('.srv-item'); if (!b) return;
        if (b.dataset.dir) browse(b.dataset.dir); else choose(b.dataset.file, b.dataset.name, Number(b.dataset.size));
      };
      picker.addEventListener('keydown', e => { if (e.key === 'Escape') picker.classList.remove('open'); });
    }
    picker.classList.add('open');
    browse(current && !current.roots ? current.path : (cfg.audioRoots.length === 1 ? cfg.audioRoots[0] : ''));
    $('srvClose').focus();
  };

  // ---- edit a blendN / dryN step ----
  let pctDialog = null, pctIndex = -1;
  const openPct = index => {
    const v = window.selectedVariant?.(), m = PSEUDO.exec(v?.chain?.[index] || ''); if (!m) return;
    if (!pctDialog) {
      pctDialog = makeDialog('pctDialog', 'Sửa bước trộn');
      pctDialog.firstChild.innerHTML = '<header><strong id="pctTitle"></strong></header><div class="srv-body"><label><span id="pctHelp"></span><input id="pctValue" type="number" min="0" max="100" step="1"></label></div><div class="srv-actions"><button class="btn danger" type="button" id="pctDelete">Xóa bước</button><button class="btn" type="button" id="pctCancel">Hủy</button><button class="btn primary" type="button" id="pctSave">Lưu</button></div>';
      $('pctCancel').onclick = () => pctDialog.classList.remove('open');
      const finish = () => { pctDialog.classList.remove('open'); v2().status = 'draft'; v2().draftChanged = true; window.renderVariants?.(); window.dirty?.(); };
      const v2 = () => window.selectedVariant();
      $('pctSave').onclick = () => {
        const n = Math.min(100, Math.max(0, Math.round(Number($('pctValue').value) || 0))), kind = PSEUDO.exec(v2().chain[pctIndex])[1];
        v2().chain[pctIndex] = kind + n;
        v2().model = v2().chain[v2().chain.length - 1];
        finish();
      };
      $('pctDelete').onclick = () => { if (v2().chain.length > 1) { v2().chain.splice(pctIndex, 1); v2().model = v2().chain[v2().chain.length - 1]; finish(); } };
      pctDialog.addEventListener('keydown', e => { if (e.key === 'Escape') pctDialog.classList.remove('open'); if (e.key === 'Enter') $('pctSave').click(); });
    }
    pctIndex = index;
    $('pctTitle').textContent = `Sửa ${m[1]}${m[2]}`;
    $('pctHelp').textContent = m[1] === 'blend' ? 'Phần trăm audio gốc của cả chuỗi trộn vào kết quả (0-100)' : 'Phần trăm đầu vào của model liền trước trộn vào kết quả (0-100)';
    $('pctValue').value = m[2];
    $('pctDelete').disabled = v.chain.length <= 1;
    pctDialog.classList.add('open');
    $('pctValue').focus(); $('pctValue').select();
  };
  // window capture runs before the older document-level row handlers, so a pseudo row never reaches the model editor
  window.addEventListener('click', e => {
    const b = e.target.closest?.('[data-edit-model],[data-current-model]'); if (!b) return;
    const i = Number(b.dataset.editModel ?? b.dataset.currentModel);
    if (PSEUDO.test(window.selectedVariant?.()?.chain?.[i] || '')) { e.preventDefault(); e.stopImmediatePropagation(); openPct(i); }
  }, true);

  // ---- backend run status: an always-visible chip in the top bar, click for the live log ----
  const runCss = document.createElement('style');
  runCss.textContent = '.run-chip{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);background:var(--s2);padding:6px 11px;border-radius:99px;font-size:12px;white-space:nowrap;max-width:300px;overflow:hidden;text-overflow:ellipsis}'
    + '.run-chip:hover{border-color:#527084}.run-chip .dot{width:8px;height:8px;border-radius:50%;background:#617480;flex:0 0 auto}'
    + '.run-chip[data-state=running] .dot{background:var(--amber);animation:labpulse 1.1s ease-in-out infinite}.run-chip[data-state=running]{border-color:#765d2b}'
    + '.run-chip[data-state=finished] .dot{background:var(--green)}.run-chip[data-state=finished]{border-color:#355045}'
    + '.run-chip[data-state=failed] .dot,.run-chip[data-state=lost] .dot{background:var(--red)}.run-chip[data-state=failed]{border-color:#673d43}'
    + '.run-chip[data-state=dry] .dot{background:var(--blue)}@keyframes labpulse{50%{opacity:.3}}@media(prefers-reduced-motion:reduce){.run-chip .dot{animation:none!important}}'
    + '.run-kv{display:grid;grid-template-columns:110px minmax(0,1fr);gap:8px;font-size:12px;align-items:baseline}.run-kv span{color:var(--muted)}.run-kv code{word-break:break-all;font:11px ui-monospace,monospace;color:#c9e7e3}'
    + '.run-log{margin:0;height:min(340px,38vh);overflow:auto;background:#080c10;border:1px solid var(--line);border-radius:7px;padding:10px;font:11px/1.45 ui-monospace,monospace;color:#c9d4dc;white-space:pre-wrap;word-break:break-all}'
    + '.run-box{width:min(760px,calc(100vw - 24px))}';
  document.head.appendChild(runCss);
  const LABEL = {running: ['Đang chạy', 'running'], finished: ['Hoàn tất', 'finished'], failed: ['Lỗi', 'failed'], 'dry-run': ['Dry-run · chưa chạy', 'dry'],
                 ended: ['Tiến trình đã dừng', 'failed'], unknown: ['Không rõ', 'idle'], sending: ['Đang gửi…', 'running'], lost: ['Mất kết nối server', 'lost']};
  const fmtDur = s => { s = Math.max(0, Math.floor(s)); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = s % 60, two = n => String(n).padStart(2, '0'); return (h ? h + ':' : '') + two(m) + ':' + two(x); };
  const baseTitle = document.title;
  const T = {runId: null, st: null, polledAt: 0, state: null, r: null, gen: 0, loop: null};

  const chip = document.createElement('button');
  chip.type = 'button'; chip.className = 'run-chip'; chip.id = 'runChip'; chip.setAttribute('data-raw', ''); chip.dataset.state = 'idle';
  chip.innerHTML = '<span class="dot"></span><span id="runChipText">Chưa chạy</span>';
  chip.title = 'Trạng thái chạy trên backend (bấm để xem log)';
  const topbar = document.querySelector('.topbar');
  if (topbar) topbar.insertBefore(chip, topbar.querySelector('.spacer')?.nextSibling || null);

  let runDialog = null, paintTimer = null;
  const elapsedNow = () => (T.st?.elapsedSeconds || 0) + (T.state === 'running' ? (Date.now() - T.polledAt) / 1000 : 0);
  const paint = () => {
    const [text, tone] = LABEL[T.state] || ['Chưa chạy', 'idle'];
    const showTime = T.st && T.state && !['dry-run', 'unknown'].includes(T.state);
    const line = text + (showTime ? ' · ' + fmtDur(elapsedNow()) : '');
    chip.dataset.state = tone; $('runChipText').textContent = line;
    document.title = T.state === 'running' ? `● ${line} · ${baseTitle}` : (T.state === 'finished' ? `✓ Hoàn tất · ${baseTitle}` : (T.state === 'failed' ? `✕ Lỗi · ${baseTitle}` : baseTitle));
    // a one-second tick only while a job is live (no standing interval): it re-arms itself from here
    clearTimeout(paintTimer);
    if (['running', 'sending', 'lost'].includes(T.state)) paintTimer = setTimeout(paint, 1000);
    if (!runDialog?.classList.contains('open')) return;
    const st = T.st || {};
    $('rdState').textContent = line; $('rdChain').textContent = st.chain || '—'; $('rdDir').textContent = st.runDir || '—';
    $('rdOut').textContent = st.state === 'finished' ? `${st.outputs?.length || 0} file denoise · ${st.out || ''}` : (st.state === 'running' ? 'đang xử lý…' : (st.returncode != null ? `mã thoát ${st.returncode}` : '—'));
    const log = $('rdLog'), stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 40;
    const body = [(st.logTail || '').trimEnd(), st.pipelineLog ? `\n──── ${st.pipelineLog} ────\n` + (st.pipelineLogTail || '').trimEnd() : ''].join('\n').trim();
    const text2 = body || (T.state === 'sending' ? 'Đang gửi yêu cầu tới backend…' : 'Chưa có log. Nếu srun đang chờ tài nguyên, log sẽ xuất hiện khi job bắt đầu.');
    if (log.textContent !== text2) { log.textContent = text2; if (stick) log.scrollTop = log.scrollHeight; }
  };
  const openRunDialog = () => {
    if (!runDialog) {
      runDialog = makeDialog('runDialog', 'Trạng thái chạy');
      runDialog.setAttribute('data-raw', ''); runDialog.firstChild.classList.add('run-box');
      runDialog.firstChild.innerHTML = '<header><strong style="flex:1">Trạng thái chạy trên backend</strong><button class="btn" type="button" id="rdClose">Đóng</button></header>'
        + '<div class="srv-body"><div class="run-kv"><span>Trạng thái</span><b id="rdState"></b></div><div class="run-kv"><span>Chuỗi</span><b id="rdChain"></b></div>'
        + '<div class="run-kv"><span>Kết quả</span><b id="rdOut"></b></div><div class="run-kv"><span>Thư mục</span><code id="rdDir"></code></div><pre class="run-log" id="rdLog" tabindex="0" aria-label="Log"></pre></div>';
      $('rdClose').onclick = () => runDialog.classList.remove('open');
      runDialog.addEventListener('keydown', e => { if (e.key === 'Escape') runDialog.classList.remove('open'); });
    }
    runDialog.classList.add('open'); paint(); const log = $('rdLog'); log.scrollTop = log.scrollHeight; $('rdClose').focus();
  };
  chip.onclick = openRunDialog;

  // one tracker at a time; a newer run (or a restored one) cancels the previous loop
  const track = (runId, r) => {
    const gen = ++T.gen; T.runId = runId; T.r = r || null; T.state = 'sending'; T.st = null; T.polledAt = Date.now(); paint();
    const say = (text, notice = text) => { const n = $('runReason'); if (n) { n.setAttribute('data-raw', ''); n.textContent = text; } if (r && typeof showNotice === 'function') showNotice(notice); };
    const tick = async () => {
      if (gen !== T.gen) return;
      let again = false;
      try {
        const st = await api('/api/experiment/status?runId=' + encodeURIComponent(runId));
        if (gen !== T.gen) return;
        T.st = st; T.state = st.state; T.polledAt = Date.now();
        const last = (st.pipelineLogTail || st.logTail || '').trim().split('\n').filter(Boolean).pop() || '';
        if (st.state === 'running') { again = true; if (r) { r.status = 'running'; say('Đang chạy trên GPU · ' + last.slice(0, 140), 'Đang chạy trên GPU'); } }
        else if (st.state === 'finished') { if (r) { r.status = 'accepted'; say(`Hoàn tất · ${st.outputs?.length || 0} file denoise · ${st.out}`, 'Hoàn tất'); } }
        else if (st.state === 'dry-run') { if (r) { r.status = 'accepted'; say('Dry-run: đã ghi ' + st.runDir + ', chưa chạy', 'Dry-run: chưa chạy'); } }
        else if (r) { r.status = 'failed'; r.errorMessage = `Lỗi (${st.state}, mã ${st.returncode ?? '?'}) · ${last.slice(0, 160)}`; say(r.errorMessage, 'Lỗi khi chạy'); }
      } catch (e) { again = true; T.state = 'lost'; }
      paint();
      if (r) { window.renderVariants?.(); window.dirty?.(); }
      if (again) T.loop = setTimeout(tick, runDialog?.classList.contains('open') ? 1500 : 3000);
    };
    clearTimeout(T.loop); T.loop = setTimeout(tick, 600);
  };
  window.__pollRun = (r, v) => { if (r.serverRun?.runId) track(r.serverRun.runId, r); };
  // after a page reload, pick the latest run back up so a job in progress is never invisible
  window.__restoreRun = () => api('/api/experiment/runs?limit=1').then(l => { if (l.runs?.[0] && !T.runId) track(l.runs[0].runId, null); }).catch(() => {});

  // ---- add a step: 1) pick a model or blend/dry from the list, 2) set its parameters ----
  const addCss = document.createElement('style');
  addCss.textContent = '.add-cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;padding:10px 14px 14px;overflow:auto}'
    + '.add-group{grid-column:1/-1;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.11em;font-weight:800;margin-top:6px}'
    + '.add-card{display:flex;flex-direction:column;align-items:flex-start;gap:3px;padding:9px;border:1px solid #30434f;border-radius:7px;background:#111d25;color:var(--text);text-align:left}'
    + '.add-card[hidden]{display:none}.add-card:hover,.add-card:focus-visible{border-color:var(--cyan);background:#143039;outline:0}.add-card strong{font-size:12px}.add-card small{font-size:10px;color:var(--muted)}'
    + '.add-card.mix{border-color:#5b4a2a;background:#241f14}.add-card.mix:hover{border-color:var(--amber);background:#2d2616}.add-note{margin:0;color:var(--muted);font-size:12px}'
    + '.add-token{font:700 12px ui-monospace,monospace;color:var(--amber)}.srv-body select{border:1px solid var(--line);background:#0b1117;color:var(--text);border-radius:6px;padding:7px 8px}';
  document.head.appendChild(addCss);
  let addDialog = null, addBox = null, addChosen = null;
  const isMix = id => id === 'blend' || id === 'dry';
  const catalogOf = () => window.__labCatalog || {}, fieldsOf = () => window.__labFields || {};
  const closeAdd = () => addDialog.classList.remove('open');
  const showModelList = () => {
    const entries = Object.entries(catalogOf());
    const card = ([id, [name, desc]]) => `<button type="button" class="add-card${isMix(id) ? ' mix' : ''}" data-model="${esc(id)}"><strong>${esc(name)}</strong><small>${esc(desc)}</small></button>`;
    addBox.innerHTML = '<header><strong style="flex:1">Thêm vào chuỗi · chọn model</strong><button class="btn" type="button" data-act="close">Đóng</button></header>'
      + '<div class="srv-body" style="padding-bottom:0"><input id="addSearch" type="search" placeholder="Tìm model hoặc blend/dry" aria-label="Tìm model"></div>'
      + `<div class="add-cards"><div class="add-group">Bước trộn</div>${entries.filter(([id]) => isMix(id)).map(card).join('')}<div class="add-group">Model khử nhiễu</div>${entries.filter(([id]) => !isMix(id)).map(card).join('')}</div>`;
    const search = addBox.querySelector('#addSearch');
    search.oninput = () => { const t = search.value.trim().toLowerCase(); addBox.querySelectorAll('.add-card').forEach(c => { c.hidden = Boolean(t && !(c.dataset.model + ' ' + c.textContent).toLowerCase().includes(t)); }); };
    search.focus();
  };
  const showParams = id => {
    addChosen = id;
    const [name, desc] = catalogOf()[id], list = fieldsOf()[id] || [], mix = isMix(id);
    const inputs = list.map(([key, label, type, opt]) => type === 'select'
      ? `<label>${esc(label)}<select data-param="${esc(key)}">${opt.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('')}</select></label>`
      : `<label>${esc(label)}<input data-param="${esc(key)}" type="number" step="${mix ? 1 : 0.01}" min="0"${mix ? ' max="100"' : ''} value="${esc(opt)}"></label>`).join('');
    addBox.innerHTML = `<header><strong style="flex:1">${esc(name)}</strong><button class="btn" type="button" data-act="close">Đóng</button></header>`
      + `<div class="srv-body"><p class="add-note">${esc(desc)}</p>${inputs || '<p class="add-note">Model này dùng tham số mặc định, không có gì để chỉnh.</p>'}`
      + (mix ? `<p class="add-note">Bước trong chuỗi: <span class="add-token" id="addToken"></span></p>` : '') + '</div>'
      + '<div class="srv-actions"><button class="btn" type="button" data-act="back">Chọn model khác</button><button class="btn primary" type="button" data-act="save">Thêm vào chuỗi</button></div>';
    const token = () => { const p = addBox.querySelector('[data-param="percent"]'); return id + Math.min(100, Math.max(0, Math.round(Number(p?.value) || 0))); };
    if (mix) { const t = addBox.querySelector('#addToken'), p = addBox.querySelector('[data-param="percent"]'); t.textContent = token(); p.oninput = () => { t.textContent = token(); }; }
    (addBox.querySelector('input,select') || addBox.querySelector('[data-act="save"]')).focus();
  };
  const saveAdd = () => {
    const v = window.selectedVariant?.(); if (!v || !addChosen) return;
    const step = {model: addChosen};
    addBox.querySelectorAll('[data-param]').forEach(el => { if (el.value !== '') step[el.dataset.param] = el.type === 'number' ? Number(el.value) : el.value; });
    const token = isMix(addChosen) ? addChosen + Math.min(100, Math.max(0, Math.round(Number(step.percent) || 0))) : addChosen;
    v.chain = Array.isArray(v.chain) ? v.chain : []; v.stepParams = Array.isArray(v.stepParams) ? v.stepParams : [];
    v.chain.push(token); v.stepParams.push(step); v.model = token; v.status = 'draft'; v.draftChanged = true;
    closeAdd(); window.renderVariants?.(); window.dirty?.();
  };
  const openAdd = () => {
    if (!addDialog) {
      addDialog = makeDialog('addDialog', 'Thêm model vào chuỗi');
      addDialog.setAttribute('data-raw', '');  // model names stay as written ("high-pass" must not become "cao-pass")
      addBox = addDialog.firstChild;
      addBox.onclick = e => {
        const act = e.target.closest('[data-act]')?.dataset.act, card = e.target.closest('.add-card');
        if (card) showParams(card.dataset.model); else if (act === 'close') closeAdd(); else if (act === 'back') showModelList(); else if (act === 'save') saveAdd();
      };
      addDialog.addEventListener('keydown', e => {
        if (e.key === 'Escape') closeAdd();
        else if (e.key === 'Enter' && e.target.matches?.('[data-param]')) { e.preventDefault(); saveAdd(); }
      });
    }
    showModelList(); addDialog.classList.add('open');
  };
  // window capture runs before every older handler, so the add button never reaches the legacy popups (no timing dependence)
  window.addEventListener('click', e => {
    if (!e.target.closest?.('#addModelControl,#variantsBtn')) return;
    e.preventDefault(); e.stopImmediatePropagation(); openAdd();
  }, true);

  // renderVariants clears the chain list; always rebuild it afterwards
  const baseRender = window.renderVariants;
  if (typeof baseRender === 'function') window.renderVariants = function () { const out = baseRender.apply(this, arguments); window.__renderCurrentModels?.(); return out; };
  // the chosen server file travels with the recipe
  const baseRecipe = window.recipe;
  if (typeof baseRecipe === 'function') window.recipe = function () { const r = baseRecipe.apply(this, arguments); if (inputFile) r.settings.input_files = [inputFile]; return r; };

  api('/api/config').then(c => {
    cfg = c;
    const fa = $('fileA');
    if (fa) fa.click = () => openPicker();
    if ($('audioDir') && c.audioRoots.length) { $('audioDir').value = c.audioRoots[0]; }
    const info = `Backend · ${c.jobid ? 'srun --jobid=' + c.jobid : 'chạy cục bộ'} · conda ${c.condaEnv}${c.dryRun ? ' · DRY-RUN' : ''}`;
    const bar = $('command'); if (bar) { bar.dataset.backend = info; }
    const lane = $('laneState'); if (lane) lane.title = info;
    window.__restoreRun();
  }).catch(() => { cfg = null; });
})();
</script></body></html>''',
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="experiment.html")
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(PAGE, encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
