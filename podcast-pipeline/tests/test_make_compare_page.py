import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))
import make_compare_page as mcp  # noqa: E402

FIXTURE = TOOLS / "fixtures" / "compare_data_example.json"
sf = pytest.importorskip("soundfile")
np = pytest.importorskip("numpy")


def make_wavs(data, wav_dir, sr=16000):
    """Write a tone for every audio path in `data`; returns data with the paths pointed at them."""
    wav_dir.mkdir(parents=True, exist_ok=True)
    for f in data["files"]:
        for i, (src, _) in enumerate(sorted(f["audio"].items())):
            name = f"{f['name']}_{src.replace('+', '_')}.wav"
            f["audio"][src] = str(wav_dir / name)
        for src, p in sorted(f["audio"].items()):
            if not os.path.exists(p):
                t = np.arange(int(sr * f["duration"])) / sr
                sf.write(p, (0.2 * np.sin(2 * np.pi * (200 + 50 * len(src)) * t)).astype("float32"), sr)
        f["audio"]["none"] = f["audio"]["raw"]
    return data


@pytest.fixture
def data(tmp_path):
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return make_wavs(d, tmp_path / "wav")


def build(tmp_path, data, *args, name="compare.html"):
    src = tmp_path / "compare_data.json"
    src.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / name
    mcp.main([str(src), "-o", str(out), *args])
    return out.read_text(encoding="utf-8")


def embedded(page, ident):
    m = re.search(rf'<script type="application/json" id="{ident}">(.*?)</script>', page, re.S)
    return json.loads(m.group(1))


def script_of(page):
    return re.search(r"<script>(.*)</script>\s*</body>", page, re.S).group(1)


def test_structure_tabs_chips_and_banner(tmp_path, data):
    page = build(tmp_path, data)
    assert page.startswith("<!doctype html>") and 'charset="utf-8"' in page
    for t in ("text", "speakers", "metrics"):
        assert f'id="tab-{t}"' in page and f'data-tab="{t}"' in page
    assert page.count('<button type="button" class="chip') == 4
    assert 'data-vid="dfn3+mf2se48"' in page and "DFN3 &gt; MF2-SE48" in page
    assert "Different does not mean better" in page
    assert "prefers-color-scheme:dark" in page
    assert 'name="viewport"' in page
    assert "tap_01" in page and "tap_02" in page
    assert "http://" not in re.sub(r"http://www\.w3\.org/\S+", "", page) and "https://" not in page


def test_embedded_data_keeps_vietnamese_and_nulls(tmp_path, data):
    page = build(tmp_path, data)
    d = embedded(page, "data")
    assert "cà phê rang xay" in page  # not \u-escaped
    f = d["files"][0]
    assert f["metrics"]["dfn3"]["separation"] is None
    assert "audio" in f and f["audio"] == {}  # wav paths do not leak into the page
    assert f["text_diff"]["dfn3"]["segments"][0]["_k"].startswith("0/")


def test_js_has_diff_markup_for_every_op(tmp_path, data):
    s = script_of(build(tmp_path, data))
    for needle in ('class="w sub"', '<del class="w del">', '<ins class="w ins">', "band band-", "rrow", "DASH"):
        assert needle in s
    css = build(tmp_path, data)
    for cls in (".w.sub", "del.w.del", "ins.w.ins", ".band-swap", ".band-missing", ".band-extra", ".band-overlap"):
        assert cls in css


NODE = shutil.which("node")


def run_node(page, expr):
    """Run the inline script outside a browser (no document: it only exports helpers)."""
    d = json.dumps(embedded(page, "data"))
    code = (f"const window=globalThis;{script_of(page)}\n"
            f"const c=globalThis.__cmp;c.setData({d},{{}});console.log(JSON.stringify({expr}));")
    r = subprocess.run([NODE, "-e", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_inline_script_parses(tmp_path, data):
    js = tmp_path / "inline.js"
    js.write_text(script_of(build(tmp_path, data)), encoding="utf-8")
    r = subprocess.run([NODE, "--check", str(js)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_diff_rendering_per_op(tmp_path, data):
    page = build(tmp_path, data)
    ops = [["equal", "a"], ["sub", "b", "v"], ["del", "x<y"], ["ins", "z"]]
    base = run_node(page, f"c.diffHtml({json.dumps(ops)},'base')")
    var = run_node(page, f"c.diffHtml({json.dumps(ops)},'var')")
    assert base == 'a <span class="w sub">b</span> <del class="w del">x&lt;y</del>'
    assert var == 'a <span class="w sub">v</span> <ins class="w ins">z</ins>'
    assert run_node(page, "c.diffHtml([],'base','plain <b>')") == "plain &lt;b&gt;"


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_disagreement_bands_list_and_null_rendering(tmp_path, data):
    page = build(tmp_path, data)
    rs = data["files"][0]["speakers"]["mf2se48"]["disagree_ranges"]
    svg = run_node(page, "c.timelineSvg({t0:0,t1:20,lanes:[{label:'b',segs:[],cls:function(){return 0}}],"
                         f"bands:{json.dumps(rs)},sel:1}})")
    for kind in ("swap", "missing", "extra"):
        assert f"band band-{kind}" in svg
    assert svg.count('class="band ') == 3 and "sel" in svg
    lst = run_node(page, f"c.rangesHtml({json.dumps(rs)},-1,200)")
    assert lst.count('class="rrow') == 3 and "kind-swap" in lst and "—" in lst
    assert run_node(page, "[c.fmt(null),c.fmt(undefined),c.fmt(0),c.fmt(true)]") == ["—", "—", "0", "true"]
    metrics = run_node(page, "c.metricsHtml('file',%s)" % json.dumps(data["files"][0]))
    assert "—" in metrics and "separation" in metrics.lower()
    assert "closer to 0 is better" in metrics and "informative" in metrics
    assert run_node(page, "c.judge('asr.wer_vs_baseline',0.2,0)") == "bad"
    assert run_node(page, "c.judge('audio.noise_floor_db',-60,-50)") == ""
    assert run_node(page, "c.judge('audio.length_equal',false,true)") == "bad"


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_page_logic_tolerates_empty_data(tmp_path):
    page = build(tmp_path, {"version": 1, "baseline": "none", "variants": [{"id": "none"}], "files": []})
    assert run_node(page, "c.metricsHtml('summary',null)").find("No metrics") >= 0
    assert "No metrics" in run_node(page, "c.summaryHtml()")


def test_embed_clips_count_and_cap(tmp_path, data):
    page = build(tmp_path, data, "--embed", "clips")
    d, clips = embedded(page, "data"), embedded(page, "clips")
    n_uri = page.count("data:audio/wav;base64,")
    assert n_uri == d["embed"]["clips"] > 0
    assert n_uri == sum(len(v) for v in clips.values())
    assert d["embed"]["left_out"] == 0
    tiny = build(tmp_path, data, "--embed", "clips", "--max-mb", "0.05", name="small.html")
    ds = embedded(tiny, "data")
    assert 0 < ds["embed"]["clips"] < d["embed"]["clips"]
    assert ds["embed"]["left_out"] == d["embed"]["clips"] - ds["embed"]["clips"]
    assert ds["embed"]["bytes"] <= 0.05 * 1024 * 1024
    zero = build(tmp_path, data, "--embed", "clips", "--max-mb", "0", name="zero.html")
    assert "data:audio" not in zero and embedded(zero, "data")["embed"]["left_out"] > 0


def test_top_segments_limits_clips(tmp_path, data):
    all_ = embedded(build(tmp_path, data, "--top-segments", "100"), "data")["embed"]["clips"]
    few = embedded(build(tmp_path, data, "--top-segments", "1", name="few.html"), "data")["embed"]["clips"]
    assert few < all_


def test_embed_link_and_none(tmp_path, data):
    link = build(tmp_path, data, "--embed", "link")
    urls = embedded(link, "data")["embed"]["urls"]
    assert urls["0"]["raw"].endswith("tap_01_raw.wav") and not urls["0"]["raw"].startswith("/")
    assert "data:audio" not in link
    none = build(tmp_path, data, "--embed", "none", name="none.html")
    d = embedded(none, "data")
    assert "data:audio" not in none and d["embed"]["urls"] == {} and embedded(none, "clips") == {}


def test_missing_wavs_degrade(tmp_path, data):
    for f in data["files"]:
        f["audio"] = {k: str(tmp_path / "nope" / (k + ".wav")) for k in f["audio"]}
    page = build(tmp_path, data)
    assert "data:audio" not in page and embedded(page, "data")["embed"]["clips"] == 0
    assert embedded(build(tmp_path, data, "--embed", "link", name="l.html"), "data")["embed"]["urls"] == {"0": {}, "1": {}}
    for f in data["files"]:
        f.pop("audio")
    assert "tab-text" in build(tmp_path, data, name="noaudio.html")


def test_missing_soundfile_degrades(tmp_path, data, monkeypatch):
    monkeypatch.setitem(sys.modules, "soundfile", None)
    page = build(tmp_path, data)
    d = embedded(page, "data")
    assert d["embed"]["clips"] == 0 and "soundfile" in d["embed"]["note"]


def test_escaping_nasty_strings(tmp_path, data):
    nasty = "</script><b>x</b> & \"q\" __DATA__  "
    f = data["files"][0]
    f["name"] = nasty
    data["variants"][1]["label"] = nasty
    data["variants"][1]["note"] = nasty
    f["text_diff"]["dfn3"]["segments"][0]["variant_text"] = nasty
    f["text_diff"]["dfn3"]["segments"][0]["ops"] = [["sub", "a", nasty]]
    f["speakers"]["mf2se48"]["disagree_ranges"][0]["variant"] = nasty
    page = build(tmp_path, data)
    assert "</script><b>" not in page
    assert page.count("</script>") == 3  # data, clips, script: none leaks from content
    assert embedded(page, "data")["files"][0]["name"] == nasty
    assert nasty not in page
    assert "&lt;/script&gt;&lt;b&gt;x" in page


def test_deterministic(tmp_path, data):
    a = build(tmp_path, data)
    b = build(tmp_path, data)
    assert a == b


def test_files_filter(tmp_path, data):
    page = build(tmp_path, data, "--files", "tap_02")
    d = embedded(page, "data")
    assert [f["name"] for f in d["files"]] == ["tap_02"]
    with pytest.raises(SystemExit):
        build(tmp_path, data, "--files", "zzz", name="x.html")


def test_baseline_only_and_empty_segments(tmp_path, data):
    d = {"version": 1, "baseline": "none", "variants": [{"id": "none", "chain": [], "label": "No filter"}],
         "files": [{"name": "a", "duration": 3.0, "audio": {}, "text": {}, "text_diff": {}, "speakers": {}, "metrics": {}}],
         "summary": {}}
    page = build(tmp_path, d)
    assert page.count('<button type="button" class="chip') == 1 and embedded(page, "data")["files"][0]["name"] == "a"
    d["files"] = []
    assert "tab-text" in build(tmp_path, d, name="e.html")


def test_many_segments_page_is_paginated(tmp_path, data):
    segs = [{"start": i * 0.5, "end": i * 0.5 + 0.4, "baseline_text": "a b", "variant_text": "a c",
             "ops": [["equal", "a"], ["sub", "b", "c"]], "wer": 0.5} for i in range(5000)]
    data["files"][0]["text_diff"]["dfn3"]["segments"] = segs
    page = build(tmp_path, data)
    assert "S.shown" in script_of(page) and "tmore" in script_of(page)  # rendered 50 at a time
    assert "<article" not in page.split("<script")[0]  # no static cards in the HTML


def test_old_review_page_untouched():
    path = "podcast-pipeline/tools/make_review_page.py"
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=TOOLS, capture_output=True, text=True)
    if top.returncode != 0:
        pytest.skip("not a git checkout")
    head = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=top.stdout.strip(), capture_output=True)
    if head.returncode != 0:
        pytest.skip("review page not in HEAD")
    assert (TOOLS / "make_review_page.py").read_bytes() == head.stdout


def _with_mos(data, vals):
    for f in data["files"]:
        for vid, m in f["metrics"].items():
            m["audio"]["learned_mos"] = vals
            m["audio"]["learned_mos_delta"] = vals
    return data


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_learned_mos_block_with_and_without_keys(tmp_path, data):
    f = json.dumps(data["files"][0])
    old = run_node(build(tmp_path, data), f"c.metricsHtml('file',{f})")
    assert "Learned MOS" not in old  # old compare_data.json: no block, page still renders
    data = _with_mos(data, {"mos": 0.25, "dnsmos_ovrl": -0.5, "scoreq": None, "utmos": 0.0, "nisqa_mos": 0.1})
    f = json.dumps(data["files"][0])
    html = run_node(build(tmp_path, data, name="m.html"), f"c.metricsHtml('file',{f})")
    assert "Learned MOS (delta vs raw, informational)" in html
    assert "hallucinated" in html and "never a gate" in html
    assert "+0.25" in html and "-0.5" in html and "—" in html
    assert "learned_mos_delta" not in html.split("Learned MOS")[0]  # not in the generic audio table


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_learned_mos_null_renders_dash(tmp_path, data):
    data = _with_mos(data, None)
    f = json.dumps(data["files"][0])
    html = run_node(build(tmp_path, data), f"c.metricsHtml('file',{f})")
    block = html.split("Learned MOS")[1]
    assert block.count("—") >= 5


def test_group_tag_next_to_chip(tmp_path, data):
    data["variants"][1]["group"] = "analysis_only"
    page = build(tmp_path, data)
    assert page.count('<span class="grp">analysis_only</span>') == 1
    assert page.count('<button type="button" class="chip') == 4


# ---- degraded / suspect variants -------------------------------------------------
def _mark(data, vid, **kw):
    next(v for v in data["variants"] if v["id"] == vid).update(kw)


def test_degraded_and_suspect_badges_banner(tmp_path, data):
    _mark(data, "dfn3", status="degraded", degraded=True, degraded_files=2, total_files=3,
          degraded_reason="fell back to unfiltered audio in 2/3 files: dfn3: no <weights>")
    _mark(data, "mf2se48", suspect_noop_files=1)
    page = build(tmp_path, data)
    assert page.count('class="badge bad"') == 1
    assert "fell back to unfiltered audio in 2/3 files" in page
    assert page.count('class="badge warn"') == 1 and "output identical to input" in page
    assert 'class="banner bad" role="alert"' in page and "Degraded variants" in page
    assert "no &lt;weights&gt;" in page and "no <weights>" not in page.split("<script")[0]
    assert "chip degraded" in page


def test_page_without_the_new_keys_has_no_badges(tmp_path, data):
    page = build(tmp_path, data)
    assert 'class="badge' not in page.split("<style>")[1].split("</style>")[1].split("<script")[0]
    assert 'role="alert"' not in page.split("<script")[0]
    assert page.count('<button type="button" class="chip') == 4


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_per_file_warning_and_summary_note(tmp_path, data):
    f = data["files"][0]
    f["denoise"] = {"dfn3": {"verdict": "fallback", "errors": ["dfn3: no <weights>\ntrace"], "corr_with_input": 1.0},
                    "mf2se48": {"verdict": "suspect_noop", "errors": [], "corr_with_input": 1.0},
                    "dfn3+mf2se48": {"verdict": "ok", "errors": []}}
    data["summary_n"] = {"dfn3": 1, "mf2se48": 2}
    data["summary_excluded_degraded"] = {"dfn3": 1, "mf2se48": 0}
    page = build(tmp_path, data)
    expr = "[c.denoiseWarn(%s,'dfn3'),c.denoiseWarn(%s,'mf2se48'),c.denoiseWarn(%s,'dfn3+mf2se48'),c.denoiseWarn(%s,'none')]"
    fj = json.dumps(embedded(page, "data")["files"][0])
    w = run_node(page, expr % (fj, fj, fj, fj))
    assert 'role="alert"' in w[0] and "First error: dfn3: no &lt;weights&gt;" in w[0] and "trace" not in w[0]
    assert "identical to its input" in w[1] and "vwarn warn" in w[1]
    assert w[2] == "" and w[3] == ""
    note = run_node(page, "c.summaryNote()")
    assert "DeepFilterNet3: mean over 1/" in note and "1 fell back to unfiltered audio" in note


@pytest.mark.skipif(not NODE, reason="node not on PATH")
def test_warning_and_note_absent_in_old_data(tmp_path, data):
    page = build(tmp_path, data)
    fj = json.dumps(embedded(page, "data")["files"][0])
    assert run_node(page, "c.denoiseWarn(%s,'dfn3')" % fj) == ""
    assert run_node(page, "c.summaryNote()") == ""
    assert "Summary" not in run_node(page, "c.metricsHtml('summary',%s)" % fj)
