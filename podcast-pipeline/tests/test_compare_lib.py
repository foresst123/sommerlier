import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.compare_lib import metrics as M
from tools.compare_lib.build import build_compare_data
from tools.compare_lib.io_stage import load_file_output
from tools.compare_lib.speaker_align import compare_speakers
from tools.compare_lib.text_align import clean_text, compare_transcripts, edit_ops

SR = 16000


def seg(s, e, spk="A", text="", words=None):
    return {"start": s, "end": e, "speaker": spk, "text": text, "words": words}


def test_clean_text_keeps_diacritics():
    assert clean_text("Xin chào, Việt Nam!") == "xin chào việt nam"
    assert clean_text("Café") == "café"  # NFC
    assert clean_text(None) == ""


def test_edit_ops_counts():
    ops = edit_ops("a b c d".split(), "a x c d e".split())
    from tools.compare_lib.text_align import count_ops
    assert count_ops(ops) == {"sub": 1, "ins": 1, "del": 0, "equal": 3}
    assert ["sub", "b", "x"] in ops and ["ins", "e"] in ops
    assert count_ops(edit_ops(["a", "b"], ["b"])) == {"sub": 0, "ins": 0, "del": 1, "equal": 1}


def test_text_diff_known_edits_and_diacritics():
    base = [seg(0, 2, text="tôi đi học"), seg(2, 4, text="hôm nay trời đẹp")]
    var = [seg(0, 2, text="toi đi học"), seg(2, 4, text="hôm nay đẹp")]
    r = compare_transcripts(base, var)
    assert r["ops"] == {"sub": 1, "ins": 0, "del": 1, "equal": 5}  # tôi != toi
    assert r["wer"] == pytest.approx(2 / 7)
    assert r["segments"][0]["wer"] == pytest.approx(1 / 3)
    assert r["segments"][1]["ops"][1] == ["del", "nay"] or ["del", "trời"] in r["segments"][1]["ops"]


def test_time_alignment_split_and_merge():
    base = [seg(0, 2, text="một hai"), seg(2, 4, text="ba bốn")]
    merged = [seg(0, 4, text="một hai ba bốn")]  # merged, uniform words
    r = compare_transcripts(base, merged)
    assert r["wer"] == 0.0
    split = [seg(0, 1, text="một"), seg(1, 3, text="hai ba"), seg(3, 4, text="bốn")]
    r = compare_transcripts(base, split)
    assert r["ops"]["equal"] == 4 and r["wer"] == 0.0
    # word timestamps decide the segment a word lands in
    words = [{"word": "ba", "start": 0.1, "end": 0.3}, {"word": "bốn", "start": 0.3, "end": 0.5}]
    r = compare_transcripts(base, [seg(0, 4, text="ba bốn", words=words)])
    assert r["segments"][0]["variant_text"] == "ba bốn" and r["ops"]["sub"] == 2 and r["ops"]["del"] == 2


def test_empty_segments_and_missing():
    assert compare_transcripts(None, []) is None
    r = compare_transcripts([seg(0, 1, text="")], [seg(0, 1, text="")])
    assert r["wer"] is None and r["segments"][0]["wer"] == 0.0
    r = compare_transcripts([], [seg(0, 1, text="a b")])
    assert r["ops"]["ins"] == 2


def test_speaker_mapping_with_permuted_labels():
    base = [seg(0, 5, "A"), seg(5, 10, "B")]
    var = [seg(0, 5, "X"), seg(5, 10, "Y")]
    r = compare_speakers(base, var, 10)
    assert r["label_map"] == {"X": "A", "Y": "B"}
    assert r["disagree_seconds"] == 0 and r["der_vs_baseline"] == 0
    var = [seg(0, 5, "Y"), seg(5, 10, "X")]
    assert compare_speakers(base, var, 10)["label_map"] == {"Y": "A", "X": "B"}


def test_disagreement_kinds_and_der():
    base = [seg(0, 4, "A"), seg(4, 6, "B"), seg(8, 10, "A")]
    var = [seg(0, 4, "A"), seg(4, 5, "B"), seg(6, 7, "B"), seg(8, 9, "A"), seg(8, 10, "B")]
    # baseline B at 4-6 ; variant B 4-5 ok, 5-6 missing, 6-7 extra, 8-9 overlap (A+B vs A), 9-10 swap
    r = compare_speakers(base, var, 10)
    assert r["label_map"] == {"A": "A", "B": "B"}
    kinds = {(d["start"], d["end"]): d["kind"] for d in r["disagree_ranges"]}
    assert kinds[(5.0, 6.0)] == "missing"
    assert kinds[(6.0, 7.0)] == "extra"
    assert kinds[(8.0, 9.0)] == "overlap"
    assert kinds[(9.0, 10.0)] == "swap"
    assert r["disagree_seconds"] == pytest.approx(4.0, abs=0.02)
    # baseline speaker-seconds 4+2+2=8; missed 1 + false alarm 1(6-7) + 1(8-9 extra B) + confusion 1 = 4
    assert r["der_vs_baseline"] == pytest.approx(4 / 8, abs=0.01)
    assert r["speaker_seconds"]["B"] == pytest.approx(4.0)


def tone(freq, secs=1.0, amp=0.5):
    t = np.arange(int(SR * secs)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_hf_energy_ratio():
    assert M.hf_energy_ratio(tone(6000), SR) > 0.99
    assert M.hf_energy_ratio(tone(1000), SR) < 0.01


def test_lag_detection_on_shifted_wav():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(SR * 2).astype(np.float32) * 0.1
    var = np.concatenate([np.zeros(240, np.float32), raw])[:len(raw)]
    assert M.estimate_lag(raw, var, SR) == 240
    assert M.estimate_lag(var, raw, SR) == -240
    assert M.aligned_corr(raw, var, 240) > 0.99


def test_noise_floor_and_gating(tmp_path):
    x = np.concatenate([np.zeros(SR, np.float32), tone(440, 1.0, 0.5)])
    p = tmp_path / "a.wav"
    sf.write(str(p), x, SR)
    m = M.audio_metrics(str(p), str(p), same_as_raw=True)
    assert m["gated_frame_pct"] == pytest.approx(50.0, abs=2)
    assert m["noise_floor_db"] < -60
    assert m["lag_samples"] == 0 and m["length_equal"] is True


def make_stage_dir(root, name, diar, asr, audio=None, denoise=None, sep=None, raw=None):
    d = root / name
    (d / "01_music").mkdir(parents=True)
    sf.write(str(d / "01_music" / "after_music.wav"),
             raw if raw is not None else tone(300, 4.0), SR)
    if audio is not None:
        (d / "01b_denoise").mkdir()
        sf.write(str(d / "01b_denoise" / "after_denoise.wav"), audio, SR)
    if diar is not None:
        (d / "02_diarization").mkdir()
        (d / "02_diarization" / "segments.json").write_text(json.dumps(diar))
    if sep is not None:
        (d / "03_separation").mkdir()
        (d / "03_separation" / "segments.json").write_text(json.dumps(sep["segments"]))
        (d / "03_separation" / "stats.json").write_text(json.dumps(sep["stats"]))
    if asr is not None:
        (d / "05_asr").mkdir()
        (d / "05_asr" / "transcripts.json").write_text(json.dumps(asr, ensure_ascii=False))
    return str(d)


def ds(s, e, spk):
    return {"index": "0", "start": s, "end": e, "speaker": spk}


def ts(s, e, spk, text, words=None):
    return {"index": "0", "start": s, "end": e, "speaker": spk, "text": text, "words": words}


@pytest.fixture
def manifest(tmp_path):
    raw = tone(300, 4.0)
    sep = {"segments": [{"index": "0", "start": 0, "end": 2, "speaker": "S0", "bss": True,
                         "bss_spans": [[0, 1, 0.8], [1, 2, 0.6]], "bss_failed_spans": [[2, 3, "empty_track", ""]]}],
           "stats": {"spans_spliced": 2, "spans_failed": 1, "segments_separated": 1}}
    base = make_stage_dir(
        tmp_path, "none_f1", [ds(0, 2, "S0"), ds(2, 4, "S1")],
        [ts(0, 2, "S0", "xin chào các bạn"), ts(2, 4, "S1", "hôm nay trời đẹp")], sep=sep, raw=raw)
    var = make_stage_dir(
        tmp_path, "dfn3_f1", [ds(0, 2, "S1"), ds(2, 4, "S0")],
        [ts(0, 2, "S1", "xin chào bạn"), ts(2, 4, "S0", "hôm nay trời đẹp")],
        audio=raw * 0.5, raw=raw,
        sep={"segments": [], "stats": {"spans_spliced": 0, "spans_failed": 0, "segments_separated": 0}})
    bad = str(tmp_path / "nope")
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps([{"start": 0, "end": 2, "text": "x", "text_edited": "xin chào các bạn"},
                               {"start": 2, "end": 4, "text": "hôm nay trời đẹp"}], ensure_ascii=False))
    m = {"version": 1, "baseline": "none", "created": "t",
         "variants": [
             {"id": "none", "chain": [], "label": "none", "status": "ok", "files": {"f1": base}},
             {"id": "dfn3", "chain": ["dfn3"], "label": "DFN3", "status": "ok", "files": {"f1": var}},
             {"id": "bad", "chain": ["bad"], "label": "bad", "status": "failed", "files": {"f1": bad}}],
         "reference": {"f1": str(ref)}}
    p = tmp_path / "variants_manifest.json"
    p.write_text(json.dumps(m))
    return str(p)


def test_missing_stage_tolerance(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    o = load_file_output(str(d))
    assert o["diarization"] is None and o["asr"] is None and o["separation"] is None
    assert o["audio"]["raw"] is None
    assert M.audio_metrics(None, None)["noise_floor_db"] is None
    assert M.separation_metrics(None)["failed_spans"] is None
    assert M.asr_metrics(None, None)["n_words"] is None
    assert M.diarization_metrics(None)["n_segments"] is None


def test_build_schema_and_values(manifest):
    data = build_compare_data(manifest, embed=False)
    assert data["version"] == 1 and data["baseline"] == "none"
    assert [v["id"] for v in data["variants"]] == ["none", "dfn3", "bad"]
    assert "note" in data["variants"][2]
    f = data["files"][0]
    assert set(f) >= {"name", "duration", "audio", "text", "text_diff", "speakers", "metrics"}
    assert f["audio"]["raw"].endswith("after_music.wav") and f["audio"]["none"] == f["audio"]["raw"]
    assert f["audio"]["dfn3"].endswith("after_denoise.wav")
    assert "bad" not in f["text"] and "bad" not in f["metrics"]
    td = f["text_diff"]["dfn3"]
    assert td["ops"] == {"sub": 0, "ins": 0, "del": 1, "equal": 7}
    assert td["wer_vs_baseline"] == pytest.approx(1 / 8)
    assert td["wer_vs_reference"] == pytest.approx(1 / 8)
    assert f["text_diff"]["none"]["wer_vs_baseline"] == 0.0
    sp = f["speakers"]["dfn3"]
    assert sp["label_map"] == {"S0": "S1", "S1": "S0"} and sp["der_vs_baseline"] == 0
    m = f["metrics"]["dfn3"]
    assert set(m) == {"audio", "diarization", "separation", "asr"}
    assert set(m["audio"]) == {"noise_floor_db", "gated_frame_pct", "lag_samples", "corr_with_raw",
                               "hf_energy_ratio", "length_equal", "ecapa_cos", "learned_mos",
                               "learned_mos_delta"}
    assert set(m["asr"]) == {"n_words", "n_segments", "empty_segments", "words_per_sec",
                             "wer_vs_baseline", "wer_vs_reference", "words_in_quiet_frames"}
    assert m["audio"]["lag_samples"] == 0 and m["audio"]["corr_with_raw"] == pytest.approx(1.0)
    assert m["audio"]["length_equal"] is True and m["audio"]["ecapa_cos"] is None
    assert f["metrics"]["none"]["separation"]["overlap_spans"] == 3
    assert f["metrics"]["none"]["separation"]["mean_similarity"] == pytest.approx(0.7)
    assert m["asr"]["words_in_quiet_frames"] is None
    json.dumps(data)


def test_summary_means(tmp_path, manifest):
    # second file: same tree, different transcript for the variant
    m = json.loads(open(manifest).read())
    raw = tone(300, 4.0)
    b2 = make_stage_dir(tmp_path, "none_f2", [ds(0, 4, "S0")], [ts(0, 4, "S0", "a b c d")], raw=raw)
    v2 = make_stage_dir(tmp_path, "dfn3_f2", [ds(0, 4, "S0")], [ts(0, 4, "S0", "a b")],
                        audio=raw, raw=raw)
    m["variants"][0]["files"]["f2"] = b2
    m["variants"][1]["files"]["f2"] = v2
    p = tmp_path / "m2.json"
    p.write_text(json.dumps(m))
    data = build_compare_data(str(p), embed=False)
    assert len(data["files"]) == 2
    w1 = data["files"][0]["metrics"]["dfn3"]["asr"]["wer_vs_baseline"]
    w2 = data["files"][1]["metrics"]["dfn3"]["asr"]["wer_vs_baseline"]
    assert w2 == pytest.approx(0.5)
    assert data["summary"]["dfn3"]["asr.wer_vs_baseline"] == pytest.approx((w1 + w2) / 2)
    only = build_compare_data(str(p), files=["f2"], embed=False)
    assert [f["name"] for f in only["files"]] == ["f2"]


def test_words_in_quiet_frames(tmp_path):
    x = np.concatenate([tone(300, 1.0), np.zeros(SR, np.float32)])
    p = tmp_path / "a.wav"
    sf.write(str(p), x, SR)
    segs = [{"start": 0, "end": 2, "speaker": "A", "text": "a b",
             "words": [{"word": "a", "start": 0.1, "end": 0.5}, {"word": "b", "start": 1.2, "end": 1.8}]}]
    assert M.asr_metrics(segs, str(p))["words_in_quiet_frames"] == 1


def test_group_note_and_skipped_reason_pass_through(manifest):
    m = json.loads(open(manifest).read())
    m["variants"][1].update({"group": "masking", "note": "listen first"})
    m["variants"][2].update({"status": "skipped", "reason": "dbs_tiger: DBS_TIGER_DIR is not set", "group": "analysis_only"})
    json.dump(m, open(manifest, "w"))
    v = {x["id"]: x for x in build_compare_data(manifest, embed=False)["variants"]}
    assert v["dfn3"]["group"] == "masking" and v["dfn3"]["note"] == "listen first"
    assert "group" not in v["none"] and "note" not in v["none"]
    assert v["bad"]["status"] == "skipped" and "DBS_TIGER_DIR" in v["bad"]["note"] and v["bad"]["group"] == "analysis_only"


# ---- degraded / suspect denoise passthrough --------------------------------------
def _edit_manifest(manifest, fn):
    m = json.loads(open(manifest).read())
    fn(m)
    json.dump(m, open(manifest, "w"))


def test_old_manifest_without_denoise_keys_still_builds(manifest):
    data = build_compare_data(manifest, embed=False)
    v = {x["id"]: x for x in data["variants"]}
    assert v["dfn3"]["degraded"] is False and v["dfn3"]["degraded_files"] == 0 and v["dfn3"]["total_files"] == 0
    assert "degraded_reason" not in v["dfn3"] and "degraded" not in v["bad"]
    assert data["files"][0]["denoise"] == {}                       # no denoise.json, no report: nothing invented
    assert data["summary_n"]["dfn3"] == 1 and data["summary_excluded_degraded"]["dfn3"] == 0


def test_manifest_denoise_report_reaches_variants_files_and_summary(manifest):
    rep = {"f1": {"verdict": "fallback", "errors": ["dfn3: no weights\nmore"], "corr_with_input": 1.0, "lag_samples": 0}}

    def edit(m):
        m["variants"][1].update({"status": "degraded", "denoise_report": rep, "degraded_files": 1, "total_files": 1})
    _edit_manifest(manifest, edit)
    data = build_compare_data(manifest, embed=False)
    v = {x["id"]: x for x in data["variants"]}["dfn3"]
    assert v["status"] == "degraded" and v["degraded"] is True
    assert v["degraded_files"] == 1 and v["total_files"] == 1
    assert v["degraded_reason"] == "fell back to unfiltered audio in 1/1 files: dfn3: no weights"
    d = data["files"][0]["denoise"]["dfn3"]
    assert d["verdict"] == "fallback" and d["errors"] == ["dfn3: no weights\nmore"]
    assert d["steps"] == [] and d["blend"] is None and d["corr_with_input"] == 1.0
    assert data["summary_excluded_degraded"] == {"none": 0, "dfn3": 1, "bad": 0}
    assert data["summary_n"]["dfn3"] == 1                         # the mean is not changed, only annotated
    assert "audio.noise_floor_db" in data["summary"]["dfn3"]


def test_denoise_entry_derived_from_the_stage_dir(manifest):
    m = json.loads(open(manifest).read())
    d = m["variants"][1]["files"]["f1"]
    (open(os.path.join(d, "01b_denoise", "denoise.json"), "w")).write(json.dumps(
        {"errors": [], "corr_with_input": 1.0, "lag_samples": 0, "steps": [{"token": "dfn3"}], "blend": None}))
    data = build_compare_data(manifest, embed=False)
    e = data["files"][0]["denoise"]["dfn3"]
    assert e["verdict"] == "suspect_noop" and e["steps"] == [{"token": "dfn3"}]
    assert {x["id"]: x for x in data["variants"]}["dfn3"]["suspect_noop_files"] == 1
    assert data["summary_excluded_degraded"]["dfn3"] == 0
    open(os.path.join(d, "01b_denoise", "denoise.json"), "w").write(json.dumps({"errors": ["x: y"]}))
    data = build_compare_data(manifest, embed=False)
    assert data["files"][0]["denoise"]["dfn3"]["verdict"] == "fallback"
    assert {x["id"]: x for x in data["variants"]}["dfn3"]["degraded"] is True
