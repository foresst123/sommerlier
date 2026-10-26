import json

from tools import make_variant_lanes_page as lanes
from tools import preview_manifest as pm


def test_preview_lists_only_variants_with_transcripts_and_feeds_the_lanes_page(tmp_path):
    out = tmp_path
    for vid, stem in (("none", "a"), ("l_p1", "a"), ("l_p1", "b")):
        d = out / vid / stem / "05_asr"
        d.mkdir(parents=True)
        (d / "transcripts.json").write_text(json.dumps([{"start": 0, "end": 1, "speaker": "S0", "text": "xin chào"}]))
    (out / "none" / ".done").write_text("{}")
    (out / "m_p3").mkdir()                                            # started, no transcript yet
    plan = {"baseline": "none", "variants": [
        {"id": "raw", "chain": "none"}, {"id": "none"},
        {"id": "l_p1", "chain": "rf_crowd>rf_vocals_resurrection"},
        {"id": "m_p3", "chain_pre_asr": "mpsenet_dns"}]}
    m = pm.build(out, plan)
    assert [v["id"] for v in m["variants"]] == ["none", "l_p1"]
    by = {v["id"]: v for v in m["variants"]}
    assert by["l_p1"]["chain"] == ["rf_crowd", "rf_vocals_resurrection"] and by["l_p1"]["chain_pre_asr"] == []
    assert by["none"]["status"] == "ok" and by["l_p1"]["status"] == "partial" and set(by["l_p1"]["files"]) == {"a", "b"}
    path = out / "variants_manifest_preview.json"
    path.write_text(json.dumps(m))
    data = lanes.build_page_data(str(path), "a", str(out / "lanes"))
    assert [v["id"] for v in data["variants"]] == ["none", "l_p1"]
    assert data["variants"][1]["flow"].endswith("rf_crowd → rf_vocals_resurrection → diarization → separation → asr")
