"""Tests for the conversation-export to Lhotse manifest converter (small fake items, no GPU)."""
import copy
import gzip
import json
import os
import sys
import uuid

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("utils.lhotse_duplex")

from services.lhotse_manifest_service import (LhotseManifestParams, build_manifest,
                                              convert_item, short_uuid)

SR = 16000
FIXED = lambda: uuid.UUID(int=1)                      # noqa: E731


def _words(text, start, end):
    toks = text.split()
    step = (end - start) / len(toks)
    return [{"word": t, "start": round(start + i * step, 3), "end": round(start + (i + 1) * step - 0.02, 3)}
            for i, t in enumerate(toks)]


def _row(index, role, start, end, text, state="NORMAL", words=True, **extra):
    row = {"index": index, "role": role, "speaker_id": f"S{role[0]}", "start": start, "end": end,
           "text": text, "state": state}
    if words:
        row["words"] = _words(text, start, end)
    row.update(extra)
    return row


def _base_rows():
    return [
        _row(0, "assistant", 0.2, 1.0, "Chào bạn nhé"),
        _row(1, "user", 2.0, 3.0, "Mình khỏe lắm"),
        _row(2, "assistant", 4.0, 6.0, "Hôm nay trời đẹp quá phải không"),
        _row(3, "user", 4.8, 5.1, "ừ", state="BACKCHANNEL"),
        _row(4, "user", 7.0, 8.0, "Đúng rồi"),
    ]


def _make_item(directory, rows=None, *, left="user", method="strict_separation_tracks",
               failed=None, events=None, duration=9.0, **meta_extra):
    rows = _base_rows() if rows is None else rows
    os.makedirs(directory, exist_ok=True)
    n = int(duration * SR)
    data = np.zeros((n, 2), dtype=np.float32)
    t = np.arange(n) / SR
    tone = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    for r in rows:
        ch = 0 if r["role"] == "user" else 1
        if left == "assistant":
            ch = 1 - ch
        a, b = int(r["start"] * SR), int(r["end"] * SR)
        data[a:b, ch] = tone[a:b]
    sf.write(os.path.join(directory, "stereo_2ch.wav"), data, SR, subtype="PCM_16")
    meta = {"id": "x", "channels": {"left": left, "right": "user" if left == "assistant" else "assistant",
                                    "method": method},
            "events": events if events is not None else [], "conversation": rows,
            "topic": "chủ đề", "summary": "tóm tắt", "duration": duration, "sample_rate": SR,
            "num_samples": n, "source_stem": "podX", "files": {"stereo_2ch": "stereo_2ch.wav"}}
    if failed is not None:
        meta["failed_separation_spans"] = failed
    meta.update(meta_extra)
    with open(os.path.join(directory, "conversation.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    return str(directory)


def _convert(path, **kw):
    return convert_item(path, "batch1", "topic1", LhotseManifestParams(**kw), uuid_factory=FIXED)


def _sups(cut, event_type):
    return [s for s in cut["supervisions"] if s["custom"]["event_type"] == event_type]


def test_short_uuid_is_22_chars_urlsafe():
    s = short_uuid()
    assert len(s) == 22 and "=" not in s
    assert short_uuid(uuid.UUID(int=1)) == short_uuid(uuid.UUID(int=1))


def test_recording_and_turn_stretch(tmp_path):
    res = _convert(_make_item(tmp_path / "item"))
    assert res.cut is not None, res.reasons
    cut = res.cut
    rec = cut["recording"]
    uid = short_uuid(uuid.UUID(int=1))
    assert rec["id"] == f"batch1-topic1-stereo_2ch-{uid}" and len(uid) == 22
    assert cut["id"] == rec["id"] + "-0" and cut["type"] == "MultiCut"
    assert cut["channel"] == [0, 1] and cut["start"] == 0.0
    assert rec["channel_ids"] == [0, 1] and rec["sampling_rate"] == SR
    assert rec["num_samples"] == 9 * SR and rec["duration"] == cut["duration"] == 9.0
    assert os.path.isabs(rec["sources"][0]["source"])
    turns = _sups(cut, "turn")
    assert [t["channel"] for t in turns] == [1, 0, 1, 0]
    t0, t1, t2, t3 = turns
    assert t0["start"] == 0.2 and t0["duration"] == pytest.approx(2.0 - 0.2)   # to next start
    assert t2["start"] + t2["duration"] == pytest.approx(7.0)
    assert t3["start"] + t3["duration"] == pytest.approx(9.0)                  # to end of recording
    assert t1["custom"]["role"] == "user" and t0["custom"]["role"] == "assistant"
    assert t0["custom"]["assistance_voice"] is True and t1["custom"]["assistance_voice"] is False
    assert [t["custom"]["turn_channel_position"] for t in turns] == [0, 0, 1, 1]
    assert t0["speaker"] == "podX_Sa" and "gender" not in t0
    assert t0["custom"]["transition"] == {"type": "gap", "duration": 0.0}
    assert t0["custom"]["conversation_description"] == "tóm tắt"
    assert t0["custom"]["age"] is None and t0["custom"]["emotion"] is None
    assert len({s["id"] for s in cut["supervisions"]}) == len(cut["supervisions"])


def test_alignment_uses_duration_inside_turn(tmp_path):
    cut = _convert(_make_item(tmp_path / "item")).cut
    for sup in cut["supervisions"]:
        lo, hi = sup["start"], sup["start"] + sup["duration"]
        assert lo >= 0 and hi <= cut["duration"] + 1e-6
        prev_end = lo
        for word, start, dur, score in sup["alignment"]["word"]:
            assert start >= prev_end - 1e-9 and dur >= 0 and score is None
            assert start + dur <= hi + 1e-6
            prev_end = start + dur
    first = _sups(cut, "turn")[0]["alignment"]["word"][0]
    assert first[1] == 0.2 and first[2] < 1.0          # third element is a duration, not an end


def test_swapped_channels_write_new_wav(tmp_path):
    item = _make_item(tmp_path / "item", left="assistant")
    cut = _convert(item).cut
    assert cut is not None
    src = cut["recording"]["sources"][0]["source"]
    assert os.path.basename(src) == "stereo_duplex.wav"
    data, _ = sf.read(src)
    first = _sups(cut, "turn")[0]                       # assistant -> channel 1
    assert first["channel"] == 1 and first["custom"]["role"] == "assistant"
    a = int(0.3 * SR)
    assert abs(data[a, 1]) > 0 or abs(data[a + 50, 1]) > 0
    assert np.abs(data[a:a + 4000, 0]).max() == 0
    # the original file is left alone
    orig, _ = sf.read(os.path.join(item, "stereo_2ch.wav"))
    assert np.abs(orig[a:a + 4000, 0]).max() > 0


def test_straight_channels_use_original_wav(tmp_path):
    item = _make_item(tmp_path / "item", left="user")
    cut = _convert(item).cut
    assert cut["recording"]["sources"][0]["source"] == os.path.join(os.path.abspath(item), "stereo_2ch.wav")


def test_adjacent_same_speaker_segments_merge(tmp_path):
    rows = [_row(0, "assistant", 0.5, 1.5, "một hai"), _row(1, "assistant", 1.6, 2.5, "ba bốn"),
            _row(2, "user", 3.0, 4.0, "năm sáu")]
    cut = _convert(_make_item(tmp_path / "item", rows)).cut
    turns = _sups(cut, "turn")
    assert len(turns) == 2
    assert turns[0]["text"] == "một hai ba bốn"
    assert [w[0] for w in turns[0]["alignment"]["word"]] == ["một", "hai", "ba", "bốn"]


def test_long_silence_splits_a_speakers_turn(tmp_path):
    rows = [_row(0, "assistant", 0.5, 1.5, "một hai"), _row(1, "assistant", 5.0, 6.0, "ba bốn"),
            _row(2, "user", 7.0, 8.0, "năm sáu")]
    cut = _convert(_make_item(tmp_path / "item", rows)).cut
    assert [t["text"] for t in _sups(cut, "turn")] == ["một hai", "ba bốn", "năm sáu"]
    merged = _convert(_make_item(tmp_path / "item2", rows), max_turn_silence=10.0).cut
    assert [t["text"] for t in _sups(merged, "turn")] == ["một hai ba bốn", "năm sáu"]


def test_text_stripped_but_raw_text_kept(tmp_path):
    rows = [_row(0, "assistant", 0.5, 2.0, "xin > chào [ bạn <"), _row(1, "user", 3.0, 4.0, "ừm vâng")]
    cut = _convert(_make_item(tmp_path / "item", rows)).cut
    t = _sups(cut, "turn")[0]
    assert t["text"] == "xin chào bạn"
    assert t["custom"]["raw_text"] == "xin > chào [ bạn <"
    assert [w[0] for w in t["alignment"]["word"]] == ["xin", "chào", "bạn"]


def test_empty_text_after_strip_drops_supervision(tmp_path):
    rows = [_row(0, "assistant", 0.5, 1.5, "xin chào"), _row(1, "user", 2.0, 2.5, "[ >"),
            _row(2, "assistant", 3.0, 4.0, "tạm biệt")]
    cut = _convert(_make_item(tmp_path / "item", rows, duration=5.0), coverage_check=False).cut
    assert 0 not in [s["channel"] for s in cut["supervisions"]]     # no empty user supervision


@pytest.mark.parametrize("kind", ["method", "failed", "tag", "angle_tag", "untimed"])
def test_item_rejections(tmp_path, kind):
    rows = _base_rows()
    kw = {}
    if kind == "method":
        kw["method"] = "time_gated"
    elif kind == "failed":
        kw["failed"] = [{"start": 1.0, "end": 2.0}]
    elif kind == "tag":
        rows[1]["text"] = "[Nhạc] mình khỏe"
    elif kind == "angle_tag":
        rows[1]["text"] = "mình <laugh> khỏe"
    elif kind == "untimed":                    # chosen: untimed text rejects the item
        rows[1].pop("words")
    res = _convert(_make_item(tmp_path / "item", rows, **kw))
    assert res.cut is None
    assert res.reasons
    key = {"method": "method:", "failed": "failed_separation_spans", "tag": "tag_in_text",
           "angle_tag": "tag_in_text", "untimed": "untimed_segment"}[kind]
    assert res.reasons[0].startswith(key)


def test_empty_failed_spans_is_fine(tmp_path):
    assert _convert(_make_item(tmp_path / "item", failed=[])).cut is not None


def test_backchannel_attached_to_anchor(tmp_path):
    cut = _convert(_make_item(tmp_path / "item")).cut
    bcs = _sups(cut, "backchannel")
    assert len(bcs) == 1, cut["supervisions"]
    bc = bcs[0]
    anchor = next(s for s in cut["supervisions"] if s["id"] == bc["custom"]["anchor_id"])
    assert anchor["channel"] == 1 and bc["channel"] == 0
    assert bc["id"] == anchor["id"] + "-bc0"
    assert anchor["start"] <= bc["start"] and bc["start"] + bc["duration"] <= anchor["start"] + anchor["duration"] + 1e-6
    assert bc["start"] == pytest.approx(4.8) and bc["duration"] == pytest.approx(0.3 - 0.02, abs=0.02)
    assert bc["speaker"] == "podX_Su" and bc["text"] == "ừ"
    assert bc["custom"]["role"] == "user" and bc["custom"]["event_type"] == "backchannel"
    assert bc["custom"]["anchor_word_index"] >= 1
    assert anchor["custom"]["backchannel_ids"] == [bc["id"]]
    for t in _sups(cut, "turn"):
        if t is not anchor:
            assert t["custom"]["backchannel_ids"] == []
    # the backchannel is not also a turn
    assert all(t["text"] != "ừ" for t in _sups(cut, "turn"))


def test_backchannel_without_anchor_is_demoted_to_turn(tmp_path):
    # nothing on the other channel covers it (before the first turn), so it becomes a turn
    rows = [_row(0, "user", 0.1, 0.4, "ừ", state="BACKCHANNEL"),
            _row(1, "assistant", 1.0, 2.0, "xin chào"), _row(2, "user", 4.0, 5.0, "hết rồi")]
    res = _convert(_make_item(tmp_path / "item", rows))
    assert res.cut is not None, res.reasons
    assert not _sups(res.cut, "backchannel")
    assert any(t["text"] == "ừ" for t in _sups(res.cut, "turn"))
    assert any(r.startswith("backchannel_demoted") for r in res.reasons)


def test_event_backchannel_candidate(tmp_path):
    rows = _base_rows()
    rows[3]["state"] = "NORMAL"
    events = [{"type": "backchannel", "by": "user", "start": 4.8, "end": 5.1, "text": "ừ",
               "segment": "3"}]
    cut = _convert(_make_item(tmp_path / "item", rows, events=events)).cut
    assert len(_sups(cut, "backchannel")) == 1


def test_untimed_backchannel_is_dropped_not_item(tmp_path):
    rows = _base_rows()
    rows[3].pop("words")
    res = _convert(_make_item(tmp_path / "item", rows), coverage_check=False)
    assert res.cut is not None
    assert not _sups(res.cut, "backchannel")
    assert any("backchannel_dropped" in r for r in res.reasons)


def test_emotion_threshold(tmp_path):
    rows = _base_rows()
    rows[0]["emotion"] = {"label": "vui", "conf": 0.9}
    rows[1]["emotion"] = {"label": "buồn", "conf": 0.3}
    cut = _convert(_make_item(tmp_path / "item", rows), emotion_min_conf=0.6).cut
    turns = _sups(cut, "turn")
    assert turns[0]["custom"]["emotion"] == "vui" and turns[1]["custom"]["emotion"] is None


def test_overlap_transition(tmp_path):
    rows = [_row(0, "assistant", 0.5, 3.0, "một hai ba bốn năm sáu"),
            _row(1, "user", 2.0, 4.0, "bảy tám chín mười")]
    cut = _convert(_make_item(tmp_path / "item", rows, duration=5.0)).cut
    t0, t1 = _sups(cut, "turn")
    assert t0["start"] + t0["duration"] == pytest.approx(3.0 - 0.02, abs=0.03)     # keeps its speech
    assert t1["custom"]["transition"]["type"] == "overlap"
    assert t1["custom"]["transition"]["duration"] > 0


def test_uncovered_speech_rejects_item(tmp_path):
    rows = _base_rows()
    item = _make_item(tmp_path / "item", rows)
    data, _ = sf.read(os.path.join(item, "stereo_2ch.wav"))
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(SR) / SR)
    # speech on a channel 1 stretch inside the 7.0-9.0 region where only user is annotated
    data[int(7.2 * SR):int(8.2 * SR), 1] = tone
    sf.write(os.path.join(item, "stereo_2ch.wav"), data, SR, subtype="PCM_16")
    res = _convert(item)
    assert res.cut is None and res.reasons[0].startswith("coverage")


def test_missing_wav_rejects(tmp_path):
    item = _make_item(tmp_path / "item")
    os.remove(os.path.join(item, "stereo_2ch.wav"))
    assert _convert(item).reasons[0] == "missing_wav"


def test_build_manifest_writes_gz_and_skips(tmp_path):
    root = tmp_path / "root"
    _make_item(root / "b1" / "t1" / "interrupt" / "a")
    _make_item(root / "b1" / "t1" / "interrupt" / "b")
    _make_item(root / "b1" / "t2" / "backchannel" / "c", method="time_gated")
    out = tmp_path / "out" / "m.jsonl.gz"
    report = build_manifest(str(root), str(out))
    assert report.written == 2 and len(report.skipped) == 1
    assert report.skipped[0]["reasons"][0].startswith("method:")
    with gzip.open(out, "rt", encoding="utf-8") as fh:
        cuts = [json.loads(line) for line in fh]
    assert [c["id"].rsplit("-", 1)[1] for c in cuts] == ["0", "1"]
    assert all(c["recording"]["id"].startswith("b1-t1-") for c in cuts)
    ids = [s["id"] for c in cuts for s in c["supervisions"]]
    assert len(ids) == len(set(ids))


def test_build_manifest_plain_jsonl_keeps_unicode(tmp_path):
    root = tmp_path / "root"
    _make_item(root / "b1" / "t1" / "interrupt" / "a")
    out = tmp_path / "m.jsonl"
    assert build_manifest(str(root), str(out)).written == 1
    text = out.read_text(encoding="utf-8")
    assert "Chào" in text and len(text.strip().splitlines()) == 1
