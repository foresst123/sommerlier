"""Lhotse manifest validator: one passing and one violating case per invariant.

Run:  python -m pytest tests/test_lhotse_validate.py -q     (from podcast-pipeline/)
"""
import copy
import gzip
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lhotse_validate import (  # noqa: E402
    validate_cut, validate_cuts, validate_file, validate_with_lhotse)

REC = "b1-t1-conv-AAAAAAAAAAAAAAAAAAAAAA"


def make_cut():
    """Spec section 10: a user turn on channel 0 with a backchannel from the assistant on channel 1."""
    turn = {"id": f"{REC}-turn_0_0", "recording_id": REC, "start": 0.0, "duration": 8.0, "channel": 0,
            "text": "cuối tuần này mình đi biển không", "language": "vi",
            "speaker": "F_22_N_BacNinh0007", "gender": "female",
            "alignment": {"word": [["cuối", 0.05, 0.22, None], ["tuần", 0.27, 0.25, None],
                                   ["này", 0.55, 0.2, None], ["mình", 0.8, 0.2, None],
                                   ["đi", 1.0, 0.2, None], ["biển", 1.3, 0.3, None],
                                   ["không", 1.7, 0.3, None]]},
            "custom": {"role": "user", "event_type": "turn", "raw_text": "Cuối tuần này mình đi biển không?",
                       "speaker_name": "x", "age": 22, "dislect": None, "assistance_voice": False,
                       "turn_channel_position": 0, "emotion": "excited",
                       "transition": {"type": "gap", "duration": 0.0},
                       "backchannel_ids": [f"{REC}-turn_0_0-bc0"]}}
    bc = {"id": f"{REC}-turn_0_0-bc0", "recording_id": REC, "start": 3.0, "duration": 0.34, "channel": 1,
          "text": "ừ", "language": "vi", "speaker": "M_30_N_Hanoi0001",
          "alignment": {"word": [["ừ", 3.0, 0.34, None]]},
          "custom": {"role": "assistant", "event_type": "backchannel", "anchor_id": turn["id"],
                     "anchor_word_index": 5, "speaker_name": "y", "age": None, "dislect": None,
                     "assistance_voice": True}}
    reply = {"id": f"{REC}-turn_1_0", "recording_id": REC, "start": 8.5, "duration": 3.0, "channel": 1,
             "text": "ừ đi chứ", "language": "vi", "speaker": "M_30_N_Hanoi0001",
             "alignment": {"word": [["ừ", 8.6, 0.2, None], ["đi", 8.9, 0.2, None], ["chứ", 9.2, 0.3, 0.9]]},
             "custom": {"role": "assistant", "event_type": "turn", "assistance_voice": True,
                        "turn_channel_position": 0, "emotion": None,
                        "transition": {"type": "gap", "duration": 0.5}, "backchannel_ids": []}}
    return {"id": f"{REC}-0", "start": 0, "duration": 12.0, "channel": [0, 1], "type": "MultiCut",
            "recording": {"id": REC, "sources": [{"type": "file", "channels": [0, 1], "source": "/x.wav"}],
                          "sampling_rate": 16000, "num_samples": 192000, "duration": 12.0,
                          "channel_ids": [0, 1]},
            "supervisions": [turn, bc, reply]}


def codes(cut, **kw):
    return {v.code for v in validate_cut(cut, **kw)}


def sup(cut, n):
    return cut["supervisions"][n]


def test_valid_cut_has_no_violation():
    assert validate_cut(make_cut()) == []
    assert validate_cuts([make_cut()]) == []


# name -> (invariant, mutation that breaks it)
BREAKERS = {
    "INV01_type": (1, lambda c: c.update(type="DataCut")),
    "INV01_channel": (1, lambda c: c.update(channel=[0])),
    "INV01_start": (1, lambda c: c.update(start=1.0)),
    "INV01_duration": (1, lambda c: c.update(duration=11.0)),
    "INV01_channel_ids": (1, lambda c: c["recording"].update(channel_ids=[1, 0])),
    "INV01_cut_id": (1, lambda c: c.update(id="other-0")),
    "INV01_recording_id": (1, lambda c: sup(c, 0).update(recording_id="zzz")),
    "INV03_past_end": (3, lambda c: sup(c, 2).update(start=11.0)),
    "INV03_negative": (3, lambda c: sup(c, 0).update(start=-0.5)),
    "INV04_before_start": (4, lambda c: sup(c, 0)["alignment"]["word"][0].__setitem__(1, -0.5)),
    "INV04_after_end": (4, lambda c: sup(c, 0)["alignment"]["word"][-1].__setitem__(1, 7.9)),
    "INV04_order": (4, lambda c: sup(c, 0)["alignment"]["word"][2].__setitem__(1, 0.3)),
    "INV05_end_time_as_duration": (5, lambda c: sup(c, 0)["alignment"]["word"][0].__setitem__(2, 20.0)),
    "INV05_negative": (5, lambda c: sup(c, 0)["alignment"]["word"][0].__setitem__(2, -0.1)),
    "INV05_shape": (5, lambda c: sup(c, 0)["alignment"]["word"].append(["x", 5.0, 0.1])),
    "INV05_score": (5, lambda c: sup(c, 0)["alignment"]["word"][0].__setitem__(3, "n/a")),
    "INV06": (6, lambda c: sup(c, 2).update(channel=0, start=7.0)),
    "INV07_same_channel": (7, lambda c: sup(c, 1).update(channel=0)),
    "INV07_outside": (7, lambda c: sup(c, 1).update(start=7.9)),
    "INV07_index_zero": (7, lambda c: sup(c, 1)["custom"].update(anchor_word_index=0)),
    "INV07_index_big": (7, lambda c: sup(c, 1)["custom"].update(anchor_word_index=8)),
    "INV07_id_format": (7, lambda c: (sup(c, 1).update(id="bc-zero"),
                                      sup(c, 0)["custom"].update(backchannel_ids=["bc-zero"]))),
    "INV07_missing_anchor": (7, lambda c: sup(c, 1)["custom"].update(anchor_id="nope")),
    "INV08_not_listed": (8, lambda c: sup(c, 0)["custom"].update(backchannel_ids=[])),
    "INV08_dangling": (8, lambda c: sup(c, 0)["custom"].update(backchannel_ids=["ghost"])),
    "INV08_wrong_back": (8, lambda c: sup(c, 2)["custom"].update(backchannel_ids=[sup(c, 1)["id"]])),
    "INV09": (9, lambda c: sup(c, 0)["custom"].update(event_type="interrupt")),
    "INV10_angle": (10, lambda c: sup(c, 0).update(text="xin <chào>")),
    "INV10_bracket": (10, lambda c: sup(c, 2).update(text="ừ [laugh] đi")),
    "INV11_no_role": (11, lambda c: sup(c, 0)["custom"].pop("role")),
    "INV11_bad_pair": (11, lambda c: sup(c, 0)["custom"].update(role="system")),
    "INV11_no_speaker": (11, lambda c: sup(c, 0).pop("speaker")),
    "INV11_assistance_voice": (11, lambda c: sup(c, 0)["custom"].update(assistance_voice=True)),
    "INV11_gender_null": (11, lambda c: sup(c, 0).update(gender="null")),
    "INV11_speaker_backchannel": (11, lambda c: sup(c, 1).update(speaker="Backchannel")),
}


@pytest.mark.parametrize("name", sorted(BREAKERS))
def test_each_invariant_is_caught(name):
    invariant, mutate = BREAKERS[name]
    cut = make_cut()
    mutate(cut)
    assert f"INV{invariant:02d}" in codes(cut), name


def test_inv02_duplicate_ids_across_the_manifest():
    a, b = make_cut(), make_cut()
    assert validate_cuts([a, b]) and {v.code for v in validate_cuts([a, b])} == {"INV02"}
    b2 = make_cut()
    for s in b2["supervisions"]:
        s["id"] = "other-" + s["id"]
        s["custom"].get("backchannel_ids") and s["custom"].update(
            backchannel_ids=["other-" + i for i in s["custom"]["backchannel_ids"]])
        if "anchor_id" in s["custom"]:
            s["custom"]["anchor_id"] = "other-" + s["custom"]["anchor_id"]
    b2["id"] = REC + "-1"
    assert validate_cuts([make_cut(), b2]) == []


def test_tolerances():
    cut = make_cut()
    sup(cut, 2)["start"] = 11.0                          # ends at 14: far outside
    assert "INV03" in codes(cut)
    cut = make_cut()
    sup(cut, 2).update(start=9.0, duration=3.0005)        # 0.5 ms over the end: inside 1 ms
    assert "INV03" not in codes(cut)
    cut = make_cut()
    sup(cut, 2).update(start=9.0, duration=3.002)         # 2 ms over: caught
    assert "INV03" in codes(cut)
    cut = make_cut()
    ali = sup(cut, 0)["alignment"]["word"]
    ali[1][1] = 0.265                                      # 5 ms overlap with the previous word: inside 10 ms
    assert "INV04" not in codes(cut)
    ali[1][1] = 0.25                                       # 20 ms overlap: caught
    assert "INV04" in codes(cut)


def test_event_rules():
    cut = make_cut()
    event = {"id": f"{REC}-ev0", "recording_id": REC, "start": 2.0, "duration": 0.0, "channel": [0, 1],
             "text": "", "language": "vi", "custom": {"role": "assistant", "event_type": "function_call",
                                                      "anchor_id": sup(cut, 0)["id"], "anchor_word_index": 1}}
    cut["supervisions"].append(event)
    assert codes(cut) == set()
    event["duration"] = 0.5
    assert "INV11" in codes(cut)
    event["duration"] = 0.0
    event["speaker"] = "someone"
    assert "INV11" in codes(cut)
    del event["speaker"]
    event["custom"]["assistance_voice"] = False
    assert "INV11" in codes(cut)


def test_same_channel_touching_turns_are_fine():
    cut = make_cut()
    sup(cut, 2).update(channel=0, start=8.0, duration=3.0)
    assert "INV06" not in codes(cut)


def test_absent_event_type_means_turn():
    cut = make_cut()
    del sup(cut, 2)["custom"]["event_type"]
    assert validate_cut(cut) == []


def test_violation_fields_and_inputs_untouched():
    cut = make_cut()
    before = copy.deepcopy(cut)
    cut["supervisions"][0]["text"] = "a<b"
    v = validate_cut(cut)[0]
    assert (v.invariant, v.code, v.cut_id, v.supervision_id) == (10, "INV10", cut["id"], sup(cut, 0)["id"])
    cut["supervisions"][0]["text"] = before["supervisions"][0]["text"]
    assert cut == before


@pytest.mark.parametrize("name", ["m.jsonl", "m.jsonl.gz"])
def test_validate_file(tmp_path, name):
    good, bad = make_cut(), make_cut()
    bad["supervisions"][2]["text"] = "ừ <x>"
    path = tmp_path / name
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(good) + "\n")
    assert validate_file(str(path)) == []
    with opener(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(bad) + "\n\n")
    assert {v.code for v in validate_file(str(path))} == {"INV10"}


def test_validate_with_lhotse_loads_a_valid_manifest(tmp_path):
    pytest.importorskip("lhotse")
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps(make_cut()) + "\n", encoding="utf-8")
    assert validate_with_lhotse(str(path)) == []


def test_validate_with_lhotse_without_lhotse_is_empty(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "lhotse", None)       # makes `import lhotse` raise ImportError
    assert validate_with_lhotse(str(tmp_path / "missing.jsonl")) == []
