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


def prompt(**kw):
    s = {"id": f"{REC}-prompt", "recording_id": REC, "start": 0.0, "duration": 0.0, "channel": [0, 1],
         "text": "Bạn là trợ lý.", "language": "vi", "custom": {"role": "system", "event_type": "prompt"}}
    s.update(kw)
    return s


def fcall(**kw):
    custom = {"role": "assistant", "event_type": "function_call", "anchor_id": f"{REC}-turn_0_0",
              "anchor_word_index": 2}
    custom.update({k: kw.pop(k) for k in list(kw) if k in ("anchor_id", "anchor_word_index")})
    s = {"id": f"{REC}-fc", "recording_id": REC, "start": 1.0, "duration": 0.0, "channel": [0, 1],
         "text": '{"name": "f", "arguments": {}}', "language": "vi", "custom": custom}
    s.update(kw)
    return s


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
    "EXT12_prompt_start": (12, lambda c: c["supervisions"].append(prompt(start=1.0))),
    "EXT13_no_anchor": (13, lambda c: c["supervisions"].append(fcall(anchor_id="nope"))),
    "EXT13_word_index": (13, lambda c: c["supervisions"].append(fcall(anchor_word_index=99))),
    "EXT13_text": (13, lambda c: c["supervisions"].append(fcall(text="not json"))),
    "EXT13_keys": (13, lambda c: c["supervisions"].append(fcall(text='{"name": "f"}'))),
    "EXT14_language": (14, lambda c: sup(c, 0).pop("language")),
    "EXT15_gap": (15, lambda c: (sup(c, 1).update(id=sup(c, 0)["id"] + "-bc1"),
                                 sup(c, 0)["custom"].update(backchannel_ids=[sup(c, 0)["id"] + "-bc1"]))),
    "EXT16_null": (16, lambda c: sup(c, 0).update(gender=None)),
    "EXT17_backchannel_ids": (17, lambda c: sup(c, 2)["custom"].pop("backchannel_ids")),
    "EXT17_transition": (17, lambda c: sup(c, 2)["custom"].update(transition={"type": "cut"})),
    "INV11_speaker_backchannel": (11, lambda c: sup(c, 1).update(speaker="Backchannel")),
}


@pytest.mark.parametrize("name", sorted(BREAKERS))
def test_each_invariant_is_caught(name):
    invariant, mutate = BREAKERS[name]
    cut = make_cut()
    mutate(cut)
    code = f"INV{invariant:02d}" if invariant <= 11 else f"EXT{invariant:02d}"
    assert code in codes(cut), name


def test_valid_prompt_and_function_events_pass():
    cut = make_cut()
    cut["supervisions"] += [prompt(), fcall()]
    assert validate_cut(cut) == []


def test_events_may_omit_text_but_turns_may_not():
    cut = make_cut()
    ev = prompt()
    del ev["text"]
    cut["supervisions"].append(ev)
    assert "INV10" not in codes(cut)
    del sup(cut, 0)["text"]
    assert "INV10" in codes(cut)


def test_spec_section_10_example_is_valid():
    """The full example line of the PDF (section 10), with the '...' placeholders filled in."""
    rec = "vi_casual_demo_batch_0-DiBienCuoiTuan-DiBienCuoiTuan-KeHoachCuoiTuan-Qk3nAAAAAAAAAAAAAAAAAA"
    base = "DiBienCuoiTuan-KeHoachCuoiTuan"
    cut = {"id": rec + "-0", "start": 0, "duration": 12.84, "channel": [0, 1], "type": "MultiCut",
           "recording": {"id": rec, "sources": [{"type": "file", "channels": [0, 1],
                                                 "source": "/data/DiBienCuoiTuan-KeHoachCuoiTuan.wav"}],
                         "sampling_rate": 16000, "num_samples": 205440, "duration": 12.84, "channel_ids": [0, 1]},
           "supervisions": [
               {"id": f"{base}-prompt", "recording_id": rec, "start": 0.0, "duration": 0.0, "channel": [0, 1],
                "text": "Bạn là Vivi, trợ lý bán hàng. Luôn trả lời ngắn gọn, thân thiện.", "language": "vi",
                "custom": {"role": "system", "event_type": "prompt"}},
               {"id": f"{base}-turn_0_0", "recording_id": rec, "start": 0.0, "duration": 4.21, "channel": 0,
                "text": "cuối tuần này mình đi biển không ?", "language": "vi", "speaker": "F_22_N_BacNinh0007",
                "gender": "female",
                "alignment": {"word": [["cuối", 0.05, 0.22, None], ["tuần", 0.27, 0.25, None]]},
                "custom": {"role": "user", "event_type": "turn", "raw_text": "Cuối tuần này mình đi biển không?",
                           "speaker_name": "F_22_N_BacNinh0007", "age": 22, "dislect": None,
                           "assistance_voice": False, "turn_channel_position": 0, "emotion": "excited",
                           "transition": {"type": "gap", "duration": 0.0}, "backchannel_ids": []}},
               {"id": f"{base}-turn_1_0", "recording_id": rec, "start": 3.89, "duration": 5.02, "channel": 1,
                "text": "ừ hay đó , mình rủ thêm mấy bạn nữa nhé .", "language": "vi", "speaker": "Vivi",
                "gender": "female", "alignment": {"word": [["ừ", 3.95, 0.18, None]]},
                "custom": {"role": "assistant", "event_type": "turn", "raw_text": "Ừ hay đó, mình rủ thêm mấy bạn nữa nhé.",
                           "speaker_name": "Vivi", "age": 25, "dislect": None, "assistance_voice": True,
                           "turn_channel_position": 0, "emotion": "happy",
                           "transition": {"type": "overlap", "duration": 0.32},
                           "backchannel_ids": [f"{base}-turn_1_0-bc0"]}},
               {"id": f"{base}-turn_1_0-bc0", "recording_id": rec, "start": 6.10, "duration": 0.34, "channel": 0,
                "text": "ừ", "language": "vi", "speaker": "F_22_N_BacNinh0007", "gender": "female",
                "alignment": {"word": [["ừ", 6.10, 0.34, None]]},
                "custom": {"role": "user", "event_type": "backchannel", "anchor_id": f"{base}-turn_1_0",
                           "anchor_word_index": 5, "speaker_name": "F_22_N_BacNinh0007", "age": 22,
                           "dislect": None, "assistance_voice": False}}]}
    assert validate_cut(cut) == []


def test_cli_exit_code_and_summary(tmp_path, capsys):
    from utils.lhotse_validate import main
    good, bad = tmp_path / "g.jsonl", tmp_path / "b.jsonl"
    good.write_text(json.dumps(make_cut()) + "\n", encoding="utf-8")
    broken = make_cut()
    broken["supervisions"][2]["text"] = "ừ <x>"
    bad.write_text(json.dumps(broken) + "\n", encoding="utf-8")
    assert main([str(good), "--no-lhotse"]) == 0
    assert main([str(bad), "--no-lhotse"]) == 1
    out = capsys.readouterr().out
    assert main([str(tmp_path), "--no-lhotse"]) == 1          # a directory: both files, together
    assert "2 files, 2 cuts" in capsys.readouterr().out
    assert "INV10: 1" in out and "1 files, 1 cuts checked, 1 violations in 1 cuts" in out


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
             "text": '{"name": "f", "arguments": {}}', "language": "vi",
             "custom": {"role": "assistant", "event_type": "function_call",
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
