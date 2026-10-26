import json
import os

from tools import diff_variants as dv


def write(root, vid, stem, diar, asr):
    d = os.path.join(root, vid, stem)
    os.makedirs(os.path.join(d, "02_diarization"))
    os.makedirs(os.path.join(d, "05_asr"))
    json.dump(diar, open(os.path.join(d, "02_diarization", "segments.json"), "w"))
    json.dump(asr, open(os.path.join(d, "05_asr", "transcripts.json"), "w"))


def seg(s, e, spk, text=""):
    return {"start": s, "end": e, "speaker": spk, "text": text}


BASE = [seg(0, 10, "A", "xin chào các bạn hôm nay chúng ta nói về lọc nhiễu"), seg(12, 20, "B", "vâng cảm ơn anh đã mời tôi"),
        seg(22, 30, "A", "chúng ta bắt đầu từ năm ngoái nhé các bạn")]


def test_report_lists_diarization_and_text_differences(tmp_path):
    root = str(tmp_path)
    write(root, "none", "a", [{k: v for k, v in s.items() if k != "text"} for s in BASE], BASE)
    v1 = [seg(0, 10, "X", "xin chào các bạn hôm nay chúng ta nói về lọc nhiễu"), seg(12, 20, "Y", "vâng cảm ơn anh đã mời tôi"),
          seg(22, 30, "X", "chúng ta bắt đầu từ năm ngoái nhé các bạn")]          # same, labels renamed
    v2 = [seg(0, 10, "X", "xin chào các bạn hôm nay chúng ta nói về lọc tiếng"), seg(12, 20, "X", "vâng cảm ơn"),
          seg(22, 30, "X", "chúng ta bắt đầu từ năm ngoái nhé các bạn")]          # B merged into A, words lost
    for vid, segs in (("v1", v1), ("v2", v2)):
        write(root, vid, "a", [{k: v for k, v in s.items() if k != "text"} for s in segs], segs)
    text = dv.report(root, "a", ["v1", "v2", "v3"], "none", min_sec=1.0, min_wer=0.05)
    assert "CHƯA CÓ kết quả (bỏ qua): v3" in text
    assert "3. v1 so với none" in text and "0 đoạn khác nhau" in text            # renamed labels are not a difference
    assert "[nhiễu→tiếng]" in text and "[-anh]" in text                          # changed words are marked
    assert "swap" in text or "missing" in text                                   # the merged speaker shows up
    assert "2. Khoảng cách giữa các variant" in text


def test_main_writes_the_report_next_to_the_results(tmp_path, capsys):
    root = str(tmp_path)
    write(root, "none", "a", [{"start": 0, "end": 5, "speaker": "A"}], [seg(0, 5, "A", "một hai ba")])
    write(root, "l_p3", "a", [{"start": 0, "end": 5, "speaker": "A"}], [seg(0, 5, "A", "một hai bốn")])
    assert dv.main(["--out-root", root, "--file", "a", "--variants", "l_p3"]) == 0
    assert "[ba→bốn]" in open(os.path.join(root, "diff_a.txt"), encoding="utf-8").read()
