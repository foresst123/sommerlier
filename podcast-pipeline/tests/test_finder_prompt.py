"""The finder prompt and the finder's line format: what the model is told, and what it is shown."""

import json
import re

import test_conversation_export_service as base
from services.conversation_export_service import ConversationExportService, parse_finder_reply
from utils.conversation_selection import ConversationSelectionConfig, ConversationSelectionFinder
from utils.excise import TimelineMap
from utils.transcript_windows import line_number  # noqa: F401  (keeps the import path honest)


def _service():
    return ConversationExportService(base.FakeLLM("[]"))


def _prompt():
    return _service()._finder_system_prompt()


def _ov_and_gap(lines):
    """Per line: (ov, gap) recomputed from the timestamps alone, as the finder does.

    ov is the time a line shares with lines of the other speaker; gap is from the end of the
    line before it in the list (negative when they overlap). The first line has no gap.
    """
    spans = []
    for line in lines:
        m1, s1, m2, s2 = re.match(r"\[\d+\] (\d+):(\d\d)-(\d+):(\d\d)", line).groups()
        spans.append(((int(m1) * 60 + int(s1)), (int(m2) * 60 + int(s2))))
    speakers = [re.search(r"\) ([A-Z]) ov=", line).group(1) for line in lines]
    out = []
    for i, (a, b) in enumerate(spans):
        other = sorted((max(a, c), min(b, d)) for k, (c, d) in enumerate(spans)
                       if speakers[k] != speakers[i] and min(b, d) > max(a, c))
        ov, reach = 0.0, None
        for lo, hi in other:                     # union of the other speaker's overlaps
            lo = lo if reach is None else max(lo, reach)
            if hi > lo:
                ov += hi - lo
                reach = hi
        gap = None if i == 0 else a - spans[i - 1][1]
        out.append((ov, gap))
    return out


def _examples(prompt):
    """[(example lines, answer regions)] read back out of the prompt text."""
    blocks = re.split(r"\nVí dụ \d+ - [^\n]*\n", "\n" + prompt)[1:]
    out = []
    for block in blocks:
        lines = [l for l in block.splitlines() if re.match(r"\[\d+\] \d+:\d\d-", l)]
        answer = next(l for l in block.splitlines() if l.startswith("-> "))
        out.append((lines, parse_finder_reply(answer[3:])))
    return out


def test_a_finder_line_carries_its_own_duration():
    segs = base._talk(3, seg=7.0)
    finder = ConversationSelectionFinder(
        segs, TimelineMap(), base._quiet(), None, ConversationSelectionConfig())
    line = _service()._finder_line(finder, 0, 1, {base.A: "A", base.B: "B"})
    assert line.startswith("[1] 0:00-0:07 (d=7.0s) A ov=")


def test_the_prompt_says_to_take_durations_from_the_timestamps():
    prompt = _prompt()
    assert "(d=giây)" in prompt and "m:ss-m:ss" in prompt
    assert "không ước lượng theo số dòng" in prompt
    assert not re.search(r"\{[a-z_]+\}", prompt), "an unfilled {placeholder} is left in the prompt"


def test_the_prompt_carries_five_worked_examples():
    examples = _examples(_prompt())
    assert len(examples) == 5
    assert [len(lines) for lines, _ in examples] == [6, 7, 13, 3, 6]


def test_each_example_answer_covers_its_lines_exactly_and_is_valid_json_of_the_right_shape():
    from services.conversation_export_service import coverage_of_regions
    for lines, regions in _examples(_prompt()):
        assert regions, "the example answer did not parse"
        coverage = coverage_of_regions(regions, len(lines))
        assert coverage["coverage_fraction"] == 1.0 and not coverage["overlap_ranges"]
        assert all(set(r) >= {"start_line", "end_line", "keep", "self_contained", "topic"}
                   for r in regions)


def test_the_durations_in_the_examples_match_their_timestamps():
    pattern = re.compile(r"\[\d+\] (\d+):(\d\d)-(\d+):(\d\d) \(d=([\d.]+)s\)")
    for lines, _ in _examples(_prompt()):
        for line in lines:
            m1, s1, m2, s2, d = pattern.match(line).groups()
            span = (int(m2) * 60 + int(s2)) - (int(m1) * 60 + int(s1))
            assert abs(float(d) - span) <= 1.0, line


def test_the_examples_show_each_category_a_break_point_and_a_one_sided_region():
    (one, two, three, four, five) = [regions for _, regions in _examples(_prompt())]
    assert [r["category"] for r in one] == ["interrupt"]
    assert [r["category"] for r in two] == ["backchannel"]
    assert [(r["start_line"], r["end_line"], r["keep"]) for r in three] == [
        (1, 6, True), (7, 7, False), (8, 13, True)]
    assert [r["category"] for r in three if r["keep"]] == ["turn_by_turn", "turn_by_turn"]
    assert [r["keep"] for r in four] == [False]
    assert [r["keep"] for r in five] == [False]


def test_the_ov_and_gap_in_the_examples_match_their_timestamps():
    for lines, _ in _examples(_prompt()):
        for line, (ov, gap) in zip(lines, _ov_and_gap(lines)):
            shown_ov = float(re.search(r" ov=([\d.]+)s", line).group(1))
            assert abs(shown_ov - ov) <= 0.01, (line, ov)
            shown_gap = re.search(r" gap=(-?[\d.]+)s", line)
            if gap is None:
                assert shown_gap is None, line       # the first line of an example has no gap
            else:
                assert shown_gap and abs(float(shown_gap.group(1)) - gap) <= 0.01, (line, gap)


def test_the_examples_line_up_with_the_format_the_finder_really_prints():
    lines = [l for lines, _ in _examples(_prompt()) for l in lines]
    assert lines and all(re.match(
        r"\[\d+\] \d+:\d\d-\d+:\d\d \(d=[\d.]+s\) [A-Z] ov=[\d.]+s( gap=-?[\d.]+s)?( hard=\w+)?: ",
        l) for l in lines)
    assert any("hard=bad_separation" in l for l in lines)
    assert not any(re.search(r"seam|noise|music|flags=", l) for l in lines)


def _reply_for(topic):
    return json.dumps([
        {"start_line": 1, "end_line": 20, "keep": True, "self_contained": 5,
         "topic": topic, "reason": "x"},
        {"start_line": 21, "end_line": 30, "keep": False, "self_contained": 2,
         "topic": "phần còn lại", "reason": "x"}])


def test_a_region_copied_from_an_example_is_refused(tmp_path):
    _, result, _ = base._run(tmp_path, base.FakeLLM(_reply_for("nuôi mèo và chi phí")),
                             max_candidates=1)
    (row,) = result.report["replies"]
    assert row["proposal_results"][0]["validator"] == "copied_example"
    assert result.exports == []


def test_the_same_region_with_its_own_topic_is_not_refused(tmp_path):
    _, result, _ = base._run(tmp_path, base.FakeLLM(_reply_for("chuyện làm vườn cuối tuần")),
                             max_candidates=1)
    (row,) = result.report["replies"]
    assert row["proposal_results"][0]["validator"] != "copied_example"


def test_the_prompt_has_no_scratch_steps_the_json_only_reply_cannot_hold():
    prompt = _prompt()
    assert "CÁCH LÀM" not in prompt and "Ghi số dòng của các chỗ phải ngắt" not in prompt


def test_the_prompt_has_no_seam_noise_music_or_language_rules_left():
    prompt = _prompt().lower()
    for word in ("seam", "noise", "noisy", "music", "not_vietnamese", "ngoại", "bss", "unsep"):
        assert word not in prompt, word


def test_the_prompt_states_the_definition_the_thresholds_and_the_backchannels():
    prompt = _prompt()
    assert "ĐỊNH NGHĨA HỘI THOẠI" in prompt and "LƯỢT VÀ LỜI ĐỆM" in prompt
    assert "Bắt buộc 60-240s" in prompt                  # the config defaults
    assert "Độ dài tốt nhất 90-180s" in prompt
    assert "ừm" in prompt and "đúng rồi" in prompt and "dạ" in prompt
    assert "Số người đếm từ nhãn" in prompt


def test_the_finder_line_marks_a_break_point_as_hard_and_nothing_else():
    segs = base._talk(3, seg=7.0)
    finder = ConversationSelectionFinder(
        segs, TimelineMap(), base._quiet(), None, ConversationSelectionConfig(),
        bad_separation={1})
    service = _service()
    names = {base.A: "A", base.B: "B"}
    assert "hard=bad_separation" in service._finder_line(finder, 1, 2, names)
    assert "hard=" not in service._finder_line(finder, 0, 1, names)
    message = service._finder_message(finder, (0, 3), names)
    assert "#" not in message and "SPEAKER_" not in message


def test_the_first_line_of_a_window_has_no_gap_even_mid_record():
    segs = base._talk(3, seg=7.0)
    finder = ConversationSelectionFinder(
        segs, TimelineMap(), base._quiet(), None, ConversationSelectionConfig())
    service = _service()
    names = {base.A: "A", base.B: "B"}
    assert "gap=" in service._finder_line(finder, 1, 2, names)
    assert "gap=" not in service._finder_line(finder, 1, 1, names)
