"""The finder prompt: per-line durations, and the worked examples it carries."""

import json
import re

import test_conversation_export_service as base
from services.conversation_export_service import (
    ConversationExportService, parse_finder_reply)
from utils.conversation_selection import ConversationSelectionConfig, ConversationSelectionFinder
from utils.excise import TimelineMap
from utils.noise_map import NoiseTrack
from utils.transcript_windows import line_number  # noqa: F401  (keeps the import path honest)


def _service():
    return ConversationExportService(base.FakeLLM("[]"))


def _prompt():
    return _service()._finder_system_prompt()


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
    assert line.startswith("[1] 0:00-0:07 (d=7.0s) A {")


def test_the_prompt_says_to_take_durations_from_the_timestamps():
    prompt = _prompt()
    assert "(d=thời lượng)" in prompt and "TÍNH THỜI LƯỢNG" in prompt
    assert "không ước lượng theo số dòng" in prompt.replace("TUYỆT ĐỐI ", "")
    assert not re.search(r"\{[a-z_]+\}", prompt), "an unfilled {placeholder} is left in the prompt"


def test_the_prompt_carries_three_worked_examples():
    examples = _examples(_prompt())
    assert len(examples) == 3
    assert [len(lines) for lines, _ in examples] == [7, 7, 3]


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


def test_the_second_example_isolates_a_single_line_and_the_third_rejects_a_monologue():
    (_, one), (_, two), (_, three) = _examples(_prompt())
    assert [(r["start_line"], r["end_line"], r["keep"]) for r in two] == [
        (1, 3, True), (4, 4, False), (5, 7, True)]
    assert [r["keep"] for r in one] == [True] and [r["keep"] for r in three] == [False]


def _reply_for(topic):
    return json.dumps([
        {"start_line": 1, "end_line": 20, "keep": True, "self_contained": 5,
         "topic": topic, "reason": "x"},
        {"start_line": 21, "end_line": 30, "keep": False, "self_contained": 2,
         "topic": "phần còn lại", "reason": "x"}])


def test_a_region_copied_from_an_example_is_refused(tmp_path):
    _, result, _ = base._run(tmp_path, base.FakeLLM(_reply_for("khám sức khỏe định kỳ")),
                             max_candidates=1)
    (row,) = result.report["replies"]
    assert row["proposal_results"][0]["validator"] == "copied_example"
    assert result.exports == []


def test_the_same_region_with_its_own_topic_is_not_refused(tmp_path):
    _, result, _ = base._run(tmp_path, base.FakeLLM(_reply_for("chuyện làm vườn cuối tuần")),
                             max_candidates=1)
    (row,) = result.report["replies"]
    assert row["proposal_results"][0]["validator"] != "copied_example"
