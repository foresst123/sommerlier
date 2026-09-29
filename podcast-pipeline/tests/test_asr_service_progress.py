"""ASR progress reaches the log as plain lines from one reporter: nothing is written
over another line, and the numbers end at the real totals. Fake models, no GPU."""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_asr_cross_file import _audio, _segments, _service


def _capture(svc):
    records = []

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger(f"asr-service-progress-{id(svc)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(Handler())
    svc.logger = logger
    return records


def _last_report(records):
    """The head and lane lines of the last progress report."""
    lines = [r for r in records if r.startswith("[ASR]")]
    head = max(i for i, r in enumerate(lines) if " files " in r)
    return lines[head], lines[head + 1]


def test_the_per_file_path_writes_no_carriage_return_line_and_no_vote_bar(capsys):
    svc = _service(cross_file=False)
    records = _capture(svc)

    svc.process(_segments(3), _audio())

    out = capsys.readouterr()
    assert "\r" not in out.out and "[ASR]" not in out.out
    assert "[ROVER]" not in out.err and "Bầu chọn" not in out.err
    _head, lanes = _last_report(records)
    assert "whisper 3/3" in lanes and "phowhisper 3/3" in lanes and "qwen3 3/3" in lanes


def test_the_cross_file_path_reports_every_lane_complete_and_the_file_voted(capsys):
    svc = _service(cross_file=True)
    records = _capture(svc)
    svc.begin_cross_file_stage(1)
    try:
        svc.process(_segments(3), _audio())
        svc.settle_file()
    finally:
        svc.end_cross_file_stage()

    assert "\r" not in capsys.readouterr().out
    head, lanes = _last_report(records)
    assert "files 1/1 queued" in head and "voted 1/1" in head
    assert all(f"{name} 3/3" in lanes for name in ("whisper", "phowhisper", "qwen3"))


def test_a_finished_file_logs_how_many_transcripts_were_voted():
    svc = _service(cross_file=False)
    records = _capture(svc)
    svc.process(_segments(3), _audio())
    assert any("3 transcripts voted" in r for r in records)


def test_progress_lines_carry_the_file_name_when_one_is_known():
    from utils import profiling
    svc = _service(cross_file=False)
    records = _capture(svc)
    with profiling.file_stage("asr", "/data/talk one.mp3"):
        svc.process(_segments(2), _audio())

    assert any(r.startswith("[ASR] talk one.mp3 ") for r in records)
