"""One place reports ASR progress: numbers only move forward, one writer, no \\r."""

import logging
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.asr_progress import AsrProgress


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _logger():
    logger = logging.getLogger(f"asr-progress-{id(object())}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = Capture()
    logger.addHandler(handler)
    return logger, handler


def _progress(**kwargs):
    clock = Clock()
    kwargs.setdefault("interval", 0)
    return AsrProgress(clock=clock, **kwargs), clock


# --- what the lines say --------------------------------------------------------

def test_a_lane_shows_done_over_queued_with_a_percentage():
    progress, _ = _progress()
    progress.expect_files(1)
    progress.queued("a", "qwen3", 200)
    progress.done("qwen3", 50)

    line = progress.render()[1]
    assert "qwen3 50/200 (25%)" in line


def test_while_files_are_still_to_arrive_the_total_is_marked_as_growing():
    progress, _ = _progress()
    progress.expect_files(3)
    progress.queued("a", "qwen3", 100)
    progress.done("qwen3", 100)

    assert "100/100+" in progress.render()[1]
    assert "files 1/3 queued" in progress.render()[0]

    progress.queued("b", "qwen3", 50)
    progress.queued("c", "qwen3", 50)
    assert "100/200" in progress.render()[1] and "+" not in progress.render()[1].split("|")[0]


def test_a_lane_that_has_not_started_says_why():
    progress, clock = _progress()
    progress.expect_files(1)
    progress.lane_loading("phowhisper")
    clock.now += 42

    assert "phowhisper loading 42s" in progress.render()[1]

    progress.lane_ready("phowhisper")
    assert "loading" not in progress.render()[1]


def test_a_finished_lane_is_marked_and_files_report_their_stage():
    progress, _ = _progress()
    progress.expect_files(2)
    progress.queued("a", "m", 2)
    progress.queued("b", "m", 2)
    progress.done("m", 4)
    progress.lanes_done("a")
    progress.voting("a")
    progress.voted("b")

    head = progress.render()[0]
    assert "files 2/2 queued" in head and "voting 1" in head and "voted 1/2" in head


def test_rate_and_eta_come_from_time_since_the_lane_started_working():
    progress, clock = _progress()
    progress.expect_files(1)
    progress.queued("a", "m", 100)
    progress.lane_started("m")
    clock.now += 10
    progress.done("m", 20)

    line = progress.render()[1]
    assert "2.0/s" in line and "~40s" in line     # 80 left at 2 per second


# --- numbers never go backwards ------------------------------------------------

def test_done_and_total_never_decrease_and_done_never_passes_total():
    progress, _ = _progress()
    progress.expect_files(1)
    progress.queued("a", "m", 10)
    seen = []
    for _ in range(15):
        progress.done("m", 1)
        seen.append(progress.snapshot()["lanes"]["m"]["done"])

    assert seen == sorted(seen) and max(seen) == 10


# --- one writer, one line per report, no carriage returns -----------------------

def test_reports_are_log_records_without_carriage_returns():
    logger, handler = _logger()
    progress, _ = _progress(logger=logger)
    progress.expect_files(1)
    progress.queued("a", "m", 4)
    progress.done("m", 2)

    progress.report()

    assert handler.messages and all("\r" not in m for m in handler.messages)
    assert all(m.startswith("[ASR]") for m in handler.messages)


def test_an_unchanged_state_is_not_reported_twice():
    logger, handler = _logger()
    progress, _ = _progress(logger=logger)
    progress.expect_files(1)
    progress.queued("a", "m", 4)
    progress.report()
    count = len(handler.messages)

    progress.report()
    assert len(handler.messages) == count

    progress.done("m", 1)
    progress.report()
    assert len(handler.messages) > count


def test_concurrent_updates_add_up():
    progress, _ = _progress()
    progress.expect_files(1)
    progress.queued("a", "m", 4000)

    def work():
        for _ in range(500):
            progress.done("m", 1)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert progress.snapshot()["lanes"]["m"]["done"] == 4000


def test_the_reporter_thread_reports_until_stopped_and_then_once_more():
    logger, handler = _logger()
    progress = AsrProgress(logger=logger, interval=0.02)
    progress.expect_files(1)
    progress.queued("a", "m", 2)
    progress.start()
    progress.done("m", 1)
    deadline = threading.Event()
    deadline.wait(0.2)
    progress.done("m", 1)
    progress.stop()

    assert any("m 2/2" in m for m in handler.messages)
    assert not progress._thread.is_alive()


def test_a_label_names_whose_progress_a_line_is():
    progress, _ = _progress(label="talk.mp3")
    progress.expect_files(1)
    assert all(line.startswith("[ASR] talk.mp3 ") for line in progress.render())


def test_a_lane_that_could_not_start_says_so_and_stays_failed():
    progress, _ = _progress()
    progress.expect_files(1)
    progress.queued("a", "pho", 3)
    progress.lane_loading("pho")
    progress.lane_failed("pho")
    progress.lane_finished("pho")           # the scheduler releases it afterwards

    assert "pho 0/3" in progress.render()[1] and "FAILED" in progress.render()[1]
