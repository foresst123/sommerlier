"""The ledger that tracks which files a corpus run has finished.

Run:  python -m pytest tests/test_progress.py -q     (from podcast-pipeline/)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.progress import ProgressLedger


def _files(tmp_path, n):
    out = []
    for i in range(n):
        p = tmp_path / f"f{i:02d}.mp3"
        p.write_bytes(b"x")
        out.append(str(p))
    return out


def test_finished_files_are_not_offered_again(tmp_path):
    files = _files(tmp_path, 5)
    led = ProgressLedger(str(tmp_path))
    for p in files[:2]:
        led.mark_done(p)

    assert led.pending(files) == files[2:]


def test_progress_survives_a_restart(tmp_path):
    files = _files(tmp_path, 4)
    led = ProgressLedger(str(tmp_path))
    led.mark_done(files[0])
    led.save()

    assert ProgressLedger(str(tmp_path)).pending(files) == files[1:]


def test_files_added_mid_run_join_the_next_pass(tmp_path):
    """The whole reason the directory is re-scanned every pass."""
    files = _files(tmp_path, 2)
    led = ProgressLedger(str(tmp_path))
    for p in files:
        led.mark_done(p)
    assert led.pending(files) == []

    later = str(tmp_path / "new.mp3")
    (tmp_path / "new.mp3").write_bytes(b"x")
    assert led.pending(files + [later]) == [later]


def test_a_file_that_keeps_failing_is_eventually_left_out():
    """Otherwise one broken input stalls the corpus, retried every pass."""
    led = ProgressLedger(".")
    led.mark_failed("bad.mp3", "boom")
    assert led.pending(["bad.mp3"], max_attempts=2) == ["bad.mp3"]

    led.mark_failed("bad.mp3", "boom")
    assert led.pending(["bad.mp3"], max_attempts=2) == []
    assert led.attempts("bad.mp3") == 2


def test_a_file_that_later_succeeds_clears_its_failure(tmp_path):
    files = _files(tmp_path, 1)
    led = ProgressLedger(str(tmp_path))
    led.mark_failed(files[0], "transient")
    led.mark_done(files[0])

    assert led.failed == {}
    assert led.pending(files) == []


def test_a_corrupt_ledger_does_not_stop_the_run(tmp_path):
    """Worst case is reprocessing, which is wasteful but not wrong."""
    (tmp_path / "_sommelier_progress.json").write_text("{not json")
    files = _files(tmp_path, 2)

    assert ProgressLedger(str(tmp_path)).pending(files) == files


def test_the_ledger_is_written_atomically(tmp_path):
    files = _files(tmp_path, 1)
    led = ProgressLedger(str(tmp_path))
    led.mark_done(files[0])
    led.save()

    data = json.loads((tmp_path / "_sommelier_progress.json").read_text())
    assert "f00.mp3" in data["done"]
    leftovers = [p for p in os.listdir(tmp_path) if p.startswith(".progress-")]
    assert leftovers == [], "temp file should have been renamed away"


def test_the_input_audio_is_never_touched(tmp_path):
    """Deleting or moving inputs is unrecoverable and races with new copies."""
    files = _files(tmp_path, 3)
    led = ProgressLedger(str(tmp_path))
    for p in files:
        led.mark_done(p)
    led.save()

    for p in files:
        assert os.path.exists(p)
