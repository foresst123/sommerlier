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


def test_a_failed_file_has_its_partial_work_cleared(tmp_path):
    """A failed file leaves a checkpoint behind, and the checkpoint is the
    dangerous half: the next attempt would reload the stages that finished and
    jump straight back to the one that broke, with the same state that broke
    it."""
    led = ProgressLedger(str(tmp_path))

    checkpoint = tmp_path / "cache" / "job_f00"
    (checkpoint / "diarization").mkdir(parents=True)
    (checkpoint / "diarization" / "result.pkl").write_bytes(b"x")
    output = tmp_path / "out" / "f00"
    output.mkdir(parents=True)
    (output / "f00.json").write_text("{}")

    removed = led.discard_partial_output(str(checkpoint), str(output))

    assert len(removed) == 2
    assert not checkpoint.exists()
    assert not output.exists()


def test_clearing_tolerates_paths_that_are_not_there(tmp_path):
    """A file can fail before writing anything; the point is to end up with
    nothing, not to find something to delete."""
    led = ProgressLedger(str(tmp_path))
    assert led.discard_partial_output(str(tmp_path / "missing"), None) == []


def test_a_failed_file_is_offered_again_next_pass(tmp_path):
    files = _files(tmp_path, 2)
    led = ProgressLedger(str(tmp_path))
    led.mark_failed(files[0], "OOM")
    led.save()

    fresh = ProgressLedger(str(tmp_path))
    assert files[0] in fresh.pending(files), "a failure must not count as done"


def test_a_memory_only_ledger_neither_reads_nor_writes(tmp_path):
    # `--audio` runs one file with no input folder of its own: its ledger must not
    # leave a record, or the next run of the same file would be skipped as done.
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    first = ProgressLedger(str(tmp_path), persist=False)
    first.mark_done(str(audio), seconds=3)
    first.save()
    assert not (tmp_path / "_sommelier_progress.json").exists()
    assert first.pending([str(audio)]) == []

    again = ProgressLedger(str(tmp_path), persist=False)
    assert again.pending([str(audio)]) == [str(audio)]


def test_a_memory_only_ledger_ignores_an_existing_file(tmp_path):
    real = ProgressLedger(str(tmp_path))
    real.mark_done(str(tmp_path / "a.wav"))
    real.save()
    assert ProgressLedger(str(tmp_path), persist=False).pending([str(tmp_path / "a.wav")])


def test_main_gives_a_single_audio_file_a_memory_only_ledger():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "main.py"), encoding="utf-8").read()
    assert "persist=False" in src and "ProgressLedger(args.audio_dir" in src


# --- input-quality rejection is final -------------------------------------------

def test_a_rejected_file_is_never_offered_again(tmp_path):
    files = _files(tmp_path, 2)
    led = ProgressLedger(str(tmp_path))
    led.mark_rejected(files[0], "DNSMOS rejected")
    led.save()

    assert led.attempts(files[0]) == 0, "a rejection is not a counted failure"
    assert ProgressLedger(str(tmp_path)).pending(files) == files[1:]
    assert ProgressLedger(str(tmp_path)).pending(files, max_attempts=0) == files[1:]


def test_a_rejection_replaces_an_earlier_failure_and_done_replaces_a_rejection(tmp_path):
    files = _files(tmp_path, 1)
    led = ProgressLedger(str(tmp_path))
    led.mark_failed(files[0], "OOM")
    led.mark_rejected(files[0], "too noisy")
    assert led.failed == {} and led.is_rejected(files[0])

    led.mark_done(files[0])
    assert led.rejected == {} and led.is_done(files[0])


def test_record_failure_tells_a_rejection_from_a_retryable_failure(tmp_path):
    files = _files(tmp_path, 3)
    led = ProgressLedger(str(tmp_path))

    batch_style = ("music: AudioQualityRejected: DNSMOS rejected a.wav: "
                   "normalized_quality=0.412 < threshold=0.500 (OVRL=2.648)")
    file_style = "AudioQualityRejected: DNSMOS rejected a.wav: normalized_quality=0.1"
    assert led.record_failure(files[0], batch_style) == "rejected"
    assert led.record_failure(files[1], file_style) == "rejected"

    # A scorer that could not run says nothing about the audio: that one is retried.
    scorer = "music: AudioQualityGateError: DNSMOS input check failed for a.wav: OSError"
    assert led.record_failure(files[2], scorer) == "failed"
    assert led.pending(files) == [files[2]]


def test_rejections_show_in_the_summary_and_not_as_remaining(tmp_path):
    files = _files(tmp_path, 4)
    led = ProgressLedger(str(tmp_path))
    led.mark_done(files[0])
    led.mark_rejected(files[1], "x")
    led.mark_failed(files[2], "OOM")

    assert led.summary(len(files)) == "1 done, 1 failed, 1 rejected, 1 remaining"


def test_the_rejection_marker_is_the_real_exception_name():
    from services.dnsmos_gate import AudioQualityRejected
    from utils.progress import REJECTION_TYPE, is_input_rejection

    assert REJECTION_TYPE == AudioQualityRejected.__name__
    exc = AudioQualityRejected("a.wav", {"normalized_quality": 0.1, "threshold": 0.5,
                                         "OVRL": 1.4})
    assert is_input_rejection(f"{type(exc).__name__}: {exc}")
    assert not is_input_rejection("RuntimeError: Failed to start vLLM refinement workers")
    assert not is_input_rejection(None)


def test_main_records_every_failure_through_the_ledger_classifier():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "main.py"), encoding="utf-8").read()
    calls = src.count("_record_failure(ledger") - 1          # minus its own def
    assert calls == 2, "both run modes must classify"
    assert "ledger.mark_failed(" not in src, "a bare mark_failed would retry a rejection"


# --- an early-stopped run must not mark files done --------------------------------

def test_pending_can_skip_files_this_invocation_already_visited(tmp_path):
    files = _files(tmp_path, 3)
    led = ProgressLedger(str(tmp_path))
    # Nothing is marked done by an early-stopped run, so only `exclude` keeps the
    # same group from being offered again and again.
    assert led.pending(files) == files
    assert led.pending(files, exclude={files[0], files[1]}) == files[2:]
    assert led.pending(files, exclude=set(files)) == []


def test_main_does_not_mark_files_done_after_stop_after():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "main.py"), encoding="utf-8").read()
    assert 'partial_run = getattr(args, "stop_after", None) is not None' in src
    assert "ledger.pending(run_snapshot, exclude=attempted)" in src
    assert "attempted.update(group)" in src
    assert src.count("ledger.mark_done(") == 2
    assert src.count("not partial_run") == 2, "both run modes must honour it"
