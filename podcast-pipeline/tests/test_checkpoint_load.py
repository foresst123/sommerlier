import os

from utils.checkpoint import CheckpointManager


def _saved(tmp_path):
    cp = CheckpointManager(str(tmp_path), "job")
    cp.save("asr", {"a": list(range(1000))})
    return cp, cp._get_stage_path("asr", "pkl")


def test_load_round_trips_without_hashing_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CHECKPOINT_VERIFY_DIGEST", raising=False)
    cp, _ = _saved(tmp_path)
    assert cp.load("asr") == {"a": list(range(1000))}


def test_a_truncated_checkpoint_is_still_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("CHECKPOINT_VERIFY_DIGEST", raising=False)
    cp, path = _saved(tmp_path)
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 5)
    assert cp.load("asr") is None


def test_same_size_same_mtime_edit_is_caught_only_with_the_digest_switch(tmp_path, monkeypatch):
    cp, path = _saved(tmp_path)
    stat = os.stat(path)
    with open(path, "r+b") as fh:
        fh.seek(-3, os.SEEK_END)
        fh.write(b"\x00\x00\x00")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    monkeypatch.setenv("CHECKPOINT_VERIFY_DIGEST", "1")
    assert cp.load("asr") is None
