"""Rerun: delete a file's cache and output, forget it in the ledger, and nothing else."""
import os
import shutil
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.progress import ProgressLedger          # noqa: E402
from utils.rerun import (clear_for_rerun, rerun_enabled, rerun_targets,  # noqa: E402
                         unsafe_reason)


def _touch(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _world(tmp_path, names=("a", "b")):
    """An audio folder with a ledger, plus a cache folder and an output root."""
    audio_dir = tmp_path / "audio"
    cache = tmp_path / "cache"
    out_root = tmp_path / "out"
    paths = []
    for n in names:
        wav = audio_dir / f"{n}.wav"
        _touch(str(wav), "audio")
        _touch(str(cache / f"job_{n}" / "diarization" / "result.pkl"))
        _touch(str(out_root / n / "02_diarization" / "segments.json"))
        paths.append(str(wav))
    ledger = ProgressLedger(str(audio_dir))
    for p in paths:
        ledger.mark_done(p)
    ledger.set_quality(paths[0], {"normalized_quality": 0.9})
    ledger.save()
    return audio_dir, cache, out_root, paths, ledger


def _out_of(out_root):
    return lambda p: str(out_root / os.path.splitext(os.path.basename(p))[0])


def test_rerun_is_off_unless_asked():
    ns = types.SimpleNamespace
    assert rerun_enabled(ns()) is False
    assert rerun_enabled(ns(rerun=None)) is False
    assert rerun_enabled(ns(rerun=False)) is False
    assert rerun_enabled(ns(rerun=True)) is True


def test_cache_output_and_ledger_all_reset_and_audio_is_kept(tmp_path):
    audio_dir, cache, out_root, paths, ledger = _world(tmp_path)
    result = clear_for_rerun(paths, str(cache), "job", _out_of(out_root), ledger)

    assert sorted(result["cleared"]) == sorted(paths) and not result["skipped"]
    for n in ("a", "b"):
        assert not (cache / f"job_{n}").exists()
        assert not (out_root / n).exists()
        assert (audio_dir / f"{n}.wav").exists()
    assert not ledger.is_done(paths[0]) and not ledger.is_done(paths[1])


def test_the_ledger_on_disk_forgets_too(tmp_path):
    """save() merges the disk back into memory, which would resurrect a deleted key."""
    audio_dir, cache, out_root, paths, ledger = _world(tmp_path)
    clear_for_rerun(paths, str(cache), "job", _out_of(out_root), ledger)
    fresh = ProgressLedger(str(audio_dir))
    assert not fresh.is_done(paths[0]) and not fresh.is_done(paths[1])
    assert fresh.pending(paths) == paths


def test_the_dnsmos_score_survives_a_rerun(tmp_path):
    audio_dir, cache, out_root, paths, ledger = _world(tmp_path)
    clear_for_rerun(paths, str(cache), "job", _out_of(out_root), ledger)
    assert ProgressLedger(str(audio_dir)).get_quality(paths[0]) == {"normalized_quality": 0.9}


def test_only_the_files_of_this_run_are_touched(tmp_path):
    audio_dir, cache, out_root, paths, ledger = _world(tmp_path, names=("a", "b", "c"))
    clear_for_rerun(paths[:1], str(cache), "job", _out_of(out_root), ledger)
    assert not (cache / "job_a").exists()
    assert (cache / "job_b").exists() and (cache / "job_c").exists()
    assert (out_root / "b").exists() and ledger.is_done(paths[1]) and ledger.is_done(paths[2])


def test_a_file_held_by_another_pipeline_is_left_alone(tmp_path):
    audio_dir, cache, out_root, paths, ledger = _world(tmp_path)
    claims = types.SimpleNamespace(claimed_by_others=lambda ps: {paths[0]})
    result = clear_for_rerun(paths, str(cache), "job", _out_of(out_root), ledger, claims=claims)
    assert result["cleared"] == [paths[1]]
    assert result["skipped"][0][0] == paths[0]
    assert (cache / "job_a").exists() and ledger.is_done(paths[0])


def test_an_output_folder_that_holds_the_audio_is_refused(tmp_path):
    """audio at /x/audio/a.wav with an output root that makes the output /x/audio."""
    audio_dir, cache, out_root, paths, ledger = _world(tmp_path)
    result = clear_for_rerun(paths, str(cache), "job", lambda p: str(audio_dir), ledger)
    assert result["cleared"] == []
    assert all("holds the audio" in why for _, why in result["skipped"])
    assert (audio_dir / "a.wav").exists() and (audio_dir / "b.wav").exists()
    assert ledger.is_done(paths[0]) and (cache / "job_a").exists()


def test_a_symlinked_audio_is_protected_through_its_target(tmp_path):
    real = tmp_path / "real" / "ep.wav"
    _touch(str(real), "audio")
    link_dir = tmp_path / "links"
    os.makedirs(link_dir)
    link = link_dir / "ep.wav"
    os.symlink(real, link)
    assert unsafe_reason(str(tmp_path / "real"), str(link)) == "holds the audio file"
    assert unsafe_reason(str(tmp_path / "elsewhere" / "ep"), str(link)) is None


def test_the_filesystem_root_and_empty_paths_are_refused():
    assert unsafe_reason("", "/a/b.wav") == "empty path"
    assert unsafe_reason("/", "/a/b.wav") == "filesystem root"


def test_a_failed_delete_does_not_forget_the_file(tmp_path, monkeypatch):
    audio_dir, cache, out_root, paths, ledger = _world(tmp_path)
    real_rmtree = shutil.rmtree

    def flaky(path, *a, **k):
        if os.path.basename(path) == "job_a":
            raise OSError("busy")
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", flaky)
    result = clear_for_rerun(paths, str(cache), "job", _out_of(out_root), ledger)
    assert result["cleared"] == [paths[1]]
    assert "delete failed" in result["skipped"][0][1]
    assert ledger.is_done(paths[0]) and not ledger.is_done(paths[1])


def test_nothing_to_delete_still_forgets_the_ledger_entry(tmp_path):
    audio_dir, cache, out_root, paths, ledger = _world(tmp_path)
    shutil.rmtree(cache / "job_a")
    shutil.rmtree(out_root / "a")
    result = clear_for_rerun(paths[:1], str(cache), "job", _out_of(out_root), ledger)
    assert result["cleared"] == paths[:1] and not ledger.is_done(paths[0])


def test_targets_are_the_cache_folder_and_the_output_folder():
    assert rerun_targets("/x/ep 01.wav", "cache", "job", "/o/ep 01") == [
        os.path.join("cache", "job_ep 01"), "/o/ep 01"]


def test_forget_can_also_drop_the_score(tmp_path):
    audio_dir, cache, out_root, paths, ledger = _world(tmp_path)
    ledger.forget(paths, keep_quality=False)
    assert ledger.get_quality(paths[0]) is None
    assert ProgressLedger(str(audio_dir)).get_quality(paths[0]) is None
