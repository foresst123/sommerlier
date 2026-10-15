import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import filter_language as fl


def _args(folder, cache, **over):
    ns = fl.build_parser().parse_args([str(folder), "--cache_dir", str(cache), "--log", str(folder / "log.tsv")])
    for key, value in over.items():
        setattr(ns, key, value)
    return ns


def _setup(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    for name in ("done.wav", "rejected.wav", "viet.wav", "[Vlog] english.wav"):
        (folder / name).write_bytes(b"x" * 10)
    (folder / "notes.txt").write_text("not audio")
    (folder / fl.LEDGER_NAME).write_text(json.dumps(
        {"done": {"done.wav": {}}, "rejected": {"rejected.wav": {}}, "failed": {}}))
    cache = tmp_path / "cache"
    for stem in ("viet", "[Vlog] english"):
        (cache / f"default_{stem}" / "asr").mkdir(parents=True)
        (cache / f"default_{stem}" / "asr" / "result.pkl").write_bytes(b"y" * 5)
    out = folder / "_final" / "-perf-on-abc-bss-True" / "[Vlog] english" / "01_music"
    out.mkdir(parents=True)
    (out / "music_map.json").write_text("{}")
    return folder, cache


def test_plan_skips_done_and_rejected_and_non_audio(tmp_path):
    folder, _ = _setup(tmp_path)
    todo, done, rejected = fl.plan(folder)
    assert [os.path.basename(p) for p in todo] == ["[Vlog] english.wav", "viet.wav"]
    assert [os.path.basename(p) for p in done] == ["done.wav"]
    assert [os.path.basename(p) for p in rejected] == ["rejected.wav"]


def test_verdict_needs_evidence_and_uses_mean_vietnamese_probability():
    assert fl.verdict([], 0.5) == (None, None, False)
    mean, top, flagged = fl.verdict([{"vi": 0.1, "en": 0.9}, {"vi": 0.3, "en": 0.7}], 0.5)
    assert top == "en" and flagged and abs(mean - 0.2) < 1e-9
    assert fl.verdict([{"vi": 0.9, "en": 0.1}], 0.5)[2] is False


def test_dry_run_deletes_nothing(tmp_path):
    folder, cache = _setup(tmp_path)
    detector = lambda p: [{"vi": 0.0, "en": 1.0}] if "english" in p else [{"vi": 0.95, "en": 0.05}]
    rows = fl.run(_args(folder, cache), detector=detector)
    assert [os.path.basename(r[0]) for r in rows] == ["[Vlog] english.wav"]
    assert (folder / "[Vlog] english.wav").exists()
    assert (cache / "default_[Vlog] english").exists()


def test_apply_removes_audio_outputs_and_cache_of_the_english_file_only(tmp_path):
    folder, cache = _setup(tmp_path)
    detector = lambda p: [{"vi": 0.0, "en": 1.0}] if "english" in p else [{"vi": 0.95, "en": 0.05}]
    fl.run(_args(folder, cache, apply=True), detector=detector)
    assert not (folder / "[Vlog] english.wav").exists()
    assert not (folder / "_final" / "-perf-on-abc-bss-True" / "[Vlog] english").exists()
    assert not (cache / "default_[Vlog] english").exists()
    for kept in ("viet.wav", "done.wav", "rejected.wav", fl.LEDGER_NAME):
        assert (folder / kept).exists()
    assert (cache / "default_viet").exists()


def test_trash_moves_the_audio_instead_of_deleting_it(tmp_path):
    folder, cache = _setup(tmp_path)
    trash = tmp_path / "trash"
    detector = lambda p: [{"vi": 0.0, "en": 1.0}] if "english" in p else [{"vi": 0.95}]
    fl.run(_args(folder, cache, apply=True, trash=str(trash)), detector=detector)
    assert (trash / "[Vlog] english.wav").exists()
    assert not (folder / "[Vlog] english.wav").exists()


def test_a_detector_error_never_drops_the_file(tmp_path):
    folder, cache = _setup(tmp_path)

    def broken(path):
        raise RuntimeError("decode failed")

    fl.run(_args(folder, cache, apply=True), detector=broken)
    assert (folder / "[Vlog] english.wav").exists() and (folder / "viet.wav").exists()
