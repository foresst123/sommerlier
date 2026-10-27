import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import filter_language as fl
from services.lhotse_manifest_service import merge_manifests


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


def test_include_done_checks_finished_files_too(tmp_path):
    folder, _ = _setup(tmp_path)
    todo, done, _ = fl.plan(folder, include_done=True)
    assert "done.wav" in [os.path.basename(p) for p in todo] and done == []


def _dataset(tmp_path, stem="[Vlog] english"):
    root = tmp_path / "dataset"
    safe = fl.safe_name(stem)
    keep_stem = "viet"
    for category, name, source in (("turn_by_turn", f"{safe}_300s", stem),
                                   ("backchannel", f"{safe}_900s", stem),
                                   ("turn_by_turn", f"{keep_stem}_100s", keep_stem)):
        item = root / category / name
        item.mkdir(parents=True)
        (item / "conversation.json").write_text(json.dumps({"source_stem": source}))
        (item / "stereo_2ch.wav").write_bytes(b"w" * 8)
    (root / "lhotse" / "audio").mkdir(parents=True)
    parts = root / "lhotse" / ".parts"
    parts.mkdir(parents=True)
    for part_stem in (safe, keep_stem):
        with gzip.open(parts / f"{part_stem}.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({"id": f"{part_stem}-cut"}) + "\n")
    merge_manifests(str(root))
    rows = [{"source_stem": stem, "path": "a"}, {"source_stem": keep_stem, "path": "b"},
            {"source_stem": stem, "path": "c"}]
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def test_purge_dataset_removes_only_that_recordings_items_manifest_and_rows(tmp_path):
    root = _dataset(tmp_path)
    items, files = fl.dataset_targets(str(root), "[Vlog] english")
    assert len(items) == 2 and len(files) == 1
    assert fl.purge_dataset(str(root), "[Vlog] english") == (2, 2)
    assert not (root / "turn_by_turn" / "_Vlog_english_300s").exists()
    assert (root / "turn_by_turn" / "viet_100s").exists()
    assert not (root / "lhotse" / ".parts" / "_Vlog_english.jsonl.gz").exists()
    assert (root / "lhotse" / ".parts" / "viet.jsonl.gz").exists()
    with gzip.open(root / "lhotse" / "manifest.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert [json.loads(x)["id"] for x in fh] == ["viet-cut"]     # the one dataset manifest, rebuilt
    left = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()]
    assert [r["source_stem"] for r in left] == ["viet"]


def test_apply_with_include_done_and_dataset_root_cleans_the_dataset(tmp_path):
    folder, cache = _setup(tmp_path)
    (folder / fl.LEDGER_NAME).write_text(json.dumps(
        {"done": {"[Vlog] english.wav": {}}, "rejected": {}, "failed": {}}))
    root = _dataset(tmp_path)
    detector = lambda p: [{"vi": 0.0, "en": 1.0}] if "english" in p else [{"vi": 0.95}]
    fl.run(_args(folder, cache, apply=True, include_done=True, dataset_root=str(root)), detector=detector)
    assert not (folder / "[Vlog] english.wav").exists()
    assert not any(p.name.startswith("_Vlog_english") for c in ("turn_by_turn", "backchannel")
                   for p in (root / c).iterdir())
    assert (root / "turn_by_turn" / "viet_100s").exists()


# --- run as a script: `python -I scripts/filter_language.py` ----------------
#
# -I drops the cwd and PYTHON* variables from sys.path, which is what a script run gets
# for sys.path[0] (scripts/). The launcher puts scripts/ there itself, imports the file as
# a module and swaps in a fake detector, since faster-whisper is not needed to test deletion.

_LAUNCHER = """
import sys
scripts, mode, script_args = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, scripts)
if mode == "block":                     # as if the lhotse merge module were missing
    sys.modules["services.lhotse_manifest_service"] = None
import filter_language as fl


class _Detector:
    def __init__(self, *a, **k):
        pass

    def __call__(self, path):
        return [{"vi": 0.0, "en": 1.0}] if "english" in path else [{"vi": 0.95, "en": 0.05}]


fl.WhisperLanguageId = _Detector
sys.exit(fl.main(script_args))
"""


def _run_isolated(cwd, mode, *script_args):
    import subprocess
    import sys
    return subprocess.run(
        [sys.executable, "-I", "-c", _LAUNCHER, os.path.join(ROOT, "scripts"), mode, *script_args],
        cwd=str(cwd), capture_output=True, text=True, timeout=300)


def test_apply_with_dataset_root_works_from_another_cwd_with_isolated_python(tmp_path):
    """The regression: purge_dataset imported services.lhotse_manifest_service only after it had
    deleted the recording, and under a script run `services` was not importable -> the manifest
    was left broken. The import now happens at startup."""
    folder, cache = _setup(tmp_path)
    root = _dataset(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    out = _run_isolated(elsewhere, "run", str(folder), "--apply", "--dataset_root", str(root),
                        "--cache_dir", str(cache), "--log", str(tmp_path / "log.tsv"))
    assert out.returncode == 0, out.stderr[-2000:]
    assert "ModuleNotFoundError" not in out.stderr
    assert not (folder / "[Vlog] english.wav").exists()
    assert not (root / "turn_by_turn" / "_Vlog_english_300s").exists()
    with gzip.open(root / "lhotse" / "manifest.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert [json.loads(x)["id"] for x in fh] == ["viet-cut"], "manifest must be rebuilt"
    left = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()]
    assert [r["source_stem"] for r in left] == ["viet"]


def test_a_missing_merge_module_stops_the_run_before_anything_is_deleted(tmp_path):
    folder, cache = _setup(tmp_path)
    root = _dataset(tmp_path)
    out = _run_isolated(tmp_path, "block", str(folder), "--apply", "--dataset_root", str(root),
                        "--cache_dir", str(cache), "--log", str(tmp_path / "log.tsv"))
    assert out.returncode != 0
    assert (folder / "[Vlog] english.wav").exists()
    assert (root / "turn_by_turn" / "_Vlog_english_300s").exists()
    assert (cache / "default_[Vlog] english").exists()


def test_a_done_file_is_left_alone_without_include_done(tmp_path):
    folder, cache = _setup(tmp_path)
    (folder / fl.LEDGER_NAME).write_text(json.dumps(
        {"done": {"[Vlog] english.wav": {}}, "rejected": {}, "failed": {}}))
    detector = lambda p: [{"vi": 0.0, "en": 1.0}]
    fl.run(_args(folder, cache, apply=True), detector=detector)
    assert (folder / "[Vlog] english.wav").exists()


def test_depth_two_reaches_subfolders_and_uses_their_own_ledger_and_final(tmp_path):
    root = tmp_path / "root"
    sub = root / "batch1"
    sub.mkdir(parents=True)
    (root / "top.wav").write_bytes(b"x")
    (sub / "deep english.wav").write_bytes(b"x")
    (sub / "deep done.wav").write_bytes(b"x")
    (sub / fl.LEDGER_NAME).write_text(json.dumps({"done": {"deep done.wav": {}}, "rejected": {}}))
    out = sub / "_final" / "params" / "deep english" / "01_music"
    out.mkdir(parents=True)
    (out / "m.json").write_text("{}")
    assert [os.path.basename(p) for p in fl.list_audio(root, 1)] == ["top.wav"]
    todo, done, _ = fl.plan(root, depth=2)
    assert sorted(os.path.basename(p) for p in todo) == ["deep english.wav", "top.wav"]
    assert [os.path.basename(p) for p in done] == ["deep done.wav"]
    detector = lambda p: [{"vi": 0.0, "en": 1.0}] if "english" in p else [{"vi": 0.9}]
    args = fl.build_parser().parse_args([str(root), "--depth", "2", "--apply",
                                         "--cache_dir", str(tmp_path / "cache"), "--log", str(tmp_path / "l.tsv")])
    fl.run(args, detector=detector)
    assert not (sub / "deep english.wav").exists()
    assert not (sub / "_final" / "params" / "deep english").exists()
    assert (root / "top.wav").exists() and (sub / "deep done.wav").exists()
