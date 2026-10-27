"""One dataset manifest: per-recording parts in lhotse/.parts, merged under a lock."""
import gzip
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("utils.lhotse_duplex")

from services.lhotse_manifest_service import build_recording_manifest, merge_manifests
from tests.test_lhotse_manifest import _make_item


def _cuts(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh]


def _three_recordings(root):
    for stem, n in (("a", 2), ("b", 1), ("c", 1)):
        for i in range(n):
            _make_item(root / "interrupt" / f"{stem}_{i + 1}s", source_stem=stem)


def _loose(root):
    lhotse = root / "lhotse"
    return sorted(n for n in os.listdir(lhotse) if n.endswith(".jsonl.gz") and n != "manifest.jsonl.gz")


def _merged(root):
    return str(root / "lhotse" / "manifest.jsonl.gz")


def test_three_recordings_make_one_manifest(tmp_path):
    _three_recordings(tmp_path)
    for stem in ("a", "b", "c"):
        build_recording_manifest(str(tmp_path), stem)
    cuts = _cuts(_merged(tmp_path))
    assert len(cuts) == 4 and len({c["id"] for c in cuts}) == 4
    assert _loose(tmp_path) == []
    assert sorted(os.listdir(tmp_path / "lhotse" / ".parts")) == ["a.jsonl.gz", "b.jsonl.gz", "c.jsonl.gz"]
    assert sorted(os.listdir(tmp_path / "lhotse")) == [".merge.lock", ".parts", "manifest.jsonl.gz"]


def test_rerun_one_recording_does_not_duplicate(tmp_path):
    _three_recordings(tmp_path)
    for stem in ("a", "b", "c"):
        build_recording_manifest(str(tmp_path), stem)
    build_recording_manifest(str(tmp_path), "a")
    cuts = _cuts(_merged(tmp_path))
    assert len(cuts) == 4 and len({c["id"] for c in cuts}) == 4
    assert _loose(tmp_path) == []


def test_parallel_builds_do_not_break_the_manifest(tmp_path):
    _three_recordings(tmp_path)
    errors = []

    def run(stem):
        try:
            build_recording_manifest(str(tmp_path), stem)
        except Exception as exc:               # reported by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(s,)) for s in ("a", "b", "c", "a", "b", "c")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(_cuts(_merged(tmp_path))) == 4
    assert _loose(tmp_path) == []


def test_duplicate_cut_id_keeps_first_and_warns(tmp_path, caplog):
    _three_recordings(tmp_path)
    build_recording_manifest(str(tmp_path), "b")
    parts = tmp_path / "lhotse" / ".parts"
    with gzip.open(parts / "b.jsonl.gz", "rt", encoding="utf-8") as fh:
        line = fh.readline()
    with gzip.open(parts / "c.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(line)                          # same cut id in a second part
    with caplog.at_level("WARNING"):
        merge_manifests(str(tmp_path))
    assert len(_cuts(_merged(tmp_path))) == 1
    assert any("duplicate cut id" in r.getMessage() for r in caplog.records)


def test_old_loose_files_removed_but_strangers_kept(tmp_path):
    _three_recordings(tmp_path)
    build_recording_manifest(str(tmp_path), "a")
    lhotse = tmp_path / "lhotse"
    (lhotse / "a.jsonl.gz").write_bytes(b"old layout")
    (lhotse / "notes.txt").write_text("keep me", encoding="utf-8")
    merge_manifests(str(tmp_path))
    assert not (lhotse / "a.jsonl.gz").exists()
    assert (lhotse / "notes.txt").exists()
