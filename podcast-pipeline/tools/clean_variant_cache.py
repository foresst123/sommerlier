#!/usr/bin/env python3
"""Remove the pipeline cache of THESE audio files for THESE variants, and nothing else.

    python tools/clean_variant_cache.py --audio-dir DIR --variants tools/variants_placement.json            # dry run
    python tools/clean_variant_cache.py --audio-dir DIR --variants tools/variants_placement.json --delete

The pipeline keeps one checkpoint folder per (variant, file): <cache dir>/exp-<variant id>_<file stem>. The names
are built from the variants file and the files in --audio-dir, so another dataset's folders in the same cache dir
are never matched. Other things it can clear on request:

  --out-root OUT   the hand-over markers (.done*, .ready_*, .failed_*) and the <stem> output folders of those
                   variants under OUT, plus OUT/variants_manifest.json. Without this a rerun finds the markers
                   and skips the variant as "up to date".

Never touched: the denoise step cache (keyed by the audio content), other cache folders, the input audio.
"""
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CACHES = ("cache", "cache_nomusic")          # relative to podcast-pipeline/
MARKER_GLOBS = (".done", ".done.*", ".ready_*", ".failed_*")


def _run_variants():
    spec = importlib.util.spec_from_file_location("run_variants_for_clean", HERE / "run_variants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def plan(cache_dirs, out_root, variant_ids, stems):
    """[(path, kind)] of what exists and would go."""
    found = []
    for cache in cache_dirs:
        for vid in variant_ids:
            for stem in stems:
                p = Path(cache) / f"exp-{vid}_{stem}"
                if p.is_dir() and not p.is_symlink():
                    found.append((p, "cache"))
    if out_root:
        out = Path(out_root)
        for vid in variant_ids:
            vdir = out / vid
            if not vdir.is_dir():
                continue
            for pat in MARKER_GLOBS:
                found += [(m, "marker") for m in sorted(vdir.glob(pat)) if m.is_file()]
            for stem in stems:
                if (vdir / stem).is_dir() and not (vdir / stem).is_symlink():
                    found.append((vdir / stem, "output"))
        for name in ("variants_manifest.json", ".manifest.lock"):
            if (out / name).is_file():
                found.append((out / name, "manifest"))
    return found


def fmt(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--cache-dir", action="append", help="cache folder (repeatable); default: cache and cache_nomusic")
    ap.add_argument("--out-root")
    ap.add_argument("--delete", action="store_true", help="really delete (default: only list)")
    a = ap.parse_args(argv)

    rv = _run_variants()
    ids = [v["id"] for v in rv.load_variants(a.variants)]
    stems = [f.stem for f in rv.collect_files(a.audio_dir, None)]
    root = HERE.parent
    caches = [Path(c) if Path(c).is_absolute() else root / c for c in (a.cache_dir or DEFAULT_CACHES)]
    found = plan(caches, a.out_root, ids, stems)

    print(f"files: {', '.join(stems)}   variants: {len(ids)}   cache dirs: {', '.join(map(str, caches))}")
    total = 0
    for path, kind in found:
        n = size_of(path)
        total += n
        print(f"  {kind:<8} {fmt(n):>9}  {path}")
    print(f"{len(found)} item(s), {fmt(total)}" + ("" if a.delete else "   (dry run: add --delete to remove them)"))
    if a.delete:
        for path, _ in found:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        print("deleted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
