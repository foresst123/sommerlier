#!/usr/bin/env python3
"""Write OUT/variants_manifest_preview.json from the variant folders that exist on disk.

    python tools/preview_manifest.py --out-root OUT --variants tools/variants_placement.json
    python tools/make_variant_lanes_page.py --manifest OUT/variants_manifest_preview.json

For looking at the pages while some variants are still running, or after variants_manifest.json was lost or
overwritten by a run of a few variants. Lists a variant when OUT/<id>/<stem>/05_asr/transcripts.json exists for at
least one stem; never touches variants_manifest.json.
"""
import argparse
import json
import sys
from pathlib import Path

POINTS = ("chain", "chain_pre_separation", "chain_pre_asr")


def as_list(spec):
    if not spec or str(spec).strip().lower() == "none":
        return []
    return list(spec) if isinstance(spec, (list, tuple)) else [t for t in str(spec).split(">") if t]


def build(out_root: Path, plan: dict) -> dict:
    variants = []
    for v in plan.get("variants", []):
        folder = out_root / v["id"]
        files = {p.parent.parent.name: str(p.parent.parent) for p in sorted(folder.glob("*/05_asr/transcripts.json"))}
        if not files:
            continue
        row = {k: val for k, val in v.items() if k not in POINTS}
        for k in POINTS:
            row[k] = as_list(v.get(k))
        row["files"] = files
        row["status"] = "ok" if (folder / ".done").is_file() else "partial"
        variants.append(row)
    return {"baseline": plan.get("baseline", "none"), "variants": variants}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--variants", required=True)
    a = ap.parse_args(argv)
    out_root = Path(a.out_root).resolve()
    manifest = build(out_root, json.loads(Path(a.variants).read_text(encoding="utf-8")))
    path = out_root / "variants_manifest_preview.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{path}: {len(manifest['variants'])} variants -> " + ", ".join(v["id"] for v in manifest["variants"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
