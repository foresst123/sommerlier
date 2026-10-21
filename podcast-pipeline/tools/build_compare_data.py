#!/usr/bin/env python3
"""Build compare_data.json from a variants manifest (schema: compare_lib/SCHEMA.md).

  python tools/build_compare_data.py --manifest OUT/variants_manifest.json \
      --out compare_data.json [--files stem1,stem2] [--reference stem=path ...]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.compare_lib.build import build_compare_data, write_compare_data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="compare_data.json")
    ap.add_argument("--files", help="comma-separated file stems to include (default: all)")
    ap.add_argument("--reference", action="append", default=[], metavar="STEM=PATH",
                    help="reference transcript JSON for a file (overrides the manifest)")
    ap.add_argument("--no-embed", action="store_true", help="skip the speaker-embedding metric")
    args = ap.parse_args(argv)

    refs = {}
    for item in args.reference:
        stem, sep, path = item.partition("=")
        if not sep or not stem or not path:
            ap.error(f"--reference expects STEM=PATH, got {item!r}")
        refs[stem] = path
    files = [s for s in args.files.split(",") if s] if args.files else None

    data = build_compare_data(args.manifest, files=files, references=refs, embed=not args.no_embed)
    write_compare_data(data, args.out)
    print(f"wrote {args.out}: {len(data['variants'])} variants, {len(data['files'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
