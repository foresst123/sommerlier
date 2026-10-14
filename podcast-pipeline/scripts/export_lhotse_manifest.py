"""Build a SpeechWorld full-duplex Lhotse manifest from a conversation-export tree.

Run (from podcast-pipeline/):
  python scripts/export_lhotse_manifest.py --root exports/ --out manifest.jsonl.gz --validate
Exit code 1 when --validate finds a violation.
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="export tree (batch/topic/category/item folders)")
    ap.add_argument("--out", required=True, help="output .jsonl or .jsonl.gz")
    ap.add_argument("--language", default="vi")
    ap.add_argument("--emotion-min-conf", type=float, default=0.6)
    ap.add_argument("--coverage-max-uncovered", type=float, default=0.30,
                    help="seconds of unannotated speech tolerated per channel")
    ap.add_argument("--no-coverage-check", action="store_true")
    ap.add_argument("--validate", action="store_true", help="check the written manifest against spec section 9")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from services.lhotse_manifest_service import LhotseManifestParams, build_manifest

    params = LhotseManifestParams(
        language=args.language,
        emotion_min_conf=args.emotion_min_conf,
        coverage_check=not args.no_coverage_check,
        coverage_max_uncovered=args.coverage_max_uncovered)
    report = build_manifest(args.root, args.out, params)
    print(f"wrote {report.written} cut(s) to {report.output_path}; skipped {len(report.skipped)} item(s)")
    reasons = Counter(r for row in report.skipped for r in row.get("reasons", []))
    for reason, n in reasons.most_common():
        print(f"  skipped: {reason} x{n}")
    notes = Counter(str(n.get("reason") or n.get("reasons") or n) for n in report.notes)
    for note, n in notes.most_common(10):
        print(f"  note: {note} x{n}")

    if not args.validate:
        return 0
    from utils.lhotse_validate import validate_file, validate_with_lhotse
    violations = validate_file(report.output_path) + validate_with_lhotse(report.output_path)
    for v in violations[:50]:
        print(f"  {v}")
    if len(violations) > 50:
        print(f"  ... {len(violations) - 50} more")
    print(f"validation: {len(violations)} violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
