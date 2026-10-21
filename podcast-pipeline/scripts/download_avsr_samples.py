"""Download N random raw_audio wavs from the HF dataset linhnv4dev/AVSR-CT-AI.

Only needs the standard library. Example:
    python scripts/download_avsr_samples.py --n 10 --out ../samples/avsr --seed 42
"""
import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

REPO = "linhnv4dev/AVSR-CT-AI"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data?limit=1000"
FILE = f"https://huggingface.co/datasets/{REPO}/resolve/main/data/{{id}}/raw_audio/{{id}}.wav"


def list_ids() -> list[str]:
    ids, url = [], API
    while url:  # the API pages at 1000 entries and points to the next page in the Link header
        with urllib.request.urlopen(url) as r:
            items = json.load(r)
            link = r.headers.get("Link", "")
        ids += [x["path"].split("/")[-1] for x in items if x["type"] == "directory"]
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return ids


def download(video_id: str, dst: Path) -> bool:
    try:
        urllib.request.urlretrieve(FILE.format(id=video_id), dst)
        return True
    except Exception as e:  # some folders may lack raw_audio
        print(f"  skip {video_id}: {e}", file=sys.stderr)
        dst.unlink(missing_ok=True)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("avsr_samples"))
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ids = list_ids()
    print(f"{len(ids)} recordings in dataset")
    random.Random(args.seed).shuffle(ids)

    got = 0
    for vid in ids:
        if got >= args.n:
            break
        dst = args.out / f"{vid}.wav"
        if dst.exists() or download(vid, dst):
            got += 1
            print(f"[{got}/{args.n}] {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
