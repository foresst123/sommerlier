#!/usr/bin/env python3
"""Isolated TensorFlow worker. stdin: {audio_path}; stdout: {segments}."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr
    try:
        from inaSpeechSegmenter import Segmenter
        segmenter = Segmenter(vad_engine="smn", detect_gender=False, batch_size=args.batch_size)
        print(json.dumps({"status": "ready"}), file=protocol, flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("cmd") == "quit":
                    break
                segments = segmenter(request["audio_path"])
                response = {"segments": [[str(k), float(a), float(b)] for k, a, b in segments]}
            except Exception as exc:
                response = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(response), file=protocol, flush=True)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=protocol, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
