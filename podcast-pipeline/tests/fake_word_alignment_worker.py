"""Stand-in for word_alignment_worker.py: same protocol, no whisperx.

FAKE_WA_MODE picks the behaviour: ok (default), fail_multi (error when a request
carries more than one clip, so the service must retry clip by clip), error.
Every returned word carries the worker's pid so tests can see which process
did the work.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--language", default="vi")
parser.add_argument("--device", default="cpu")
parser.add_argument("--interpolate", default="nearest")
parser.add_argument("--threads", default="1")
parser.add_argument("--model-name", default=None)
parser.add_argument("--model-dir", default=None)
parser.add_argument("--cache-only", action="store_true")
parser.parse_known_args()

mode = os.environ.get("FAKE_WA_MODE", "ok")
delay = float(os.environ.get("FAKE_WA_SLEEP", "0.2"))

print(json.dumps({"status": "ready"}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    time.sleep(delay)
    if mode == "oom":
        print(json.dumps({"id": req["id"], "error":
                          "OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB"}),
              flush=True)
        continue
    if mode == "error" or (mode == "fail_multi" and len(req["items"]) > 1):
        print(json.dumps({"id": req["id"], "error": "boom"}), flush=True)
        continue
    audio = np.load(req["audio_path"])
    assert len(audio) == sum(item["n"] for item in req["items"])
    words_by_index = {}
    for item in req["items"]:
        tokens = item["text"].split()
        width = (item["end"] - item["start"]) / len(tokens)
        words_by_index[item["index"]] = [
            {"word": token, "start": round(item["start"] + n * width, 3),
             "end": round(item["start"] + (n + 1) * width, 3),
             "score": 0.9, "pid": os.getpid()}
            for n, token in enumerate(tokens)]
    print(json.dumps({"id": req["id"], "words_by_index": words_by_index}), flush=True)
