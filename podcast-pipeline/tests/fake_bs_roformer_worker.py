"""Stand-in for bs_roformer_worker.py: same protocol, no models.

Behaviour comes from the --kwargs JSON: fake_mode is echo (default; halves the
audio), none, error, or die_once (exit on the first request, once, tracked by
the die_marker file, so a restarted worker then behaves like echo).
"""
import argparse
import json
import os
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="")
parser.add_argument("--kwargs", default="{}")
args = parser.parse_args()
cfg = json.loads(args.kwargs)
mode = cfg.get("fake_mode", "echo")
marker = cfg.get("die_marker")

print(json.dumps({"status": "ready"}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if mode == "die_once" and marker and not os.path.exists(marker):
        open(marker, "w").close()
        os._exit(1)
    if mode == "error":
        print(json.dumps({"id": req["id"], "error": "boom"}), flush=True)
        continue
    if mode == "none":
        print(json.dumps({"id": req["id"], "result": None}), flush=True)
        continue
    audio = np.load(req["audio_path"])
    out_path = os.path.splitext(req["audio_path"])[0] + "_out.npy"
    np.save(out_path, (audio * 0.5).astype(np.float32))
    print(json.dumps({
        "id": req["id"], "out_path": out_path, "out_sr": req["sample_rate"],
        "stereo_in": bool(audio.ndim == 2 and audio.shape[1] > 1),
    }), flush=True)
