#!/usr/bin/env python3
"""Windowed Nemotron for diarization arm F. Runs in the Nemotron (NeMo) environment.

    <nemotron python> tools/windowed_nemotron_run.py --audio chunk.wav --out segments.json \
        [--piece-s 240] [--hop-s 10] [--window-s 30] [--config config.json --env a100]

Loads Nemotron the way nemotron_worker does, cuts pieces at Silero silences (about --piece-s long, 30 s
overlap), runs 30 s windows (hop --hop-s) through model.diarize(include_tensor_outputs=True) to get
per-frame speaker probabilities, and merges them with utils/windowed_clustering.run_file.
Writes {"segments": [{"start", "end", "speaker"}, ...]} in seconds from the start of the audio.

UNVERIFIED on GPU: written without NeMo or CUDA. The frame length is read from the first window's
output (window length / frame count), and include_tensor_outputs is the API tools/exp5_breakthrough_nemotron.py
uses; both must be checked on a real run.
"""

import argparse
import importlib.util
import json
import os
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(TOOLS_DIR)
# NeMo imports its own top-level `utils`. Keep tools/ off sys.path, and load the repo's modules by path
# (same reason as nemotron_worker._load_converter), so nothing here shadows NeMo's package.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != TOOLS_DIR]
SR = 16000


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(PIPELINE, rel))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module      # @dataclass looks its module up here while the class body runs
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="16 kHz mono WAV (the arm's chunk)")
    ap.add_argument("--out", required=True, help="segments.json to write")
    ap.add_argument("--piece-s", type=float, default=240.0)
    ap.add_argument("--hop-s", type=float, default=10.0)
    ap.add_argument("--window-s", type=float, default=30.0)
    ap.add_argument("--config", default=os.path.join(PIPELINE, "config.json"))
    ap.add_argument("--env", default="a100", help="environment profile in config.json")
    a = ap.parse_args(argv)

    import numpy as np
    import soundfile as sf
    import tempfile
    import torch

    wave, sr = sf.read(a.audio, dtype="float32", always_2d=False)
    if getattr(wave, "ndim", 1) == 2:
        wave = wave.mean(axis=1)
    if sr != SR:
        raise SystemExit(f"expected {SR} Hz audio, got {sr} Hz")
    duration = len(wave) / float(SR)

    nemo = _load("nemotron_worker", "nemotron_worker.py")
    model, cfg = nemo.load_model(a.config, a.env)
    vad = nemo.load_silero_vad()
    speech = vad(wave, SR, cfg)
    helpers = _load("nemotron_two_pass", "utils/nemotron_two_pass.py")
    wc = _load("windowed_clustering", "utils/windowed_clustering.py")

    def idx(t):
        return int(round(t * SR))

    def diarize_windows(clips):
        """Frame probabilities (frames, slots) for each clip, through the model's tensor outputs."""
        with tempfile.TemporaryDirectory(prefix="windowed_nemotron_") as tmp:
            paths = []
            for i, clip in enumerate(clips):
                path = os.path.join(tmp, f"win_{i:05d}.wav")
                sf.write(path, clip, SR)
                paths.append(path)
            out = model.diarize(audio=paths, batch_size=1, include_tensor_outputs=True)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError("model.diarize returned no tensor outputs; UNVERIFIED API, see exp5_breakthrough_nemotron.py")
        return [(p.squeeze(0) if p.ndim == 3 else p).detach().float().cpu().numpy() for p in out[1]]

    # Frame length from one real window: the model's frame grid, not an assumed 80 ms.
    probe_len = min(a.window_s, duration)
    probe = diarize_windows([wave[: idx(probe_len)]])[0]
    frame_s = probe_len / float(probe.shape[0])
    print(json.dumps({"status": "probe", "frame_s": round(frame_s, 5), "duration": round(duration, 3),
                      "speech_intervals": len(speech)}), flush=True)

    def make_prob_fn(piece):
        spans = wc.window_spans(piece.audio_duration, a.window_s, a.hop_s)
        clips = [wave[idx(piece.audio_start + s): idx(piece.audio_start + e)] for s, e in spans]
        table = dict(zip(spans, diarize_windows(clips))) if clips else {}

        def prob_fn(s, e):
            return table[(s, e)]
        return prob_fn

    wes = _load("wespeaker_embedding", "models/wespeaker_embedding.py")
    # WeSpeaker imports utils.cpu_plan lazily; NeMo's `utils` would hide ours, so register it by path.
    sys.modules["utils.cpu_plan"] = _load("utils.cpu_plan", "utils/cpu_plan.py")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedder = wes.WeSpeakerONNXEmbedder(device)

    def make_embed_fn(piece):
        def embed_fn(spans):
            clips = [wave[idx(piece.audio_start + s): idx(piece.audio_start + e)] for s, e in spans]
            rows = []
            for res in embedder.embed_batch(clips, sample_rate=SR, max_batch_size=4):
                if isinstance(res, Exception):
                    raise RuntimeError(f"WeSpeaker failed on a clip: {res}") from res
                rows.append(res.detach().cpu().numpy())
            return np.stack(rows)
        return embed_fn

    result = wc.run_file(duration, speech, make_prob_fn=make_prob_fn, make_embed_fn=make_embed_fn,
                         piece_s=a.piece_s, window_s=a.window_s, hop_s=a.hop_s, frame_s=frame_s,
                         vad_fn=helpers.build_vad_windows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"segments": result["segments"], "pieces": result["pieces"], "frame_s": frame_s,
                   "params": {"piece_s": a.piece_s, "window_s": a.window_s, "hop_s": a.hop_s}},
                  fh, ensure_ascii=False, indent=1)
    print(json.dumps({"status": "done", "segments": len(result["segments"]),
                      "pieces": len(result["pieces"])}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
