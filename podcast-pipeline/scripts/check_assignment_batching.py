"""Compare real WeSpeaker batch/single embeddings on clips from local audio.

This checks the embedding frontend/backend, not complete separation quality.
Run on the deployment GPU with its existing environment before accepting a
speedup. No package changes or model conversion are needed.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.wespeaker_embedding import WeSpeakerONNXEmbedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.clips < 2 or args.seconds <= 0 or args.threads < 1:
        parser.error("clips >= 2, seconds > 0 and threads >= 1 are required")
    if args.model_path:
        os.environ["WESPEAKER_MODEL_PATH"] = args.model_path
    torch.set_num_threads(args.threads)
    with sf.SoundFile(args.audio) as audio:
        sr = audio.samplerate
        length = int(args.seconds * sr)
        if len(audio) < length:
            parser.error("audio shorter than the requested clip duration")
        clips = []
        for index, offset in enumerate(np.linspace(0, len(audio) - length, args.clips, dtype=int)):
            audio.seek(int(offset))
            # Include one different frame length to verify the non-padded path.
            count = length if index < args.clips - 1 else max(1, length * 3 // 4)
            clips.append(audio.read(count, dtype="float32", always_2d=True).mean(axis=1))
    model = WeSpeakerONNXEmbedder(args.device, threads=args.threads)
    model.embed(clips[0], sr)     # exclude model load/warm-up from both timings
    start = time.perf_counter()
    singles = torch.stack([model.embed(clip, sr).cpu() for clip in clips])
    single_seconds = time.perf_counter() - start
    start = time.perf_counter()
    values = model.embed_batch(clips, sr)
    for value in values:
        if isinstance(value, Exception):
            raise value
    batched = torch.stack([value.cpu() for value in values])
    batch_seconds = time.perf_counter() - start
    single_norm = torch.nn.functional.normalize(singles, dim=1)
    batch_norm = torch.nn.functional.normalize(batched, dim=1)
    old_sim, new_sim = single_norm @ single_norm.T, batch_norm @ batch_norm.T
    parity = torch.allclose(singles, batched, rtol=1e-4, atol=1e-5)
    qc_equal = torch.equal(old_sim >= 0.2, new_sim >= 0.2)
    report = {
        "device": args.device, "torch": torch.__version__, "clips": len(clips),
        "input_shape": model._get_session().get_inputs()[0].shape,
        "single_seconds": single_seconds, "batch_seconds": batch_seconds,
        "max_embedding_abs_diff": float((singles - batched).abs().max()),
        "max_cosine_abs_diff": float((old_sim - new_sim).abs().max()),
        "embedding_allclose": parity, "pairwise_qc_0_2_equal": qc_equal,
        "batch_stats": dict(model.batch_stats),
        "note": "One local timing sample; not an A100 or end-to-end speedup unless run there.",
    }
    print(json.dumps(report, indent=2))
    return 0 if parity and qc_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
