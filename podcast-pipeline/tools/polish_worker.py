#!/usr/bin/env python3
"""Isolated enhancement worker: {in: input.npy, out: output.npy, sr: 24000}."""

import argparse
import json
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("resemble", "audiosr"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr
    try:
        import numpy as np
        import torch
        # These published models carry more than tensors in their checkpoints.
        original_load = torch.load
        def load_checkpoint(*positional, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*positional, **kwargs)
        torch.load = load_checkpoint
        if args.backend == "resemble":
            from resemble_enhance.enhancer.inference import enhance, load_enhancer
            load_enhancer(None, args.device)
        else:
            from audiosr import build_model, super_resolution
            model = build_model(model_name="speech", device=args.device)
        print(json.dumps({"status": "ready", "backend": args.backend}), file=protocol, flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("cmd") == "quit":
                    break
                audio = np.load(request["in"], allow_pickle=False).astype(np.float32)
                sr = int(request.get("sr", 24000))
                if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
                    raise ValueError("Expected nonempty finite mono audio")
                torch.manual_seed(args.seed)
                if args.backend == "resemble":
                    result, out_sr = enhance(torch.from_numpy(audio), sr, args.device,
                                              nfe=32, solver="midpoint", lambd=0.5, tau=0.5)
                else:
                    import soundfile as sf
                    with tempfile.TemporaryDirectory(prefix="audiosr-input-") as directory:
                        wav = str(Path(directory) / "input.wav")
                        sf.write(wav, audio, sr, subtype="FLOAT")
                        result = super_resolution(model, wav, seed=args.seed, ddim_steps=50, guidance_scale=3.5)
                    out_sr = 48000
                if hasattr(result, "detach"):
                    result = result.detach().cpu().numpy()
                result = np.asarray(result, dtype=np.float32).squeeze()
                expected = round(len(audio) * out_sr / sr)
                if result.ndim != 1 or len(result) < expected - out_sr * 0.02 or not np.isfinite(result).all():
                    raise ValueError("Enhancement returned invalid audio or duration")
                result = result[:expected]
                if len(result) < expected:
                    result = np.pad(result, (0, expected - len(result)))
                np.save(request["out"], result, allow_pickle=False)
                response = {"status": "done", "sr": int(out_sr), "samples": len(result)}
            except Exception as exc:
                response = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(response), file=protocol, flush=True)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=protocol, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
