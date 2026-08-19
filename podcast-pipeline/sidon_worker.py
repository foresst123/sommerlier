import sys
import json
# pyrefly: ignore [missing-import]
import torch
import numpy as np
import traceback
import argparse
import torchaudio.functional as F_audio

# PyTorch 2.6+ defaults to weights_only=True, which can break loading checkpoints containing TorchVersion
if hasattr(torch, "torch_version") and hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

def serve():
    parser = argparse.ArgumentParser()
    # Weights come from the DialogueSidon HF repo via sidon_infer.load_models;
    # accepted for backwards compatibility but unused.
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Fix SpeechBrain 1.1.0 logger bug where it tries to setLevel("20")
    import logging
    logging.addLevelName(10, "10")
    logging.addLevelName(20, "20")
    logging.addLevelName(30, "30")
    logging.addLevelName(40, "40")
    logging.addLevelName(50, "50")
    
    # Load model
    try:
        from sidon_infer import run_separation_chunked, load_models
    except ImportError as e:
        print(json.dumps({"status": "error", "message": f"Sidon not installed or import failed: {str(e)}"}), flush=True)
        sys.exit(1)

    try:
        # Pre-load models (downloads .pt2 from HF Hub if not cached)
        load_models(device)
        print(json.dumps({"status": "ready"}), flush=True)
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}), flush=True)
        sys.exit(1)

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue

            req = json.loads(line)
            req_id = req.get("id", "unknown")
            audio_path = req.get("audio_path")
            sample_rate = req.get("sample_rate", 16000)

            if not audio_path:
                resp = {"id": req_id, "error": "Missing audio_path"}
                print(json.dumps(resp), flush=True)
                continue

            try:
                # Load audio from numpy array
                audio_np = np.load(audio_path)
                mix_tensor = torch.from_numpy(audio_np).float().unsqueeze(0).to(device)

                with torch.inference_mode():
                    est_sources, out_sr = run_separation_chunked(
                        wav=mix_tensor,
                        sample_rate=sample_rate,
                        num_steps=30,
                        device=device
                    )
                
                # est_sources shape: (2, T)
                if est_sources.ndim == 2 and est_sources.shape[0] == 2:
                    track_1 = est_sources[0].cpu().numpy()
                    track_2 = est_sources[1].cpu().numpy()
                    
                    # Print original peak amplitude for debugging ECAPA issues
                    raw_peak = max(np.abs(track_1).max(), np.abs(track_2).max())
                    print(f"[SidonWorker] Raw output peak amplitude = {raw_peak:.6f}", file=sys.stderr)
                    
                    # Joint normalization to preserve energy correlation between speakers
                    global_max = max(raw_peak, 1e-6)
                    track_1 = np.clip(track_1 / global_max * 0.9, -1.0, 1.0)
                    track_2 = np.clip(track_2 / global_max * 0.9, -1.0, 1.0)
                else:
                    raise ValueError(f"Unexpected DialogueSidon output shape: {est_sources.shape}")
                
                out_path_1 = audio_path.replace(".npy", "_t1.npy")
                out_path_2 = audio_path.replace(".npy", "_t2.npy")
                np.save(out_path_1, track_1)
                np.save(out_path_2, track_2)

                resp = {
                    "id": req_id,
                    "track_1_path": out_path_1,
                    "track_2_path": out_path_2,
                    "target_sr": int(out_sr)
                }
                print(json.dumps(resp), flush=True)

            except Exception as e:
                resp = {"id": req_id, "error": str(e)}
                print(json.dumps(resp), flush=True)

        except Exception as e:
            traceback.print_exc()

if __name__ == "__main__":
    serve()
