import sys
import json
# pyrefly: ignore [missing-import]
import torch
import numpy as np
import traceback
import argparse
import torchaudio.functional as F_audio

def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Load model
    try:
        from sidon.lightning import DialogueSidonDiffusionLightningModule
        from sidon.infer import run_separation_chunked
    except ImportError:
        print(json.dumps({"status": "error", "message": "Sidon not installed in this environment"}), flush=True)
        sys.exit(1)

    try:
        model = DialogueSidonDiffusionLightningModule.load_from_checkpoint(args.model_path, map_location=device)
        model.eval()
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
                        model=model,
                        wav=mix_tensor,
                        sample_rate=sample_rate,
                        num_steps=30,
                        chunk_seconds=20.0,
                        overlap_seconds=5.0
                    )
                
                # est_sources shape: (2, T)
                if est_sources.ndim == 2 and est_sources.shape[0] == 2:
                    track_1 = est_sources[0].cpu().numpy()
                    track_2 = est_sources[1].cpu().numpy()
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
                    "target_sr": 16000
                }
                print(json.dumps(resp), flush=True)

            except Exception as e:
                resp = {"id": req_id, "error": str(e)}
                print(json.dumps(resp), flush=True)

        except Exception as e:
            traceback.print_exc()

if __name__ == "__main__":
    serve()
