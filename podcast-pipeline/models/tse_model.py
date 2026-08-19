import os
import sys
import torch
import numpy as np
import librosa
from typing import Dict, List, Union, Optional
import torch.nn.functional as F

class TargetSpeakerExtractor:
    """
    Dialogue Separation + ECAPA Speaker Matching.
    Uses SidonWorker for separation and ECAPA-TDNN natively for verification.
    """
    
    def __init__(self, device: torch.device, process=None, checkpoint_path: str = None):
        self.device = device
        self.process = process
        self.classifier = None
        self.target_embed_cache: Dict[str, torch.Tensor] = {}
        
        self._load_model()
        
    def _load_model(self):
        print(f"[TSE Model] Initializing ECAPA-TDNN on {self.device}...")
        
        # Load ECAPA-TDNN for Speaker Verification
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            tse_path = os.environ.get("TSE_PATH", os.path.join(os.path.dirname(__file__), "..", "tse_model"))
            cls_dir = os.path.join(tse_path, "ecapa")
            
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir=cls_dir,
                run_opts={"device": str(self.device)}
            )
            print("[TSE Model] ECAPA-TDNN loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load ECAPA-TDNN: {e}")
            
    def _get_embedding(self, audio_array: np.ndarray, sample_rate: int = 16000) -> torch.Tensor:
        """Helper to get speaker embedding from 1D numpy array."""
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
        tensor = torch.from_numpy(audio_array).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.classifier.encode_batch(tensor)
        return emb.squeeze()

    def _get_target_embedding(self, enrollment_audios: List[np.ndarray], target_id: str, sample_rate: int) -> torch.Tensor:
        """Calculate and cache the target embedding."""
        if target_id and target_id in self.target_embed_cache:
            return self.target_embed_cache[target_id]
            
        enroll_embeddings = []
        for e in enrollment_audios:
            if len(e) > 0:
                enroll_embeddings.append(self._get_embedding(e, sample_rate))
                
        if not enroll_embeddings:
            raise ValueError(f"No valid enrollment audios provided for target {target_id}")
            
        target_embed = torch.stack(enroll_embeddings).mean(dim=0)
        target_embed = F.normalize(target_embed, p=2, dim=0)
        
        if target_id:
            self.target_embed_cache[target_id] = target_embed
            
        return target_embed

    def separate_two_speakers(self, mixture_audio: np.ndarray, enroll_A: List[np.ndarray], enroll_B: List[np.ndarray], sample_rate: int = 16000, id_A: Optional[str] = None, id_B: Optional[str] = None):
        """
        Runs BSS once (via Worker), then uses ECAPA to map the 2 output tracks to Speaker A and B.
        Returns: (track_A, track_B, sim_A, sim_B)
        """
        if not self.classifier:
            raise RuntimeError("ECAPA is not loaded.")
        if not self.process:
            raise RuntimeError("Sidon worker process is not connected.")
            
        if len(mixture_audio) == 0:
            raise ValueError("Input mixture_audio is empty.")

        target_sr = 16000
        import torchaudio.functional as F_audio
        import tempfile
        import json
        import uuid
        
        req_id = str(uuid.uuid4())
        
        # Save mixture to temp npy
        fd, temp_path = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        
        np.save(temp_path, mixture_audio)
        
        # Call worker
        req = {
            "id": req_id,
            "audio_path": temp_path,
            "sample_rate": sample_rate
        }
        
        self.process.stdin.write(json.dumps(req) + "\n")
        self.process.stdin.flush()
        
        resp = None
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Sidon worker crashed or closed stdout")
            
            line_str = line.strip()
            if not line_str:
                continue
                
            try:
                parsed = json.loads(line_str)
                if parsed.get("id") == req_id:
                    resp = parsed
                    break
            except Exception:
                pass
                
        if resp.get("error"):
            raise RuntimeError(f"Sidon worker error: {resp.get('error')}")
            
        # Load results
        track_1_np = np.load(resp["track_1_path"])
        track_2_np = np.load(resp["track_2_path"])
        
        # Clean up temp files
        try:
            os.remove(temp_path)
            os.remove(resp["track_1_path"])
            os.remove(resp["track_2_path"])
        except Exception:
            pass
            
        track_1_tensor = torch.from_numpy(track_1_np).to(self.device)
        track_2_tensor = torch.from_numpy(track_2_np).to(self.device)

        # --- ECAPA-TDNN Matching ---
        embed_A = self._get_target_embedding(enroll_A, id_A, sample_rate)
        embed_B = self._get_target_embedding(enroll_B, id_B, sample_rate)
        
        emb_1 = F.normalize(self._get_embedding(track_1_tensor.cpu().numpy(), target_sr), p=2, dim=0)
        emb_2 = F.normalize(self._get_embedding(track_2_tensor.cpu().numpy(), target_sr), p=2, dim=0)
        
        # Calculate matching scores for all combinations
        score_1A = torch.dot(embed_A, emb_1).item()
        score_2A = torch.dot(embed_A, emb_2).item()
        score_1B = torch.dot(embed_B, emb_1).item()
        score_2B = torch.dot(embed_B, emb_2).item()
        
        # Determine assignment (Max Bipartite Matching for 2x2)
        if (score_1A + score_2B) > (score_2A + score_1B):
            out_A_tensor, out_B_tensor = track_1_tensor, track_2_tensor
            sim_A, sim_B = score_1A, score_2B
        else:
            out_A_tensor, out_B_tensor = track_2_tensor, track_1_tensor
            sim_A, sim_B = score_2A, score_1B
            
        # --- Resample back to original sample rate ---
        def restore_track(track_tensor_in):
            if sample_rate != target_sr:
                track_tensor_out = F_audio.resample(track_tensor_in.unsqueeze(0).cpu(), target_sr, sample_rate).squeeze(0)
            else:
                track_tensor_out = track_tensor_in.cpu()
                
            track_np = track_tensor_out.numpy()
            orig_len = len(mixture_audio)
            
            if len(track_np) > orig_len:
                track_np = track_np[:orig_len]
            elif len(track_np) < orig_len:
                track_np = np.pad(track_np, (0, orig_len - len(track_np)))
            return track_np
            
        return restore_track(out_A_tensor), restore_track(out_B_tensor), sim_A, sim_B
