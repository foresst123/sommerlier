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
    (Not pure TSE. Uses DialogueSidon to separate 2 speakers, then ECAPA-TDNN to identify the target).
    """
    
    def __init__(self, device: torch.device, checkpoint_path: str = None):
        self.device = device
        self.sidon_model = None
        self.classifier = None
        self.target_embed_cache: Dict[str, torch.Tensor] = {}
        
        # Default checkpoint path if not provided
        if not checkpoint_path:
            tse_path = os.environ.get("TSE_PATH", os.path.join(os.path.dirname(__file__), "..", "tse_model"))
            self.checkpoint_path = os.path.join(tse_path, "weights", "dialogue_sidon.ckpt")
        else:
            self.checkpoint_path = checkpoint_path
            
        self._load_model()
        
    def _load_model(self):
        print(f"[TSE Model] Initializing Dialogue Separation + ECAPA-TDNN on {self.device}...")
        
        # 1. Load DialogueSidon
        try:
            from sidon.lightning import DialogueSidonDiffusionLightningModule
        except ImportError:
            raise ImportError(
                "DialogueSidon is not installed! "
                "Please run: git clone https://github.com/sarulab-speech/Sidon.git && pip install -e Sidon"
            )
            
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"DialogueSidon checkpoint not found at {self.checkpoint_path}. "
                "Please download the correct .ckpt weights."
            )
            
        try:
            self.sidon_model = DialogueSidonDiffusionLightningModule.load_from_checkpoint(
                self.checkpoint_path, 
                map_location=self.device
            )
            self.sidon_model.eval()
            print("[TSE Model] DialogueSidon loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load DialogueSidon checkpoint: {e}")

        # 2. Load ECAPA-TDNN for Speaker Verification
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
        Runs BSS once, then uses ECAPA to map the 2 output tracks to Speaker A and Speaker B.
        Returns: (track_A, track_B, sim_A, sim_B)
        """
        if not self.sidon_model or not self.classifier:
            raise RuntimeError("Models are not fully loaded. Cannot perform separation.")
            
        if len(mixture_audio) == 0:
            raise ValueError("Input mixture_audio is empty.")

        target_sr = 16000
        import torchaudio.functional as F_audio
        
        mix_tensor = torch.from_numpy(mixture_audio).float().unsqueeze(0).to(self.device)

        # --- 1-Pass BSS with DialogueSidon ---
        try:
            from sidon.infer import run_separation_chunked
        except ImportError:
            raise ImportError("Could not import sidon.infer.run_separation_chunked. Ensure Sidon is correctly installed.")

        with torch.inference_mode():
            # Sidon's run_separation_chunked handles resampling, chunking, latent prediction, permutation, and VAE decode.
            est_sources, out_sr = run_separation_chunked(
                model=self.sidon_model,
                wav=mix_tensor,
                sample_rate=sample_rate,
                num_steps=30,
                chunk_seconds=20.0,
                overlap_seconds=5.0
            )
            # est_sources shape is [2, T_total]
            if est_sources.ndim == 2 and est_sources.shape[0] == 2:
                track_1_16k = est_sources[0]
                track_2_16k = est_sources[1]
            else:
                raise ValueError(f"Unexpected DialogueSidon output shape: {est_sources.shape}")

        # --- ECAPA-TDNN Matching ---
        embed_A = self._get_target_embedding(enroll_A, id_A, sample_rate)
        embed_B = self._get_target_embedding(enroll_B, id_B, sample_rate)
        
        emb_1 = F.normalize(self._get_embedding(track_1_16k.cpu().numpy(), target_sr), p=2, dim=0)
        emb_2 = F.normalize(self._get_embedding(track_2_16k.cpu().numpy(), target_sr), p=2, dim=0)
        
        # Calculate matching scores for all combinations
        score_1A = torch.dot(embed_A, emb_1).item()
        score_2A = torch.dot(embed_A, emb_2).item()
        score_1B = torch.dot(embed_B, emb_1).item()
        score_2B = torch.dot(embed_B, emb_2).item()
        
        # Determine assignment (Max Bipartite Matching for 2x2)
        if (score_1A + score_2B) > (score_2A + score_1B):
            out_A_16k, out_B_16k = track_1_16k, track_2_16k
            sim_A, sim_B = score_1A, score_2B
        else:
            out_A_16k, out_B_16k = track_2_16k, track_1_16k
            sim_A, sim_B = score_2A, score_1B
            
        # --- Resample back to original sample rate ---
        def restore_track(track_tensor_16k):
            if sample_rate != target_sr:
                track_tensor = F_audio.resample(track_tensor_16k.unsqueeze(0).cpu(), target_sr, sample_rate).squeeze(0)
            else:
                track_tensor = track_tensor_16k.cpu()
                
            track_np = track_tensor.numpy()
            orig_len = len(mixture_audio)
            
            if len(track_np) > orig_len:
                track_np = track_np[:orig_len]
            elif len(track_np) < orig_len:
                track_np = np.pad(track_np, (0, orig_len - len(track_np)))
            return track_np
            
        return restore_track(out_A_16k), restore_track(out_B_16k), sim_A, sim_B
