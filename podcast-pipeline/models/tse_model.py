import os
import sys
import torch
import numpy as np
import librosa
from typing import Dict, List, Tuple, Union, Optional
import torch.nn.functional as F

# Below this RMS a probe region carries no speech, only the separator's noise
# floor. Measured: a correctly-empty track sat at 3e-5 while a track holding the
# speaker sat at 1.2e-1, four orders of magnitude apart.
ABS_SILENCE_RMS = 1e-3


class TargetSpeakerExtractor:
    """
    Dialogue Separation + ECAPA Speaker Matching.
    Uses SidonWorker for separation and ECAPA-TDNN natively for verification.
    """
    
    def __init__(self, device: torch.device, process=None, checkpoint_path: str = None):
        import tempfile

        self.device = device
        self.process = process
        self.classifier = None
        self.target_embed_cache: Dict[str, torch.Tensor] = {}
        self._temp_dir = tempfile.mkdtemp(prefix="tse_exchange_")
        self._req_counter = 0

        self._load_model()

    def close(self):
        """Remove the scratch directory used to exchange arrays with the worker."""
        import shutil
        shutil.rmtree(self._temp_dir, ignore_errors=True)
        
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
                # Normalize each clip before averaging: raw ECAPA embeddings have
                # length-dependent norms, so the longest clip would otherwise
                # dominate the centroid.
                enroll_embeddings.append(
                    F.normalize(self._get_embedding(e, sample_rate), p=2, dim=0)
                )
                
        if not enroll_embeddings:
            raise ValueError(f"No valid enrollment audios provided for target {target_id}")
            
        target_embed = torch.stack(enroll_embeddings).mean(dim=0)
        target_embed = F.normalize(target_embed, p=2, dim=0)
        
        if target_id:
            self.target_embed_cache[target_id] = target_embed
            
        return target_embed

    @staticmethod
    def _gather_probe(track: np.ndarray, spans, sr: int, floor_db: float = -40.0,
                      min_voiced_sec: float = 0.30, abs_floor_rms: float = ABS_SILENCE_RMS):
        """Concatenate `spans` of `track`, keeping only frames above an energy floor.

        Returns None when there is too little voiced audio to trust, which means
        "this speaker is not present here" -- a different outcome from
        "extraction failed", and the caller must not conflate the two.
        """
        if not spans:
            return None
        pieces = [track[max(0, a):min(len(track), b)] for a, b in spans]
        pieces = [pc for pc in pieces if pc.size]
        if not pieces:
            return None
        seg = np.concatenate(pieces)

        frame = max(1, int(0.02 * sr))
        if seg.size < frame * 2:
            return seg if seg.size else None
        n = seg.size // frame
        frames = seg[: n * frame].reshape(n, frame)
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)

        # An absolute floor first. The relative test below rescales by the
        # track's own 95th percentile, so a track that is silent here -- because
        # the separator correctly put this speaker on the *other* track -- still
        # passes every frame and yields an embedding built from noise. That
        # score then competes in the A/B assignment and can invert it.
        ref_abs = float(np.percentile(rms, 95))
        if ref_abs < abs_floor_rms:
            return None

        ref = ref_abs + 1e-12
        keep = 20.0 * np.log10(rms / ref) > floor_db
        if keep.sum() * frame < min_voiced_sec * sr:
            return None
        return frames[keep].reshape(-1)

    def separate_two_speakers(self, mixture_audio: np.ndarray, enroll_A: List[np.ndarray], enroll_B: List[np.ndarray], sample_rate: int = 16000, id_A: Optional[str] = None, id_B: Optional[str] = None, probe_A: Optional[List[Tuple[int, int]]] = None, probe_B: Optional[List[Tuple[int, int]]] = None, core_range: Optional[Tuple[int, int]] = None):
        """Run blind separation, then map the two output tracks onto A and B.

        probe_A / probe_B: (start, end) sample ranges within mixture_audio where
        that speaker is known to speak ALONE. Assignment and the returned
        similarities are measured there.

        Scoring on the overlap core instead (the previous eval_pad_start_sec /
        eval_core_len_sec path) is what drove sim_A from 0.46 to -0.07: ECAPA
        pools statistics over time and cannot form an embedding from ~0.2s.
        Solo regions are seconds long, so the same model works normally there.

        core_range: overlap core in mixture samples, used only by the "not-A"
        relative test for a speaker that has no solo region.

        Returns (track_A, track_B, sim_A, sim_B, diag). A sim is None when that
        speaker had too little voiced audio in its probe to judge; diag carries
        the numbers the caller needs for the not-A decision.
        """
        if not self.classifier:
            raise RuntimeError("ECAPA is not loaded.")
        if not self.process:
            raise RuntimeError("Sidon worker process is not connected.")
            
        if len(mixture_audio) == 0:
            raise ValueError("Input mixture_audio is empty.")

        import torchaudio.functional as F_audio
        import json

        self._req_counter += 1
        req_id = str(self._req_counter)

        # One reusable scratch dir instead of mkstemp per overlap; the exchange
        # runs thousands of times on a full podcast.
        temp_path = os.path.join(self._temp_dir, f"mix_{req_id}.npy")
        np.save(temp_path, np.ascontiguousarray(mixture_audio, dtype=np.float32))

        produced_paths = [temp_path]
        try:
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

            for key in ("track_1_path", "track_2_path"):
                if resp.get(key):
                    produced_paths.append(resp[key])

            if resp.get("error"):
                raise RuntimeError(f"Sidon worker error: {resp.get('error')}")

            # The separator decodes to its own native rate (DialogueSidon's VAE
            # decoder emits 24 kHz), which is independent of the pipeline's rate.
            # Trusting the rate the worker reports is what keeps ECAPA and the
            # resample-back honest.
            target_sr = int(resp.get("target_sr") or sample_rate)

            # Load results
            track_1_np = np.load(resp["track_1_path"])
            track_2_np = np.load(resp["track_2_path"])
        finally:
            # Always clean up, including on worker errors, so a long run does not
            # accumulate one leaked .npy per failed overlap.
            for path in produced_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

        track_1_tensor = torch.from_numpy(track_1_np).to(self.device)
        track_2_tensor = torch.from_numpy(track_2_np).to(self.device)

        # --- ECAPA matching, measured on the caller's probe spans ---
        embed_A = self._get_target_embedding(enroll_A, id_A, sample_rate)
        embed_B = self._get_target_embedding(enroll_B, id_B, sample_rate)

        # Probes arrive in mixture samples; the separator decodes at its own rate.
        scale = target_sr / float(sample_rate)
        def _rescale(spans):
            if not spans:
                return None
            return [(int(a * scale), int(b * scale)) for a, b in spans]

        span_A = _rescale(probe_A) or [(0, len(track_1_np))]
        span_B = _rescale(probe_B) or [(0, len(track_1_np))]

        def _score(track_np, spans, target_embed):
            probe = self._gather_probe(track_np, spans, target_sr)
            if probe is None:
                return None
            emb = F.normalize(self._get_embedding(probe, target_sr), p=2, dim=0)
            return float(torch.dot(target_embed, emb))

        s_1A, s_2A = _score(track_1_np, span_A, embed_A), _score(track_2_np, span_A, embed_A)
        s_1B, s_2B = _score(track_1_np, span_B, embed_B), _score(track_2_np, span_B, embed_B)

        def _n(x):
            # A missing score must not win the comparison by default.
            return -1.0 if x is None else x

        if (_n(s_1A) + _n(s_2B)) >= (_n(s_2A) + _n(s_1B)):
            out_A_tensor, out_B_tensor = track_1_tensor, track_2_tensor
            out_A_np, out_B_np = track_1_np, track_2_np
            sim_A, sim_B = s_1A, s_2B
        else:
            out_A_tensor, out_B_tensor = track_2_tensor, track_1_tensor
            out_A_np, out_B_np = track_2_np, track_1_np
            sim_A, sim_B = s_2A, s_1B

        # Diagnostics for the "not-A" test: when one speaker has no solo region,
        # asking "is this track B?" is unanswerable on a 0.2s core, but asking
        # "is this track a duplicate of A?" only needs a relative comparison,
        # which survives a bad absolute embedding.
        diag = {"anchor_self": None, "anchor_other": None, "other_rms": None}
        if core_range is not None:
            c0, c1 = int(core_range[0] * scale), int(core_range[1] * scale)
            c0, c1 = max(0, c0), min(len(out_A_np), c1)
            if c1 > c0:
                core_other = out_B_np[c0:c1]
                diag["other_rms"] = float(np.sqrt((core_other ** 2).mean() + 1e-12))
                anchor_embed = embed_A if sim_A is not None else embed_B
                for key, arr in (("anchor_self", out_A_np[c0:c1]), ("anchor_other", core_other)):
                    pr = self._gather_probe(arr, [(0, len(arr))], target_sr, min_voiced_sec=0.05)
                    if pr is not None:
                        e = F.normalize(self._get_embedding(pr, target_sr), p=2, dim=0)
                        diag[key] = float(torch.dot(anchor_embed, e))

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
            
        return restore_track(out_A_tensor), restore_track(out_B_tensor), sim_A, sim_B, diag