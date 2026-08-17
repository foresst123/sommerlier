import sys
import os
import tempfile
import shutil
import subprocess
import torch
import numpy as np
import librosa
from pathlib import Path

class DemucsRemover:
    """Wrapper for Demucs htdemucs to remove background music."""
    
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
    def separate_full(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """Run Demucs on the entire audio to obtain vocal stem."""
        temp_dir = tempfile.mkdtemp(prefix="demucs_full_")
        
        try:
            temp_input = os.path.join(temp_dir, "full.wav")
            import soundfile as sf
            sf.write(temp_input, audio_array, sample_rate)

            demucs_output_dir = os.path.join(temp_dir, "separated")

            cmd = [
                sys.executable, "-m", "demucs.separate",
                "-n", "htdemucs",
                "--two-stems", "vocals",
                "-d", self.device,
                "-o", demucs_output_dir,
                temp_input,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return None

            input_stem = Path(temp_input).stem
            vocal_path = os.path.join(demucs_output_dir, "htdemucs", input_stem, "vocals.wav")
            
            if not os.path.exists(vocal_path):
                return None

            vocal_waveform, _ = librosa.load(vocal_path, sr=sample_rate, mono=True)
            return vocal_waveform.astype(np.float32)

        except Exception as e:
            return None
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    def separate_segment(self, segment_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Run Demucs on a single segment."""
        # Equivalent fallback for single segment processing
        result = self.separate_full(segment_audio, sample_rate)
        if result is None:
            return segment_audio
        return result
