import torch
from pyannote.audio import Pipeline

class PyannoteDiarizer:
    """Wrapper for Pyannote speaker diarization."""
    
    def __init__(self, token: str, device: torch.device, use_community: bool = True, kwargs: dict = None):
        model_name = "pyannote/speaker-diarization-community-1" if use_community else "pyannote/speaker-diarization"
        self.pipeline = Pipeline.from_pretrained(model_name, token=token)
        self.pipeline.to(device)
        
        # Áp dụng tham số tối ưu cho Pyannote Community-1 (hoặc model thường)
        default_params = {
            'clustering': {
                'method': 'centroid',
                'min_cluster_size': 12,
                'threshold': 0.7045654963945799
            },
            'segmentation': {
                'min_duration_off': 0.0
            }
        }
        
        if kwargs:
            # Ghi đè nếu có truyền vào
            for k, v in kwargs.items():
                if k in default_params and isinstance(v, dict):
                    default_params[k].update(v)
                else:
                    default_params[k] = v
                    
        self.pipeline.instantiate(default_params)
            
    def diarize(self, audio_path: str, max_speakers: int = None, num_speakers: int = None):
        """Run diarization on audio file."""
        if num_speakers is not None:
            return self.pipeline(audio_path, num_speakers=num_speakers)
        if max_speakers is not None:
            return self.pipeline(audio_path, max_speakers=max_speakers)
        return self.pipeline(audio_path)
