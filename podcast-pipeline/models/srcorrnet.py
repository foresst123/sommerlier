import os
import sys
import yaml
import torch
import numpy as np
import librosa

class SRCorrNetSeparator:
    """Wrapper for SR-CorrNet-L speech separation model."""
    
    def __init__(self, srcorrnet_path: str, device: torch.device):
        self.srcorrnet_path = srcorrnet_path
        self.device = device
        
        original_sys_path = sys.path.copy()
        try:
            original_models = sys.modules.get('models', None)
            original_utils = sys.modules.get('utils', None)

            podcast_pipeline_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            paths_to_remove = [p for p in sys.path if podcast_pipeline_path in p]
            for path in paths_to_remove:
                sys.path.remove(path)

            if srcorrnet_path not in sys.path:
                sys.path.insert(0, srcorrnet_path)

            modules_to_clear = [key for key in sys.modules.keys()
                              if key.startswith('models.') or key.startswith('utils.') or key in ['models', 'utils']]
            cleared_modules = {}
            for module_name in modules_to_clear:
                cleared_modules[module_name] = sys.modules[module_name]
                del sys.modules[module_name]

            try:
                from models.SR_CorrNet_L_WSJ0.model import Model
            except ModuleNotFoundError as e:
                raise RuntimeError(f"Could not import SR-CorrNet! Make sure the source code is cloned exactly at: {srcorrnet_path}\n(You can run: !git clone https://github.com/leolincoln/SR_CorrNet_SS.git {srcorrnet_path})\nOriginal error: {e}")
            for module_name, module_obj in cleared_modules.items():
                sys.modules[module_name] = module_obj

            config_path = os.path.join(srcorrnet_path, "models/SR_CorrNet_L_WSJ0/configs.yaml")
            with open(config_path, 'r') as f:
                yaml_dict = yaml.safe_load(f)
            self.config = yaml_dict["config"]

            self.model = Model(**self.config["model"])

            checkpoint_dir = os.path.join(srcorrnet_path, "models/SR_CorrNet_L_WSJ0/log/pretrain_weights")
            if not os.path.exists(checkpoint_dir) or not os.listdir(checkpoint_dir):
                checkpoint_dir = os.path.join(srcorrnet_path, "models/SR_CorrNet_L_WSJ0/log/scratch_weights")

            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pt', '.pth'))]
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(device)
            self.model.eval()
            
        finally:
            sys.path = original_sys_path
            
    def separate(self, audio: np.ndarray, sample_rate: int) -> tuple:
        """Separate mixed audio into 2 speaker sources."""
        try:
            if sample_rate != 8000:
                audio_8k = librosa.resample(audio, orig_sr=sample_rate, target_sr=8000)
            else:
                audio_8k = audio
                
            mixture_tensor = torch.tensor(audio_8k, dtype=torch.float32).unsqueeze(0)
            
            stride = self.config["model"]["module_audio_enc"]["stride"]
            remains = mixture_tensor.shape[-1] % stride
            if remains != 0:
                padding = stride - remains
                mixture_padded = torch.nn.functional.pad(mixture_tensor, (0, padding), "constant", 0)
            else:
                mixture_padded = mixture_tensor
                
            with torch.inference_mode():
                nnet_input = mixture_padded.to(self.device)
                estim_src, _ = self.model(nnet_input)
                
                src1 = estim_src[0][..., :mixture_tensor.shape[-1]].squeeze().cpu().numpy()
                src2 = estim_src[1][..., :mixture_tensor.shape[-1]].squeeze().cpu().numpy()
                
            if sample_rate != 8000:
                src1 = librosa.resample(src1, orig_sr=8000, target_sr=sample_rate)
                src2 = librosa.resample(src2, orig_sr=8000, target_sr=sample_rate)
                
            return src1, src2
            
        except Exception as e:
            return audio, audio
