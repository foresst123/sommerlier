import os
import sys
import yaml
import torch
import numpy as np
import librosa
import glob

class SRCorrNetSeparator:
    """Wrapper for SR-CorrNet-L speech separation model."""
    
    def __init__(self, srcorrnet_path: str, device: torch.device):
        self.srcorrnet_path = srcorrnet_path
        self.device = device
        self.use_ss_inference = False
        
        original_sys_path = sys.path.copy()
        try:
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
                # 1. Try Old Structure (leolincoln repo)
                from models.SR_CorrNet_L_WSJ0.model import Model
                
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
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model = self.model.to(device)
                self.model.eval()

            except ModuleNotFoundError:
                # 2. Try New Structure (dmlguq456 repo) using SSInference
                for module_name, module_obj in cleared_modules.items():
                    sys.modules[module_name] = module_obj
                
                try:
                    from sr_corrnet import SSInference
                    config_path = os.path.join(srcorrnet_path, "sr_corrnet/models/SR_CorrNet_SS/configs/1ch_WSJ_fix_2spk_L_DM.yaml")
                    
                    # Find model.pt in the new structure
                    model_files = glob.glob(os.path.join(srcorrnet_path, "**", "model.pt"), recursive=True)
                    
                    if not model_files:
                        sommerlier_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        offline_weights_dir = os.path.join(sommerlier_dir, "offline_weights")
                        if os.path.exists(offline_weights_dir):
                            model_files = glob.glob(os.path.join(offline_weights_dir, "**", "model.pt"), recursive=True)
                            
                    if model_files:
                        checkpoint_path = model_files[0]
                    else:
                        # Fallback to HF Hub ID so SSInference downloads it automatically
                        checkpoint_path = "shinuh/sr-corrnet-ss-1ch-wsj-fix-2spk-l-dm"
                        
                    self.model = SSInference.from_pretrained(
                        config=config_path,
                        checkpoint_path=checkpoint_path,
                        device=str(device)
                    )
                    self.use_ss_inference = True
                    
                except Exception as e:
                    raise RuntimeError(f"Could not import SR-CorrNet! The repo structure at {srcorrnet_path} is unrecognized or missing weights.\nOriginal error: {e}")

        finally:
            sys.path = original_sys_path
            
    def separate(self, audio: np.ndarray, sample_rate: int) -> tuple:
        """Separate mixed audio into 2 speaker sources."""
        try:
            if self.use_ss_inference:
                # --- New repo logic (SSInference) ---
                if sample_rate != 8000:
                    audio_8k = librosa.resample(audio, orig_sr=sample_rate, target_sr=8000)
                else:
                    audio_8k = audio
                    
                waveform = torch.from_numpy(audio_8k).float()
                
                # Run inference via public API
                num_speakers = torch.tensor(2).to(self.device)
                result = self.model.process_waveform(waveform, n_spks=num_speakers)
                waveforms = result["waveforms"]
                
                # waveforms is a list of 1-D tensors
                src1 = waveforms[0].cpu().numpy()
                src2 = waveforms[1].cpu().numpy()
                
                # Resample back to original sample rate
                if sample_rate != 8000:
                    src1 = librosa.resample(src1, orig_sr=8000, target_sr=sample_rate)
                    src2 = librosa.resample(src2, orig_sr=8000, target_sr=sample_rate)
                    
                return src1, src2
            
            else:
                # --- Old repo logic (Raw Model) ---
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
            # Fallback on failure
            return audio, audio
