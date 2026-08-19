import requests
import json
import os
import soundfile as sf
import tempfile

# Qwen3-Omni's audio feature extractor runs at 16 kHz (preprocessor_config.json:
# sampling_rate 16000, n_fft 400, hop_length 160). The 24 kHz figure in the model
# card refers to its speech *output*, not what it listens to.
QWEN3_OMNI_SAMPLE_RATE = 16000


class Qwen3OmniCaptioner:
    """Wrapper for Qwen3-Omni caption API (HTTP localhost)."""

    def __init__(self, port: str = "11500"):
        self.api_url = f"http://localhost:{port}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}

    def caption(self, audio_array, sample_rate: int) -> str:
        """Send audio to Qwen3-Omni API and get caption text."""
        if sample_rate != QWEN3_OMNI_SAMPLE_RATE:
            import librosa
            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=QWEN3_OMNI_SAMPLE_RATE
            )
            sample_rate = QWEN3_OMNI_SAMPLE_RATE

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
            sf.write(temp_wav.name, audio_array, sample_rate)
            temp_wav.flush()
            
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {"url": f"file://{temp_wav.name}"}
                            }
                        ]
                    }
                ]
            }
            
            try:
                response = requests.post(self.api_url, headers=self.headers, json=data, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                pass
                
        return ""
