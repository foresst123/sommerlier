import requests
import json
import os
import soundfile as sf
import tempfile

class Qwen3OmniCaptioner:
    """Wrapper for Qwen3-Omni caption API (HTTP localhost)."""
    
    def __init__(self, port: str = "11500"):
        self.api_url = f"http://localhost:{port}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
    
    def caption(self, audio_array, sample_rate: int) -> str:
        """Send audio to Qwen3-Omni API and get caption text."""
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
