import json

class Qwen3ASRClient:
    """Interface to communicate with the standalone qwen3_worker process."""
    
    def __init__(self, process):
        self.process = process
        
    def transcribe(self, audio_path: str) -> str:
        """Send audio path to worker and get transcript."""
        if not self.process:
            return ""
            
        try:
            req = {"audio_path": audio_path}
            self.process.stdin.write(json.dumps(req) + "\n")
            self.process.stdin.flush()
            
            resp_line = self.process.stdout.readline()
            if not resp_line:
                return ""
                
            resp = json.loads(resp_line.strip())
            if resp.get("status") == "ok":
                return resp.get("text", "")
            else:
                return ""
        except Exception as e:
            return ""
