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
                print("[Qwen3ASRClient] Empty response from worker", flush=True)
                return ""
                
            resp = json.loads(resp_line.strip())
            if resp.get("status") == "ok":
                return resp.get("text", "")
            else:
                print(f"[Qwen3ASRClient] Worker error: {resp.get('message', resp_line)}", flush=True)
                return ""
        except Exception as e:
            print(f"[Qwen3ASRClient] Exception in transcribe: {e}", flush=True)
            return ""
