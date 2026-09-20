import json
import threading

class Qwen3ASRClient:
    """Interface to communicate with the standalone qwen3_worker process."""
    
    def __init__(self, process):
        self.process = process
        self._lock = threading.Lock()
        
    def transcribe(self, audio_path: str) -> str:
        """Send audio path to worker and get transcript."""
        if not self.process:
            return ""
            
        try:
            req = {"audio_path": audio_path}
            with self._lock:
                self.process.stdin.write(json.dumps(req) + "\n")
                self.process.stdin.flush()
                resp_line = self.process.stdout.readline()
            if not resp_line:
                print("[Qwen3ASRClient] Empty response from worker", flush=True)
                return ""
                
            resp = json.loads(resp_line.strip())
            if "text" in resp:
                return resp["text"]
            else:
                print(f"[Qwen3ASRClient] Worker error: {resp.get('error', resp_line)}", flush=True)
                return ""
        except Exception as e:
            print(f"[Qwen3ASRClient] Exception in transcribe: {e}", flush=True)
            return ""

    def transcribe_batch(self, jobs, language="vi"):
        """Return texts in input order; IDs protect mapping across replicas."""
        if not jobs or not self.process:
            return [""] * len(jobs)
        request = {
            "cmd": "transcribe_batch",
            "language": language,
            "jobs": [{"id": str(job[0]), "audio_path": job[1]}
                     for job in jobs],
        }
        try:
            with self._lock:
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                response_line = self.process.stdout.readline()
            if not response_line:
                return [""] * len(jobs)
            response = json.loads(response_line)
            by_id = {str(item.get("id")): item.get("text", "")
                     for item in response.get("results", [])}
            return [by_id.get(str(job[0]), "") for job in jobs]
        except Exception as exc:
            print(f"[Qwen3ASRClient] batch exception: {exc}", flush=True)
            return [""] * len(jobs)
