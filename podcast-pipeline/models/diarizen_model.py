import json
import tempfile
import os
import soundfile as sf

class DiariZenClient:
    """Interface to communicate with the standalone diarizen_worker process."""
    
    def __init__(self, process):
        self.process = process
        
    def diarize(self, audio_input: dict):
        """Send audio to worker and get segments."""
        if not self.process:
            return None
            
        waveform = audio_input["waveform"]
        sample_rate = audio_input["sample_rate"]
        
        # Write to a temporary file
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        try:
            # Waveform shape is (1, T) or (T,). soundfile expects (T,) or (T, 1)
            audio_data = waveform.squeeze().cpu().numpy()
            sf.write(temp_path, audio_data, sample_rate)
            
            req = {"audio_path": temp_path}
            self.process.stdin.write(json.dumps(req) + "\n")
            self.process.stdin.flush()
            
            resp = None
            for _ in range(10000): # large number to accommodate long audio progress events and noise
                resp_line = self.process.stdout.readline()
                if not resp_line:
                    break
                
                line_str = resp_line.strip()
                if not line_str:
                    continue
                
                try:
                    resp = json.loads(line_str)
                    if "progress" in resp:
                        # Use carriage return to overwrite the same line
                        print(f"\r[DiariZenClient] {resp['progress']}", end="", flush=True)
                        continue # Keep listening for the final segments response
                    
                    # Print a newline once we get the final response so subsequent logs don't overwrite the progress
                    print()
                    break # Successfully parsed final output
                except json.JSONDecodeError:
                    print(f"[DiariZenClient] Ignoring stdout noise: {line_str}", flush=True)
                    continue

            if resp is None:
                print("[DiariZenClient] Empty or invalid response from worker", flush=True)
                return None
                
            if "segments" in resp:
                import pandas as pd
                # Return something that has .itertracks(yield_label=True) mock
                class DummyTurn:
                    def __init__(self, start, end):
                        self.start = start
                        self.end = end
                        
                class DummyAnnotation:
                    def __init__(self, segments):
                        self.segments = segments
                        
                    def itertracks(self, yield_label=True):
                        for seg in self.segments:
                            yield DummyTurn(seg["start"], seg["end"]), None, seg["speaker"]
                            
                return DummyAnnotation(resp["segments"])
            else:
                print(f"[DiariZenClient] Worker error: {resp.get('error', resp_line)}", flush=True)
                return None
        except Exception as e:
            print(f"[DiariZenClient] Exception in diarize: {e}", flush=True)
            return None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Backwards compatibility alias for model_loader
class DiariZenDiarizer(DiariZenClient):
    pass
