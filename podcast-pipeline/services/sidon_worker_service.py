import os
import subprocess
import json
from typing import Dict, Any

class SidonWorkerService:
    """Manages the lifecycle of the Sidon worker subprocess."""
    
    def __init__(self, config: Dict[str, Any], args: Any, logger=None):
        self.config = config
        self.args = args
        self.logger = logger
        self.process = None
        
    def start(self):
        """Start the isolated Sidon worker."""
        if not getattr(self.args, "tse", False):
            return
            
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check if Kaggle or other env has sidon_env
        sidon_env = os.path.join(base_dir, "..", "sidon_env")
        if not os.path.exists(sidon_env):
            # Try to fall back to a local venv or system python if sidon_env not found
            python_bin = "python3" 
        else:
            python_bin = os.path.join(sidon_env, "bin", "python")
            
        worker_script = os.path.join(base_dir, "sidon_worker.py")
        
        tse_path = os.environ.get("TSE_PATH", os.path.join(base_dir, "..", "tse_model"))
        checkpoint_path = os.path.join(tse_path, "weights", "dialogue_sidon.ckpt")
        
        if self.logger: self.logger.info(f"Starting Sidon worker on GPU {self.args.gpu_1}...")
        
        cmd = [
            python_bin, worker_script,
            "--device", f"cuda:{self.args.gpu_1}",
            "--model_path", checkpoint_path
        ]
        
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr to prevent it mixing with stdout json
            text=True,
            bufsize=1
        )
        
        # Wait for "ready"
        while True:
            line = self.process.stdout.readline()
            if not line:
                if self.logger: self.logger.error("Sidon worker crashed on startup.")
                stderr_out = self.process.stderr.read()
                if self.logger: self.logger.error(f"Sidon worker stderr: {stderr_out}")
                return
                
            line = line.strip()
            if not line:
                continue
                
            try:
                msg = json.loads(line)
                if msg.get("status") == "ready":
                    if self.logger: self.logger.info("Sidon worker is ready.")
                    break
                elif msg.get("status") == "error":
                    if self.logger: self.logger.error(f"Sidon worker error: {msg.get('message')}")
                    return
            except Exception:
                if self.logger: self.logger.warning(f"Sidon worker log: {line}")
                
    def stop(self):
        """Terminate the worker."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            if self.logger: self.logger.info("Sidon worker terminated.")
            self.process = None
