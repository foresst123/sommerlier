import os
import subprocess
import time

class DiarizenWorkerService:
    """Manages the lifecycle of the isolated DiariZen worker process."""
    
    def __init__(self, python_env_path: str, worker_script_path: str, device_id: int = 1, logger=None):
        self.python_env_path = python_env_path
        self.worker_script_path = worker_script_path
        self.device_id = device_id
        self.process = None
        self.logger = logger
        
    def start(self):
        """Spawns the worker subprocess and waits for readiness."""
        if self.process is not None:
            return
            
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        
        if self.logger:
            self.logger.info(f"Starting DiariZen worker subprocess on CUDA_VISIBLE_DEVICES={self.device_id}")
            
        self.process = subprocess.Popen(
            [self.python_env_path, self.worker_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        
        # Wait for "READY" signal
        ready = False
        for _ in range(60): # timeout 60s
            line = self.process.stdout.readline()
            if not line:
                break
            if "ready" in line.lower():
                ready = True
                break
            
        if not ready:
            err = self.process.stderr.read()
            if self.logger:
                self.logger.error(f"DiariZen worker failed to start: {err}")
            self.stop()
            raise RuntimeError(f"DiariZen worker did not start. Err: {err}")
            
    def stop(self):
        """Terminates the worker process safely."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            if self.logger:
                self.logger.info("DiariZen worker terminated.")
