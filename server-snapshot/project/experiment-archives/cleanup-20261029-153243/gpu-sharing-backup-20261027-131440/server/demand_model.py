"""CPU-resident controller; each inference owns a short-lived GPU process."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

class GPUWaitTimeout(RuntimeError):pass

class DemandModel:
    def __init__(self,name,revision,wait_seconds=300,inference_seconds=240):
        self.name,self.revision=name,revision
        self.wait_seconds,self.inference_seconds=wait_seconds,inference_seconds
        self.state='idle';self.cancelled=threading.Event();self.last_peak_memory_mb=None

    def load(self):
        if not (Path(self.name)/'config.json').is_file():raise RuntimeError('Local model missing')

    def cancel(self):self.cancelled.set()

    def generate(self,image,metadata,previous_code='',error=''):
        from tools.launch_training import idle_gpus
        import fcntl
        self.cancelled.clear();self.state='waiting_gpu';process=None
        lock=(Path(__file__).resolve().parents[1]/'outputs/training.lock').open('a')
        try:
            deadline=time.monotonic()+self.wait_seconds
            while True:
                if self.cancelled.is_set():raise RuntimeError('Generation cancelled')
                if time.monotonic()>deadline:raise GPUWaitTimeout('No idle GPU within queue timeout')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    self.cancelled.wait(4);continue
                candidates,_=idle_gpus()
                if self.cancelled.wait(1):raise RuntimeError('Generation cancelled')
                current,_=idle_gpus()
                gpu=next((g for g in candidates if g in current),None)
                if gpu:break
                fcntl.flock(lock,fcntl.LOCK_UN)
                if time.monotonic()>=deadline:raise GPUWaitTimeout('No idle GPU within queue timeout')
                if self.cancelled.wait(4):raise RuntimeError('Generation cancelled')
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu[1]
            env['CAD_CONTROLLER_PID']=str(os.getpid())
            self.state='loading_and_generating'
            process=subprocess.Popen([sys.executable,'-m','server.inference_once'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,start_new_session=True)
            payload=json.dumps({'model':self.name,'revision':self.revision,'image':base64.b64encode(image).decode(),'metadata':metadata,'previous_code':previous_code,'error':error}).encode()
            deadline=time.monotonic()+self.inference_seconds;first=True
            while True:
                if self.cancelled.is_set():raise RuntimeError('Generation cancelled')
                if time.monotonic()>deadline:raise RuntimeError('Inference process timeout')
                try:
                    output,stderr=process.communicate(payload if first else None,timeout=1)
                    break
                except subprocess.TimeoutExpired:first=False
            if process.returncode:raise RuntimeError('Model worker failed; consult service diagnostics')
            response=json.loads(output);self.last_peak_memory_mb=response['peak_memory_mb']
            return response['text']
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.communicate(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.communicate()
            self.state='idle'
            lock.close()
