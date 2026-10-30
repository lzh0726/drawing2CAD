"""CPU-resident controller; each inference owns a short-lived GPU process."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from .gpu_sharing import snapshot, select_gpu, SHARED_LIMIT_MIB, MIN_RESERVE_MIB

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
        import fcntl
        self.cancelled.clear();self.state='waiting_gpu';process=None
        lock=(Path(__file__).resolve().parents[1]/'outputs/training.lock').open('a')
        try:
            deadline=time.monotonic()+self.wait_seconds
            history=[]
            while True:
                if self.cancelled.is_set():raise RuntimeError('Generation cancelled')
                if time.monotonic()>deadline:raise GPUWaitTimeout('No eligible GPU within queue timeout')
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    history.clear();self.cancelled.wait(4);continue
                try:
                    history.append((time.monotonic(),snapshot()));history=history[-6:]
                    gpu=select_gpu(history)
                except (ValueError,OSError,subprocess.SubprocessError):
                    history.clear();gpu=None
                if self.cancelled.is_set():raise RuntimeError('Generation cancelled')
                if time.monotonic()>deadline:raise GPUWaitTimeout('No eligible GPU within queue timeout')
                if gpu:break
                fcntl.flock(lock,fcntl.LOCK_UN)
                if time.monotonic()>=deadline:raise GPUWaitTimeout('No eligible GPU within queue timeout')
                if self.cancelled.wait(2):raise RuntimeError('Generation cancelled')
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid']
            env.pop('CAD_GPU_MEMORY_LIMIT_MIB',None)
            if gpu['shared']:env['CAD_GPU_MEMORY_LIMIT_MIB']=str(SHARED_LIMIT_MIB)
            env['CAD_CONTROLLER_PID']=str(os.getpid())
            self.state='loading_and_generating'
            process=subprocess.Popen([sys.executable,'-m','server.inference_once'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,start_new_session=True)
            payload=json.dumps({'model':self.name,'revision':self.revision,'image':base64.b64encode(image).decode(),'metadata':metadata,'previous_code':previous_code,'error':error}).encode()
            deadline=time.monotonic()+self.inference_seconds;first=True;next_check=time.monotonic()+5
            while True:
                if self.cancelled.is_set():raise RuntimeError('Generation cancelled')
                if time.monotonic()>deadline:raise RuntimeError('Inference process timeout')
                if gpu['shared'] and time.monotonic()>=next_check:
                    current=snapshot().get(gpu['uuid'])
                    if current is None or current['free']<MIN_RESERVE_MIB:
                        raise RuntimeError('Shared GPU memory pressure; inference stopped, retry later')
                    next_check=time.monotonic()+5
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
