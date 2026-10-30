"""One inference, then process exit releases the CUDA context."""
import base64
import json
import sys
import os
import ctypes
import signal
from .model import TransformersModel

def main():
    if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0:raise RuntimeError('Parent-death signal unavailable')
    if os.getppid()!=int(os.environ['CAD_CONTROLLER_PID']):raise RuntimeError('Controller exited')
    request=json.load(sys.stdin)
    model=TransformersModel(request['model'],request['revision'])
    model.load();model.processor.image_processor.size={'longest_edge':262144,'shortest_edge':4096}
    text=model.generate(base64.b64decode(request['image']),request['metadata'],request['previous_code'],request['error'])
    print(json.dumps({'text':text,'peak_memory_mb':model.last_peak_memory_mb}))

if __name__=='__main__':main()
