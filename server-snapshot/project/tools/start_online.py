"""Start a CPU-only gateway; GPU checks happen for each inference request."""
import json
import os
from pathlib import Path
import socket
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',8001))==0:raise RuntimeError('Port 8001 is already occupied')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['CAD_MODEL']=str(ROOT/'models/drawing2cad-v1')
    os.environ['CAD_EXECUTOR']='bubblewrap'
    os.environ['CAD_MAX_REPAIRS']='0'
    record={'pid':os.getpid(),'mode':'on_demand','port':8001,'host':'127.0.0.1','started_unix':time.time()}
    (ROOT/'experiments/online-launch.json').write_text(json.dumps(record,indent=2))
    os.chdir(ROOT)
    os.execv(str(ROOT/'.venv-service/bin/python'),[str(ROOT/'.venv-service/bin/python'),'-m','uvicorn','server.online:create_online_app','--factory','--host','127.0.0.1','--port','8001'])

if __name__=='__main__':main()
