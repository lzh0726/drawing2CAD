"""Deploy CPU-tested sources; never load the model or expose credentials."""
from pathlib import Path
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request
from dotenv import dotenv_values

root=Path('/home/luzhuohong/drawing2CAD')
stage=root/'experiments/repair-integration-20261027'
summary=json.loads((stage/'results/summary.json').read_text())
assert summary['manual_repair_count']==3 and summary['bounded_failure_test']=='passed'
cfg_path=root/'server/server.env'
cfg=dotenv_values(cfg_path)
def health():
    request=urllib.request.Request('http://127.0.0.1:8001/health',headers={'Authorization':'Bearer '+cfg['CAD_SERVER_TOKEN']})
    return json.load(urllib.request.urlopen(request,timeout=10))
assert health()['model_state']=='idle','Active model request; defer deployment'
record=root/'experiments/online-launch.json';state=json.loads(record.read_text());pid=state['pid']
cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
assert b'server.online:create_online_app' in cmd and b'.venv-service/bin/python' in cmd
backup=root/'experiments'/('repair-source-backup-'+time.strftime('%Y%m%d-%H%M%S'));backup.mkdir()
files=['server/app.py','server/bubblewrap_executor.py','server/research_sandbox.py','tools/cad-runtime/research_worker.py','tools/cad-runtime/cad_diagnostics.py']
for name in files:
    src=stage/name;compile(src.read_text(),str(src),'exec')
for name in files:
    target=root/name
    if target.exists():
        save=backup/name;save.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,save)
    shutil.copy2(stage/name,target)
# Only change the nonsecret repair-count field; do not copy or print credentials.
s=cfg_path.read_text();s,n=re.subn(r'(?m)^CAD_MAX_REPAIRS=.*$','CAD_MAX_REPAIRS=1',s)
if not n:s+='\nCAD_MAX_REPAIRS=1\n'
cfg_path.write_text(s);cfg['CAD_MAX_REPAIRS']='1'
os.kill(pid,signal.SIGTERM)
for _ in range(50):
    if not Path(f'/proc/{pid}').exists():break
    time.sleep(.2)
else:raise RuntimeError('Old service has not stopped')
env=os.environ.copy();env.update({k:v for k,v in cfg.items() if v is not None})
with (root/'logs/online-sharing-service.log').open('ab') as log:
    proc=subprocess.Popen([str(root/'.venv-service/bin/python'),'-m','uvicorn','server.online:create_online_app','--factory','--host','127.0.0.1','--port','8001'],cwd=root,env=env,stdout=log,stderr=log,start_new_session=True)
state.update(pid=proc.pid,started_unix=time.time());record.write_text(json.dumps(state,indent=2))
for _ in range(30):
    try:
        status=health();assert status['model_state']=='idle';break
    except Exception:time.sleep(1)
else:raise RuntimeError('Gateway health failed')
evidence={'status':'deployed','repairs':1,'health':status,'source_backup':str(backup),'gpu_started':False}
(stage/'deployment.json').write_text(json.dumps(evidence,indent=2))
print(json.dumps(evidence))
