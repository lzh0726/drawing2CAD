"""Supplement the independent audit using the selected proper rotations."""
from pathlib import Path
import base64
import json
import os
import shutil
import signal
import subprocess
import sys

r=Path('/home/luzhuohong/drawing2CAD');sys.path.insert(0,str(r))
from server.research_sandbox import command
out=r/'experiments/a1-1k-r16/success-geometry-audit-20261027-190624'
runtime=out/'runtime';shutil.copy2(r/'tools/audit_rotations.py',runtime/'audit_rotations.py')
rows={v['id']:v for v in map(json.loads,(r/'datasets/pilot-v1/manifest.jsonl').read_text().splitlines())}
for ident in ['test-00070021','test-00878404','test-00231272','test-00753045','test-00344795']:
    previous=json.loads((out/(ident+'-rotation.json')).read_text())
    backup=out/(ident+'-rotation-initial.json')
    if not backup.exists():backup.write_text(json.dumps(previous,indent=2))
    cmd=command();cmd[cmd.index(str(r/'tools/cad-runtime'))]=str(runtime);cmd[-1]='/runner/audit_rotations.py'
    request={'prediction':base64.b64encode((r/'experiments/a1-1k-r16/entity-rate-current-20261027-141244'/(ident+'.step')).read_bytes()).decode(),
             'truth':base64.b64encode((r/'datasets/pilot-v1'/rows[ident]['step']).read_bytes()).decode(),
             'rotation_matrix':previous['best']['rotation_matrix']}
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    timeout=False
    try:raw,err=p.communicate(json.dumps(request).encode(),timeout=90)
    except subprocess.TimeoutExpired:
        timeout=True;os.killpg(p.pid,signal.SIGKILL);raw,err=p.communicate()
    try:record=json.loads(raw.decode().splitlines()[-1])
    except Exception:record={'error':'comparison_failed','returncode':p.returncode}
    record.update(id=ident,wall_timeout=timeout,returncode=p.returncode)
    (out/(ident+'-rotation.json')).write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)
