"""Offline research executor, using an explicitly scoped Bubblewrap filesystem."""
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def command():
    env=ROOT/'.venv-cad311'
    base=Path('/home/luzhuohong/miniconda3/envs/drawing2cad-train')
    return ['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv','--cap-drop','ALL',
        '--ro-bind','/usr','/usr','--symlink','usr/bin','/bin','--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64',
        '--ro-bind',str(env),str(env),'--ro-bind',str(base),str(base),
        '--ro-bind',str(ROOT/'tools/cad-runtime'),'/runner','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--tmpfs','/work',
        '--chdir','/work','--setenv','HOME','/work','--setenv','OMP_NUM_THREADS','1','--setenv','OPENBLAS_NUM_THREADS','1',
        '--setenv','LD_LIBRARY_PATH',str(base/'lib'),str(env/'bin/python'),'-I','/runner/research_worker.py']

def run(request, timeout=45):
    p=subprocess.Popen(command(),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try: output,error=p.communicate(json.dumps(request).encode(),timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);p.communicate();raise RuntimeError('CAD_timeout')
    if p.returncode: raise RuntimeError(error.decode(errors='replace')[-500:])
    if request.get('probe') or request.get('diagnose') or 'prediction_b64' in request: return json.loads(output)
    if len(output)>45*1024**2: raise RuntimeError('CAD_output_too_large')
    with zipfile.ZipFile(io.BytesIO(output)) as z:
        if z.getinfo('model.step').file_size>40*1024**2: raise RuntimeError('STEP_too_large')
        return z.read('model.step'),json.loads(z.read('geometry.json'))
