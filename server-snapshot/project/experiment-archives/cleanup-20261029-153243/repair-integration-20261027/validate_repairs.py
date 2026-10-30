"""CPU-only manual geometry validation and fixture-driven Agent integration test."""
import asyncio
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile
import subprocess
import signal

os.environ['CUDA_VISIBLE_DEVICES'] = ''
STAGE = Path(__file__).resolve().parent
ROOT = Path('/home/luzhuohong/drawing2CAD')
sys.path.insert(0,str(STAGE))
from server import research_sandbox as sandbox
sandbox.ROOT = ROOT
original_command = sandbox.command
def command():
    cmd = original_command()
    cmd[cmd.index(str(ROOT/'tools/cad-runtime'))] = str(STAGE/'tools/cad-runtime')
    return cmd
sandbox.command = command
from server.app import Settings, create_app
from server.bubblewrap_executor import BubblewrapExecutor
from server.policy import extract_code, validate_code
from fastapi.testclient import TestClient

OUT = STAGE/'results';OUT.mkdir(exist_ok=True)
source=ROOT/'experiments/finetuned-pilot-v1-repairs0'
def hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
before=hashes()
assert all(sandbox.run({'probe':True},15).values())
rows={r['id']:r for r in map(json.loads,(ROOT/'datasets/pilot-v1/manifest.jsonl').read_text().splitlines())}
TOKEN='CPU_TEST_TOKEN_NOT_A_REAL_SECRET_1234'
class FixtureModel:
    resolved_revision='cpu-fixture-not-real-model'
    def __init__(self,bad,good): self.responses=iter([bad,good]);self.calls=[]
    def load(self): pass
    def generate(self,image,metadata,code,error):
        self.calls.append({'image_sha256':hashlib.sha256(image).hexdigest(),'metadata':metadata,'code':code,'error':error})
        return next(self.responses)

results=[]
for ident,line in [('test-00306373',7),('test-00790402',6),('test-00932166',5)]:
    folder=OUT/ident;folder.mkdir(exist_ok=True)
    if (folder/'validation.json').exists():
        results.append(json.loads((folder/'validation.json').read_text()));continue
    bad=extract_code((source/(ident+'-attempt1.txt')).read_text())
    good=(STAGE/'manual'/(ident+'-repaired.py')).read_text();validate_code(good)
    diag=sandbox.run({'code':bad,'diagnose':True},45)
    assert diag['status']=='invalid_geometry' and diag['line']==line,diag
    assert sandbox.run({'code':good,'diagnose':True},45)['status']=='passed'
    step,geometry=sandbox.run({'code':good},90)
    _,roundtrip=sandbox.run({'step_b64':base64.b64encode(step).decode()},45)
    truth=(ROOT/'datasets/pilot-v1'/rows[ident]['step']).read_bytes()
    print(ident,'geometry and roundtrip passed; comparing IoU',flush=True)
    payload={'prediction_b64':base64.b64encode(step).decode(),'truth_b64':base64.b64encode(truth).decode()}
    if ident in ('test-00790402','test-00932166'):
        metric={'status':'unavailable','error':'Boolean IoU exceeded isolation limits in prior attempts'}
    else:
        try:metric=sandbox.run(payload,90)
        except RuntimeError as e:metric={'status':'unavailable','error':str(e)}
    if 'iou_raw' in metric:
        assert metric['iou_raw']>.99,metric
    else:
        if ident=='test-00932166':payload['samples']=1024
        cmd=command();cmd[-1]='/runner/sample_compare.py'
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        try:sample_out,sample_err=proc.communicate(json.dumps(payload).encode(),timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.communicate();raise
        assert proc.returncode==0,sample_err.decode()
        metric['sampling_comparison']=json.loads(sample_out)
        assert metric['sampling_comparison']['estimated_iou']>.99,metric
    (folder/'manual-repaired.step').write_bytes(step)
    (folder/'manual-repaired.py').write_text(good)
    (folder/'manual-validation.json').write_text(json.dumps({'geometry':geometry,'step_roundtrip':roundtrip,'metrics':metric},indent=2))
    drawing=(ROOT/'datasets/pilot-v1'/rows[ident]['drawing']).read_bytes()
    model=FixtureModel(bad,good)
    with TestClient(create_app(Settings(storage=folder/'agent-traces',token=TOKEN,repairs=1),model,BubblewrapExecutor())) as client:
        response=client.post('/v1/reconstruct',headers={'Authorization':'Bearer '+TOKEN},files={'drawing':('drawing.png',drawing,'image/png')},data={'metadata':json.dumps({'unit':'mm','messages':[]})})
    assert response.status_code==200,response.status_code
    assert len(model.calls)==2 and 'Isolated geometry diagnosis' in model.calls[1]['error']
    assert model.calls[1]['code']==bad
    assert model.calls[0]['image_sha256']==model.calls[1]['image_sha256']
    assert model.calls[1]['metadata']=={'unit':'mm','instructions':'','messages':[],'previous_code':''}
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        assert json.loads(z.read('generation.json'))['repair_count']==1
    trace=json.loads(next((folder/'agent-traces').glob('*/trace.json')).read_text())
    assert trace['attempts'][0]['diagnostic']==diag
    record={'id':ident,'method':'reference-assisted manual reconstruction, not model repair','diagnostic':diag,'geometry':geometry,'step_roundtrip':roundtrip,'metrics':metric,'agent_test':'passed with fixture responses; no GPU/model inference'}
    results.append(record);(folder/'validation.json').write_text(json.dumps(record,indent=2))
    print(ident,'valid',geometry['valid'],'metrics',metric,'agent fixture passed',flush=True)

# Bounded failure: the Agent must stop after initial generation plus one repair.
model=FixtureModel(bad,bad)
with TestClient(create_app(Settings(storage=OUT/'bounded-failure',token=TOKEN,repairs=1),model,BubblewrapExecutor())) as client:
    response=client.post('/v1/reconstruct',headers={'Authorization':'Bearer '+TOKEN},files={'drawing':('drawing.png',drawing,'image/png')},data={'metadata':'{}'})
assert response.status_code==422 and len(model.calls)==2
assert not list((OUT/'bounded-failure').glob('*/model.step'))
assert before==hashes()
summary={'gpu_started':False,'original_evaluation_unchanged':True,'original_evaluation_hashes':before,'manual_repair_count':3,'automatic_model_repair_rate':'not measured','bounded_failure_test':'passed','results':results}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print('ALL CPU VALIDATIONS PASSED',flush=True)
