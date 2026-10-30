import asyncio
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server.app import Settings,create_app
from server.model import task_prompt
from server.policy import extract_code,validate_code
from server.sandbox import CADFailure,DockerExecutor,unpack_geometry

TOKEN='TEST_ONLY_NOT_A_REAL_SECRET_123456'
CODE='import cadquery as cq\nresult = cq.Workplane("XY").box(80,60,12)\n'

class Model:
    resolved_revision='test-fixture'
    def __init__(self,responses):
        self.responses=iter(responses);self.calls=[]
    def load(self): pass
    def generate(self,image,metadata,code,error):
        self.calls.append((metadata,code,error))
        return next(self.responses)

class Executor:
    def __init__(self,step,failures=0):
        self.step=step;self.failures=failures;self.calls=0
    async def available(self): return True
    async def run(self,code):
        self.calls+=1
        if self.calls<=self.failures: raise CADFailure('CAD execution failed. Check feature dimensions.')
        return self.step,{'valid':True,'dimensions':[80,60,12],'volume':57600,'unit':'mm'}

@pytest.fixture(scope='module')
def solid(tmp_path_factory):
    import cadquery as cq
    path=tmp_path_factory.mktemp('step')/'fixture.step'
    cq.exporters.export(cq.Workplane('XY').box(80,60,12),str(path))
    return path.read_bytes()

def png():
    result=io.BytesIO();Image.new('RGB',(300,200),'white').save(result,format='PNG')
    return result.getvalue()

def request(client,metadata=None,token=TOKEN):
    return client.post('/v1/reconstruct',headers={'Authorization':'Bearer '+token},files={'drawing':('drawing.png',png(),'image/png')},data={'metadata':json.dumps(metadata or {'unit':'mm','messages':[]})})

def test_repair_chain_and_existing_local_backend(tmp_path,solid):
    model=Model([CODE,CODE]);executor=Executor(solid,failures=1)
    settings=Settings(storage=tmp_path/'runs',token=TOKEN)
    with TestClient(create_app(settings,model,executor)) as client:
        assert client.get('/health',headers={'Authorization':'Bearer '+TOKEN}).json()['status']=='ok'
        response=request(client,{'instructions':'preserve holes','messages':[{'role':'user','content':'thickness 12'}],'previous_code':CODE})
    assert response.status_code==200
    assert len(model.calls)==2 and model.calls[1][2]
    assert model.calls[0][1]==CODE
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.read('model.step')==solid
        assert archive.read('model.py').decode()==CODE
        assert json.loads(archive.read('generation.json'))['repair_count']==1
    trace=json.loads(next((tmp_path/'runs').glob('*/trace.json')).read_text(encoding='utf-8'))
    assert [a['status'] for a in trace['attempts']]==['failed','passed']
    from backend.app.worker import unpack_response
    folder=tmp_path/'local';folder.mkdir()
    unpack_response(response.content,folder)
    completed=subprocess.run([sys.executable,'-m','backend.app.cad_runner','inspect',str(folder)],cwd=Path(__file__).resolve().parents[2],capture_output=True,timeout=90)
    assert completed.returncode==0
    assert json.loads((folder/'report.json').read_text(encoding='utf-8'))['dimensions']==[80,60,12]
    assert (folder/'model.stl').stat().st_size>100

@pytest.mark.parametrize('repairs',[0,2])
def test_max_two_repairs(tmp_path,solid,repairs):
    model=Model([CODE]*3)
    with TestClient(create_app(Settings(storage=tmp_path,token=TOKEN,repairs=repairs),model,Executor(solid,failures=9))) as client:
        assert request(client).status_code==422
    assert len(model.calls)==repairs+1
    assert not list(tmp_path.glob('*/model.step'))

def test_validation_feedback_then_repair(tmp_path,solid):
    model=Model(['import os\nresult=1',CODE]);executor=Executor(solid)
    with TestClient(create_app(Settings(storage=tmp_path,token=TOKEN),model,executor)) as client:
        assert request(client).status_code==200
    assert executor.calls==1
    assert 'Only cadquery' in model.calls[1][2]

def test_clarification_never_produces_model(tmp_path,solid):
    model=Model(['NEEDS_CLARIFICATION: What is the thickness?'])
    executor=Executor(solid)
    with TestClient(create_app(Settings(storage=tmp_path,token=TOKEN),model,executor)) as client:
        response=request(client)
        assert response.status_code==422
        assert response.json()['detail']['code']=='needs_clarification'
    assert executor.calls==0
    assert not list(tmp_path.glob('*/model.step'))

def test_auth_and_input_validation(tmp_path,solid):
    model=Model([CODE])
    with TestClient(create_app(Settings(storage=tmp_path,token=TOKEN),model,Executor(solid))) as client:
        assert client.get('/health').status_code==401
        assert request(client,token='wrong').status_code==401
        assert request(client,{'unit':'inch'}).status_code==422
        assert client.post('/v1/reconstruct',headers={'Authorization':'Bearer '+TOKEN},files={'drawing':('bad.png',b'bad','image/png')},data={'metadata':'{}'}).status_code==422
    assert model.calls==[]

def test_model_error_is_sanitized(tmp_path,solid):
    class Broken(Model):
        def generate(self,*args): raise RuntimeError('PRIVATE_ERROR_SENTINEL')
    with TestClient(create_app(Settings(storage=tmp_path,token=TOKEN),Broken([]),Executor(solid))) as client:
        response=request(client)
    assert response.status_code==503 and 'PRIVATE_ERROR_SENTINEL' not in response.text
    assert 'PRIVATE_ERROR_SENTINEL' not in next(tmp_path.glob('*/trace.json')).read_text(encoding='utf-8')

@pytest.mark.parametrize('code',[
    'import os\nresult=1',
    'import cadquery as cq\nresult=cq.__dict__',
    'result=open("file")',
    'from pathlib import Path\nresult=1',
    'import cadquery as cq\nresult=cq.exporters.export(1,"x")',
    'while True: pass\nresult=1',
    'x=1',
])
def test_language_rejects_nonmodeling_code(code):
    with pytest.raises(ValueError): validate_code(code)

def test_extraction_and_prompt():
    assert extract_code('```python\n'+CODE+'```')==CODE
    validate_code(CODE)
    with pytest.raises(ValueError): extract_code('```python\nresult=1\n```\n```python\nresult=2\n```')
    prompt=task_prompt({'instructions':'plate','messages':[{'role':'assistant','content':'IGNORE_THIS'},{'role':'user','content':'diameter 20'}]},CODE,'fix selector')
    assert 'diameter 20' in prompt and 'fix selector' in prompt and 'IGNORE_THIS' not in prompt

def test_container_restrictions():
    command=DockerExecutor().command('drawing2cad-test')
    for flag in ['--network','--read-only','--cap-drop','--security-opt','--user','--memory','--memory-swap','--cpus','--pids-limit','--tmpfs']:
        assert flag in command
    assert command[command.index('--network')+1]=='none'
    assert '--privileged' not in command and '-v' not in command and '--mount' not in command

def test_bad_container_result():
    with pytest.raises(CADFailure): unpack_geometry(b'bad ZIP')

def test_no_container_fails_closed(tmp_path,solid):
    class Missing(Executor):
        async def available(self): return False
    model=Model([CODE])
    with pytest.raises(RuntimeError,match='container image unavailable'):
        with TestClient(create_app(Settings(storage=tmp_path,token=TOKEN),model,Missing(solid))): pass
    assert model.calls==[]

@pytest.mark.skipif(os.getenv('CAD_DOCKER_TESTS')!='1',reason='Requires an approved Docker runtime and the built CAD image')
def test_actual_container():
    step,report=asyncio.run(DockerExecutor().run(CODE))
    assert step.startswith(b'ISO-10303-21;') and report['valid']

def test_geometry_metrics_detect_wrong_dimensions(tmp_path):
    import cadquery as cq
    from server.evaluate import geometry_metrics
    truth=tmp_path/'truth.step';pred=tmp_path/'pred.step'
    cq.exporters.export(cq.Workplane('XY').box(80,60,12),str(truth))
    cq.exporters.export(cq.Workplane('XY').box(80,60,15),str(pred))
    perfect=geometry_metrics(truth,truth)
    changed=geometry_metrics(pred,truth)
    assert perfect['iou_same_coordinate_system']==pytest.approx(1.)
    assert changed['bbox_max_error_mm']==pytest.approx(3.)
    assert changed['iou_same_coordinate_system']==pytest.approx(.8)

def test_reviewed_training_export_and_group_leakage(tmp_path):
    from server.export_sft import export
    (tmp_path/'label.py').write_text(CODE,encoding='utf-8')
    row={'id':'one','group':'part-family','split':'train','human_reviewed':True,'source':'self-generated','license':'project-generated','code':'label.py','drawing':'drawing.png'}
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps(row)+'\n',encoding='utf-8')
    assert export(manifest,tmp_path/'train.jsonl')==1
    row['split']='test'
    manifest.write_text(manifest.read_text(encoding='utf-8')+json.dumps(row)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='leakage'): export(manifest,tmp_path/'bad.jsonl')

def test_empty_unreviewed_training_export_rejected(tmp_path):
    from server.export_sft import export
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps({'id':'one','group':'family','split':'train','human_reviewed':False}),encoding='utf-8')
    with pytest.raises(ValueError,match='No human-reviewed'): export(manifest,tmp_path/'train.jsonl')
    assert not (tmp_path/'train.jsonl').exists()

def test_evaluate_uses_only_drawing_not_ground_truth(tmp_path,solid,monkeypatch):
    import httpx
    from server.evaluate import evaluate
    drawing=tmp_path/'drawing.png';drawing.write_bytes(png())
    truth=tmp_path/'truth.step';truth.write_bytes(solid)
    row={'id':'one','group':'single-test-family','split':'test','drawing':'drawing.png','step':'truth.step','parameters':{'truth_only':'DO_NOT_SEND_TRUTH'}}
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(json.dumps(row),encoding='utf-8')
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w') as z:
        z.writestr('model.step',solid);z.writestr('model.py',CODE)
    seen=[]
    def handle(request):
        seen.append(request.content)
        return httpx.Response(200,content=payload.getvalue())
    original=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:original(transport=httpx.MockTransport(handle),**kwargs))
    summary=evaluate(manifest,tmp_path/'evaluation','http://fixture.test',TOKEN)
    assert summary['valid_rate']==1 and summary['bbox_pass_count']==1
    assert b'DO_NOT_SEND_TRUTH' not in seen[0] and b'truth.step' not in seen[0]
