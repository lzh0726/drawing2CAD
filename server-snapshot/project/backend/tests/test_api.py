import asyncio
import io
import time
import zipfile

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.store import Store
from backend.app.worker import unpack_response, UserError


@pytest.fixture
def app(tmp_path):
    return create_app(Settings(storage=tmp_path,remote_url='',remote_token=''))


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def png():
    out=io.BytesIO()
    Image.new('RGB',(160,100),'white').save(out,format='PNG')
    return out.getvalue()


def upload(client,content=None,**data):
    return client.post('/api/jobs',files={'file':('part.png',content if content is not None else png(),'image/png')},data=data)


def wait_job(client,job_id):
    deadline=time.monotonic()+100
    while time.monotonic()<deadline:
        job=client.get(f'/api/jobs/{job_id}').json()
        if job['status'] not in ('queued','preparing','generating','checking'):
            assert job['status']=='completed',job.get('error')
            return job
        time.sleep(.1)
    pytest.fail('CAD job did not finish in 100 seconds')


def test_upload_persistence_and_no_fabricated_model(client,app):
    assert client.get('/api/health').json()['remote_configured'] is False
    j=upload(client,instructions='安装板').json()
    assert j['status']=='waiting_server'
    assert client.get(f"/api/jobs/{j['id']}/drawing").headers['content-type']=='image/png'
    assert client.post(f"/api/jobs/{j['id']}/run",json={}).json()['status']=='waiting_server'
    assert client.get(f"/api/jobs/{j['id']}/artifacts/step").status_code==404
    reopened=Store(app.state.store.path)
    assert reopened.get(j['id'])['instructions']=='安装板'
    assert len(reopened.events(j['id']))>=2


def test_invalid_uploads_and_origin(client):
    assert upload(client,b'not an image').status_code==422
    assert upload(client,b'').status_code==413
    assert upload(client,page='2').status_code==422
    assert upload(client,page='0').status_code==422
    assert client.post('/api/demo',json={},headers={'origin':'https://foreign.example'}).status_code==403
    assert client.get('/api/health',headers={'host':'foreign.example'}).status_code==400


def test_pdf_page_selection(client):
    import pypdfium2 as pdfium
    doc=pdfium.PdfDocument.new()
    doc.new_page(200,300).close()
    doc.new_page(400,300).close()
    out=io.BytesIO();doc.save(out);doc.close()
    j=upload(client,out.getvalue(),page='2').json()
    assert j['pages']==2 and j['page']==2
    rendered=Image.open(io.BytesIO(client.get(f"/api/jobs/{j['id']}/drawing").content))
    assert rendered.width>rendered.height
    assert upload(client,out.getvalue(),page='3').status_code==422


def test_real_cad_and_immutable_versions(client,tmp_path):
    import cadquery as cq
    j=client.post('/api/demo',json={}).json()
    job_id=j['id'];base=f'/api/jobs/{job_id}'
    first=wait_job(client,job_id)
    report=first['versions'][0]['report']
    assert report['dimensions']==[80.0,60.0,12.0]
    step=client.get(base+'/artifacts/step?version=1').content
    assert b'ISO-10303-21' in step
    path=tmp_path/'download.step';path.write_bytes(step)
    solid=cq.importers.importStep(str(path)).val()
    assert solid.isValid() and solid.Volume()>0
    assert len(client.get(base+'/artifacts/stl').content)>1000
    assert 'cadquery' in client.get(base+'/artifacts/code').text
    assert client.get(base+'/artifacts/report').json()['checks'][1]['status']=='passed'
    assert client.get(base+'/artifacts/arbitrary').status_code==404
    assert client.get(base+'/artifacts/step?version=99').status_code==404
    assert client.post(base+'/run',json={'parameters':{'length':20}}).status_code==422
    client.post(base+'/messages',json={'content':'厚度改为 15，孔径改为 24'})
    second=wait_job(client,job_id)
    assert len(second['versions'])==2
    assert second['versions'][-1]['report']['dimensions'][2]==15
    assert second['versions'][-1]['parameters']['hole']==24
    assert client.get(base+'/artifacts/step?version=1').content==step
    client.post(base+'/messages',json={'content':'孔径改为 999'})
    assert len(client.get(base).json()['versions'])==2


def test_cancel_and_conflict(client,app,monkeypatch):
    async def slow(*args):
        await asyncio.sleep(60)
    monkeypatch.setattr(app.state.worker,'cad',slow)
    job=client.post('/api/demo',json={}).json();base=f"/api/jobs/{job['id']}"
    assert client.post(base+'/run',json={}).status_code==409
    assert client.post(base+'/cancel').json()['status']=='cancelled'
    assert client.get(base).json()['versions']==[]


def test_restart_marks_interrupted(app):
    app.state.store.create({'id':'pending','status':'generating','messages':[]})
    assert Store(app.state.store.path).get('pending')['status']=='interrupted'


def test_zip_contract_does_not_extract_paths_or_execute(tmp_path):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        z.writestr('model.step','placeholder')
        z.writestr('model.py',"raise RuntimeError('must not run')")
        z.writestr('../escape.txt','bad')
    unpack_response(out.getvalue(),tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir())==['model.py','model.step']
    with pytest.raises(UserError):
        unpack_response(b'not zip',tmp_path)
    incomplete=io.BytesIO()
    with zipfile.ZipFile(incomplete,'w') as z:
        z.writestr('model.step','placeholder')
    with pytest.raises(UserError):
        unpack_response(incomplete.getvalue(),tmp_path)


def test_remote_contract_with_mock_transport(tmp_path,monkeypatch):
    # Verifies HTTP adapter and local STEP import, not GPU model inference.
    import cadquery as cq
    step=tmp_path/'fixture.step';cq.exporters.export(cq.Workplane('XY').box(30,20,10),str(step))
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('model.step',step.read_bytes())
        z.writestr('model.py',"raise RuntimeError('source is not executed locally')")
    requests=[]
    def handler(request):
        requests.append(request)
        return httpx.Response(200,content=archive.getvalue(),headers={'content-type':'application/zip'})
    original=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(handler),**kwargs))
    settings=Settings(storage=tmp_path/'remote',remote_url='http://adapter.example',remote_token='')
    with TestClient(create_app(settings)) as c:
        j=upload(c,instructions='fixture instruction').json()
        assert j['status']=='ready'
        c.post(f"/api/jobs/{j['id']}/run",json={})
        result=wait_job(c,j['id'])
        assert result['versions'][0]['report']['dimensions']==[30.0,20.0,10.0]
        assert str(requests[0].url)=='http://adapter.example/v1/reconstruct'
        assert b'name="drawing"' in requests[0].content and b'fixture instruction' in requests[0].content
        assert result['versions'][0]['report']['checks'][-1]['status']=='unknown'
