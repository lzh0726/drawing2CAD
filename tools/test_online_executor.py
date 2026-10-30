"""CPU API contract check with real isolation and an explicit model stub."""
import asyncio
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

async def main():
    import httpx
    from PIL import Image
    from server.app import Settings,create_app
    from server.bubblewrap_executor import BubblewrapExecutor
    token='TEST_ONLY_NOT_A_REAL_SECRET_123456'
    class Stub:
        code='import cadquery as cq\nresult=cq.Workplane("XY").box(2,3,4)'
        def load(self):pass
        def generate(self,*args):return self.code
    with tempfile.TemporaryDirectory() as tmp:
        model=Stub();app=create_app(Settings(storage=Path(tmp),token=token,repairs=0,executor_kind='bubblewrap'),model=model)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://testserver') as client:
                assert (await client.get('/health')).status_code==401
                headers={'Authorization':'Bearer '+token}
                assert (await client.get('/health',headers=headers)).status_code==200
                stream=io.BytesIO();Image.new('RGB',(64,64),'white').save(stream,format='PNG')
                response=await client.post('/v1/reconstruct',headers=headers,files={'drawing':('sample.png',stream.getvalue(),'image/png')},data={'metadata':'{}'})
                assert response.status_code==200,response.status_code
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    assert z.read('model.step').startswith(b'ISO-10303-21;')
                model.code='import os\nresult=1'
                response=await client.post('/v1/reconstruct',headers=headers,files={'drawing':('sample.png',stream.getvalue(),'image/png')},data={'metadata':'{}'})
                assert response.status_code==422,response.status_code
        executor=BubblewrapExecutor()
        assert await executor.available()
    report={'auth_rejection':True,'isolation_probe':True,'step_roundtrip':True,'unsafe_import_rejected':True,'model':'CPU stub; real VLM end-to-end pending'}
    (ROOT/'experiments/online-executor-test.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':asyncio.run(main())
