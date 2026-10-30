import asyncio
from types import SimpleNamespace

import httpx
import pytest

from backend.app.worker import UserError, Worker


@pytest.mark.parametrize('code,fragment',[
    ('cad_generation_failed','CAD 检查'),
    ('needs_clarification','几何或尺寸信息'),
])
def test_remote_geometry_error_is_not_connection_error(tmp_path,monkeypatch,code,fragment):
    folder=tmp_path/'jobs'/'sample'/'input';folder.mkdir(parents=True)
    (folder/'preview.png').write_bytes(b'fixture-not-executed')
    original=httpx.AsyncClient
    transport=httpx.MockTransport(lambda request:httpx.Response(422,json={'detail':{'code':code}}))
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:original(transport=transport,**kwargs))
    worker=Worker(SimpleNamespace(storage=tmp_path,remote_url='http://fixture',remote_token='',remote_timeout=10),None)
    async def monitor(*args): await asyncio.sleep(60)
    monkeypatch.setattr(worker,'monitor_remote',monitor)
    job={'id':'sample','versions':[],'instructions':'','messages':[]}
    with pytest.raises(UserError,match=fragment):
        asyncio.run(worker.remote(job,tmp_path/'output'))
