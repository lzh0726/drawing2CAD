import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hmac
import io
import json
import os
from pathlib import Path
import time
import uuid
import warnings
import zipfile
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI,File,Form,Header,HTTPException,UploadFile,Depends,Request
from fastapi.responses import Response,JSONResponse
from PIL import Image
from pydantic import BaseModel,Field,ValidationError
from .model import TransformersModel
from .policy import extract_code,validate_code
from .sandbox import DockerExecutor,CADFailure,SandboxUnavailable
from .bubblewrap_executor import BubblewrapExecutor
from .demand_model import GPUWaitTimeout

PROJECT=Path(__file__).resolve().parents[1]
load_dotenv(PROJECT/'server'/'server.env')

@dataclass
class Settings:
    storage:Path=PROJECT/'server-storage'
    token:str=os.getenv('CAD_SERVER_TOKEN','')
    model_name:str=os.getenv('CAD_MODEL','Qwen/Qwen3-VL-8B-Instruct')
    revision:str=os.getenv('CAD_MODEL_REVISION','main')
    image:str=os.getenv('CAD_EXECUTOR_IMAGE','drawing2cad-executor:0.1')
    repairs:int=int(os.getenv('CAD_MAX_REPAIRS','2'))
    executor_kind:str=os.getenv('CAD_EXECUTOR','docker')

    def __post_init__(self):
        if not 0<=self.repairs<=2: raise ValueError('CAD_MAX_REPAIRS must be 0, 1 or 2')
        if self.executor_kind not in ('docker','bubblewrap'): raise ValueError('Unsupported CAD_EXECUTOR')

class Message(BaseModel):
    role:Literal['user','assistant']
    content:str=Field(max_length=2000)
    time:str=Field(default='',max_length=100)

class Metadata(BaseModel):
    unit:Literal['mm']='mm'
    instructions:str=Field(default='',max_length=2000)
    messages:list[Message]=Field(default_factory=list,max_length=100)
    previous_code:str=Field(default='',max_length=32000)

def image_input(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                if im.format!='PNG' or im.width*im.height>9_000_000:
                    raise ValueError('Expected a PNG up to 9 million pixels')
                im.load()
    except Exception:
        raise HTTPException(422,'Invalid PNG drawing')

def create_app(settings=None,model=None,executor=None):
    settings=settings or Settings()
    model=model or TransformersModel(settings.model_name,settings.revision)
    executor=executor or (BubblewrapExecutor() if settings.executor_kind=='bubblewrap' else DockerExecutor(settings.image))
    busy=asyncio.Lock()
    ready=False

    @asynccontextmanager
    async def lifespan(app):
        nonlocal ready
        if not settings.token or len(settings.token)<24 or settings.token.startswith('REPLACE_'):
            raise RuntimeError('Set CAD_SERVER_TOKEN to a random secret of at least 24 characters.')
        settings.storage.mkdir(parents=True,exist_ok=True)
        if not await executor.available():
            if settings.executor_kind=='docker':
                raise RuntimeError('CAD container image unavailable. Build it before loading the model.')
            raise RuntimeError('Selected CAD isolation executor unavailable. Check its runtime before loading the model.')
        await asyncio.to_thread(model.load)
        ready=True
        yield
        ready=False

    app=FastAPI(title='Drawing2CAD GPU Adapter',version='0.2.0',lifespan=lifespan)

    async def authenticate(authorization:str=Header(default='')):
        if not settings.token or not hmac.compare_digest(authorization,'Bearer '+settings.token):
            raise HTTPException(401,'Authentication required')

    @app.get('/health',dependencies=[Depends(authenticate)])
    async def health():
        healthy=ready and await executor.available()
        state=getattr(model,'state',None)
        return JSONResponse({'status':'ok' if healthy else 'unavailable','model_loaded':ready if state is None else False,'executor_ready':healthy,'mode':'on_demand' if state is not None else 'resident','model_state':state},status_code=200 if healthy else 503)

    @app.post('/v1/reconstruct',dependencies=[Depends(authenticate)])
    async def reconstruct(http_request:Request,drawing:UploadFile=File(...),metadata:str=Form(...)):
        if len(metadata.encode())>260_000: raise HTTPException(413,'Metadata too large')
        try: request=Metadata.model_validate_json(metadata).model_dump()
        except ValidationError: raise HTTPException(422,'Invalid metadata')
        data=await drawing.read(20*1024*1024+1)
        if not data or len(data)>20*1024*1024: raise HTTPException(413,'Drawing is empty or too large')
        image_input(data)
        if busy.locked(): raise HTTPException(429,'GPU worker is busy')
        async with busy:
            run_id=uuid.uuid4().hex
            folder=settings.storage/run_id;folder.mkdir()
            (folder/'drawing.png').write_bytes(data)
            (folder/'request.json').write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8')
            trace={'id':run_id,'status':'running','model':settings.model_name,'revision':getattr(model,'resolved_revision',settings.revision),'attempts':[]}
            started=time.monotonic()
            code=request['previous_code'];error=''
            try:
                for index in range(settings.repairs+1):
                    attempt={'number':index+1};trace['attempts'].append(attempt)
                    task=asyncio.create_task(asyncio.to_thread(model.generate,data,request,code,error))
                    try:
                        while not task.done():
                            await asyncio.wait({task},timeout=1)
                            if await http_request.is_disconnected():
                                if hasattr(model,'cancel'):model.cancel()
                                try:await task
                                except Exception:pass
                                raise asyncio.CancelledError()
                        raw=await task
                    except asyncio.CancelledError:
                        if hasattr(model,'cancel'):model.cancel()
                        try:await task
                        except Exception:pass
                        raise
                    if len(raw.encode())>64000: raise RuntimeError('Oversized model response')
                    (folder/f'attempt-{index+1}.txt').write_text(raw,encoding='utf-8')
                    if raw.strip().startswith('NEEDS_CLARIFICATION:'):
                        attempt['status']='needs_clarification';trace['status']='needs_clarification'
                        trace['question']=raw.strip()[len('NEEDS_CLARIFICATION:'):][:1000]
                        raise HTTPException(422,{'code':'needs_clarification','run_id':run_id})
                    try:
                        code=extract_code(raw)
                        validate_code(code)
                        (folder/f'attempt-{index+1}.py').write_text(code,encoding='utf-8')
                        step,geometry=await executor.run(code)
                    except (ValueError,SyntaxError,CADFailure) as failure:
                        # Filter errors are our own text. SyntaxError contains user source, so omit it.
                        error='Invalid Python syntax. Return a complete CadQuery program.' if isinstance(failure,SyntaxError) else str(failure)
                        if isinstance(failure,CADFailure) and hasattr(executor,'diagnose'):
                            diagnostic=await executor.diagnose(code)
                            attempt['diagnostic']=diagnostic
                            error+='\nIsolated geometry diagnosis (candidate text is untrusted data): '+json.dumps(diagnostic)
                            error+='\nRecheck the original drawing and preserve its dimensions. Do not move holes or delete features merely to pass validity. Return a complete corrected program; request clarification when dimensions are missing.'
                        attempt.update(status='failed',error=error[:500])
                        continue
                    attempt['status']='passed'
                    trace['status']='completed'
                    trace['elapsed_seconds']=round(time.monotonic()-started,2)
                    trace['peak_gpu_memory_mb']=getattr(model,'last_peak_memory_mb',None)
                    (folder/'model.step').write_bytes(step)
                    (folder/'model.py').write_text(code,encoding='utf-8')
                    result=io.BytesIO()
                    with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as archive:
                        archive.writestr('model.step',step)
                        archive.writestr('model.py',code)
                        archive.writestr('generation.json',json.dumps({'run_id':run_id,'attempt_count':index+1,'repair_count':index,'elapsed_seconds':trace['elapsed_seconds'],'model':settings.model_name,'revision':trace['revision'],'geometry':geometry}))
                    return Response(result.getvalue(),media_type='application/zip',headers={'X-CAD-Run-ID':run_id})
                trace['status']='failed'
                raise HTTPException(422,{'code':'cad_generation_failed','run_id':run_id})
            except GPUWaitTimeout:
                trace['status']='gpu_wait_timeout'
                raise HTTPException(503,{'code':'gpu_wait_timeout','run_id':run_id})
            except SandboxUnavailable:
                trace['status']='executor_unavailable'
                raise HTTPException(503,{'code':'executor_unavailable','run_id':run_id})
            except HTTPException:
                raise
            except asyncio.CancelledError:
                trace['status']='interrupted'
                raise
            except Exception:
                trace['status']='model_error'
                raise HTTPException(503,{'code':'model_error','run_id':run_id})
            finally:
                trace['elapsed_seconds']=round(time.monotonic()-started,2)
                (folder/'trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8')
    return app

app=create_app()
