import asyncio
from contextlib import asynccontextmanager
import io
import json
import re
import uuid
import warnings
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import ValidationError
from .config import Settings, PROJECT
from .schemas import MessageInput, Parameters, RunInput
from .store import Store, now
from .worker import Worker, ACTIVE


def create_app(settings=None):
    settings = settings or Settings()
    store = Store(settings.storage/'jobs.sqlite3')
    worker = Worker(settings,store)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await worker.close()

    app = FastAPI(title='Drawing2CAD Local API',version='0.1.0',lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
    @app.middleware('http')
    async def local_origin(request,call_next):
        origin=request.headers.get('origin')
        allowed={f'http://{host}:{port}' for host in ('127.0.0.1','localhost') for port in (8000,5173)}
        if request.method not in ('GET','HEAD','OPTIONS') and origin and origin not in allowed:
            return JSONResponse({'detail':'仅接受本地工作台请求'},status_code=403)
        return await call_next(request)
    app.state.store, app.state.worker = store, worker

    def get_job(job_id):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404,'任务不存在')
        return job

    def ensure_idle(job):
        if job['status'] in ACTIVE:
            raise HTTPException(409,'任务正在运行，请等待或先取消')

    def new_job(kind,title,instructions='',parameters=None):
        stamp = now()
        return {'id':uuid.uuid4().hex,'kind':kind,'title':title,'status':'ready','created_at':stamp,'updated_at':stamp,
                'instructions':instructions,'parameters':parameters,'versions':[],'messages':[],'error':None,'page':1,'pages':1}

    @app.get('/api/health')
    def health():
        return {'status':'ok','version':'0.1.0','remote_configured':bool(settings.remote_url),'mode':'local','cad_available':True}

    @app.get('/api/connection')
    async def connection():
        if not settings.remote_url:
            return {'configured':False,'reachable':False,'message':'未配置 GPU 建模服务器。图纸可上传保存，本地样例可独立运行。'}
        import httpx
        headers={'Authorization':f'Bearer {settings.remote_token}'} if settings.remote_token else {}
        try:
            async with httpx.AsyncClient(timeout=5,trust_env=False,follow_redirects=False) as client:
                r=await client.get(settings.remote_url+'/health',headers=headers)
                return {'configured':True,'reachable':r.status_code==200,'message':'服务器可访问。' if r.status_code==200 else f'服务器检查返回 HTTP {r.status_code}。'}
        except httpx.HTTPError:
            return {'configured':True,'reachable':False,'message':'无法连接服务器，请检查服务与 SSH 隧道。'}

    @app.get('/api/jobs')
    def jobs():
        return [{k:j[k] for k in ('id','kind','title','status','created_at','updated_at')} | {'version_count':len(j['versions'])} for j in store.list()]

    @app.post('/api/jobs',status_code=201)
    async def upload(file:UploadFile=File(...),instructions:str=Form(''),page:int=Form(1)):
        if len(instructions)>2000:
            raise HTTPException(422,'补充要求最多 2000 字')
        data=await file.read(settings.max_upload+1)
        if not data or len(data)>settings.max_upload:
            raise HTTPException(413,'文件为空或超过 20 MB')
        image,pages=prepare_image(data,page)
        name=(file.filename or 'drawing').replace('\\','/').split('/')[-1][:100]
        job=new_job('drawing',name,instructions)
        job.update(page=page,pages=pages,status='ready' if settings.remote_url else 'waiting_server')
        folder=settings.storage/'jobs'/job['id']/'input'
        folder.mkdir(parents=True)
        image.save(folder/'preview.png')
        (folder/'original').write_bytes(data)
        store.create(job)
        store.event(job['id'],'upload','success',f'图纸已保存，共 {pages} 页，当前第 {page} 页。')
        store.message(job['id'],'assistant','图纸已收到。'+('点击“开始建模”将发送到已配置的服务器。' if settings.remote_url else '当前未连接建模服务器；图纸已保存在本地，接通后可开始建模。'))
        return get_job(job['id'])

    @app.post('/api/demo',status_code=201)
    async def demo(params:Parameters):
        job=new_job('demo','五孔安装板 · 参数化样例',parameters=params.model_dump())
        store.create(job)
        store.message(job['id'],'assistant','这是本地已知 CadQuery 模板，用于验证工作台和文件导出；不代表 AI 已识别图纸。')
        worker.enqueue(job['id'])
        return get_job(job['id'])

    @app.get('/api/jobs/{job_id}')
    def detail(job_id:str):
        return get_job(job_id) | {'events':store.events(job_id)}

    @app.post('/api/jobs/{job_id}/run',status_code=202)
    async def run_job(job_id:str,body:RunInput):
        job=get_job(job_id); ensure_idle(job)
        if body.parameters:
            if job['kind']!='demo':
                raise HTTPException(422,'参数面板只适用于本地样例，真实图纸请使用对话要求')
            store.patch(job_id,parameters=body.parameters.model_dump())
        worker.enqueue(job_id)
        return get_job(job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    async def cancel(job_id:str):
        get_job(job_id)
        await worker.cancel(job_id)
        return get_job(job_id)

    @app.post('/api/jobs/{job_id}/messages')
    async def message(job_id:str,body:MessageInput):
        job=get_job(job_id); ensure_idle(job)
        content=body.content.strip()
        if not content:
            raise HTTPException(422,'请输入修改要求')
        store.message(job_id,'user',content)
        if job['kind']=='demo':
            params=job['parameters'].copy()
            found=False
            for pattern,key in [(r'(?:长度|长)\s*(?:改成|改为|设为|设置为|为|=|：|:)?\s*(\d+(?:\.\d+)?)','length'),(r'(?:宽度|宽)\s*(?:改成|改为|设为|设置为|为|=|：|:)?\s*(\d+(?:\.\d+)?)','width'),(r'(?:厚度|厚)\s*(?:改成|改为|设为|设置为|为|=|：|:)?\s*(\d+(?:\.\d+)?)','thickness'),(r'(?:中心孔径|中心孔直径|孔径)\s*(?:改成|改为|设为|设置为|为|=|：|:)?\s*(\d+(?:\.\d+)?)','hole')]:
                match=re.search(pattern,content)
                if match:
                    params[key]=float(match.group(1)); found=True
            if not found:
                store.message(job_id,'assistant','样例使用确定性参数编辑，支持“长度改为 100”“厚度改为 15”“孔径改为 24”，单位 mm。也可以使用参数面板。')
                return get_job(job_id)
            try:
                params=Parameters(**params).model_dump()
            except ValidationError:
                store.message(job_id,'assistant','参数超出模板范围，或中心孔与外边界冲突。请使用参数面板检查数值；旧版本保持不变。')
                return get_job(job_id)
            store.patch(job_id,parameters=params)
        worker.enqueue(job_id)
        return get_job(job_id)

    @app.get('/api/jobs/{job_id}/drawing')
    def drawing(job_id:str,version:int|None=None):
        job=get_job(job_id)
        if job['kind']=='demo':
            v=choose_version(job,version)
            path=settings.storage/'jobs'/job_id/'attempts'/v['attempt']/'drawing.png'
        else:
            path=settings.storage/'jobs'/job_id/'input'/'preview.png'
        if not path.is_file(): raise HTTPException(404,'图纸预览尚未生成')
        return FileResponse(path,media_type='image/png')

    @app.get('/api/jobs/{job_id}/artifacts/{key}')
    def artifact(job_id:str,key:str,version:int|None=None):
        names={'step':('model.step','application/step'),'stl':('model.stl','model/stl'),'code':('model.py','text/plain; charset=utf-8'),'report':('report.json','application/json')}
        if key not in names: raise HTTPException(404,'未知产物')
        job=get_job(job_id); v=choose_version(job,version)
        filename,mime=names[key]
        path=settings.storage/'jobs'/job_id/'attempts'/v['attempt']/filename
        if not path.is_file(): raise HTTPException(404,'文件不存在')
        return FileResponse(path,media_type=mime,filename=f"drawing2cad_v{v['number']}_{filename}")

    @app.get('/api/jobs/{job_id}/events')
    async def events(job_id:str,request:Request,after:int=0):
        get_job(job_id)
        try: cursor=max(after,int(request.headers.get('last-event-id','0')))
        except ValueError: cursor=after
        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                for event in store.events(job_id,cursor):
                    cursor=event['id']
                    yield f"id: {cursor}\ndata: {json.dumps(event,ensure_ascii=False)}\n\n"
                yield ': heartbeat\n\n'
                await asyncio.sleep(1)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    dist=PROJECT/'frontend'/'dist'
    if dist.exists():
        app.mount('/',StaticFiles(directory=dist,html=True),name='frontend')
    return app


def choose_version(job,number):
    versions=job['versions']
    if not versions: raise HTTPException(404,'还没有完成的模型版本')
    if number is None: return versions[-1]
    return next((v for v in versions if v['number']==number),None) or fail_version()


def fail_version():
    raise HTTPException(404,'版本不存在')


def prepare_image(data,page):
    if page<1: raise HTTPException(422,'页码从 1 开始')
    try:
        if data.startswith(b'%PDF'):
            import pypdfium2 as pdfium
            with pdfium.PdfDocument(data) as pdf:
                count=len(pdf)
                if count>100: raise HTTPException(422,'PDF 最多支持 100 页')
                if page>count: raise HTTPException(422,f'PDF 共 {count} 页，请输入有效页码')
                p=pdf[page-1]
                try:
                    scale=min(2,2400/max(p.get_size()))
                    image=p.render(scale=scale).to_pil().convert('RGB')
                finally: p.close()
                return image,count
        if page!=1: raise HTTPException(422,'图片只支持第 1 页')
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in ('PNG','JPEG'): raise HTTPException(415,'请上传 PNG、JPG 或 PDF')
                if im.width*im.height>30_000_000: raise HTTPException(413,'图像像素超过 3000 万')
                image=im.convert('RGB')
                image.thumbnail((3000,3000))
                return image,1
    except HTTPException: raise
    except Exception:
        raise HTTPException(422,'无法解析文件，请检查图片或 PDF 是否完整、是否加密')


app=create_app()
