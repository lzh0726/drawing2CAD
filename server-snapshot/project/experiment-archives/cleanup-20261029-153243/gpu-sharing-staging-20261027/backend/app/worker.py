import asyncio
import io
import json
import os
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path
from typing import TypedDict
import httpx
from langgraph.graph import StateGraph, START, END
from .config import PROJECT


ACTIVE = {'queued','preparing','generating','checking'}


class State(TypedDict):
    job_id: str
    attempt: str


class Worker:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.tasks = {}
        self.slot = asyncio.Semaphore(1)
        graph = StateGraph(State)
        graph.add_node('prepare', self.prepare)
        graph.add_node('generate', self.generate)
        graph.add_node('inspect', self.inspect)
        graph.add_node('publish', self.publish)
        graph.add_edge(START,'prepare')
        graph.add_edge('prepare','generate')
        graph.add_edge('generate','inspect')
        graph.add_edge('inspect','publish')
        graph.add_edge('publish',END)
        self.graph = graph.compile()

    def folder(self,state):
        return self.settings.storage / 'jobs' / state['job_id'] / 'attempts' / state['attempt']

    def enqueue(self,job_id):
        if job_id in self.tasks and not self.tasks[job_id].done():
            raise ValueError('任务正在运行')
        job = self.store.get(job_id)
        if job['kind'] == 'drawing' and not self.settings.remote_url:
            self.store.patch(job_id,status='waiting_server')
            self.store.event(job_id,'connection','warning','图纸已保存。请配置服务器地址后重新运行；本地不会猜测三维结构。')
            return
        self.store.patch(job_id,status='queued',error=None)
        self.store.event(job_id,'queue','info','任务已进入本地队列。')
        self.tasks[job_id] = asyncio.create_task(self.process(job_id))

    async def process(self,job_id):
        try:
            async with self.slot:
                await self.graph.ainvoke({'job_id':job_id,'attempt':uuid.uuid4().hex})
        except asyncio.CancelledError:
            job = self.store.get(job_id)
            if job and job['status'] in ACTIVE:
                self.store.patch(job_id,status='cancelled')
                self.store.event(job_id,'system','warning','本地任务已取消，已完成版本保留。')
            raise
        except Exception as exc:
            # Never expose remote HTTP bodies, headers, or environment credentials.
            reason = str(exc) if isinstance(exc, UserError) else '处理失败，请检查服务环境或服务器协议。'
            self.store.patch(job_id,status='failed',error=reason)
            self.store.event(job_id,'system','error',reason)
            self.store.message(job_id,'assistant',reason + ' 已完成的模型版本未被覆盖。')

    async def prepare(self,state):
        self.folder(state).mkdir(parents=True,exist_ok=True)
        self.store.patch(state['job_id'],status='preparing')
        self.store.event(state['job_id'],'prepare','info','正在准备任务输入与独立输出目录。')
        return state

    async def generate(self,state):
        job = self.store.get(state['job_id'])
        folder = self.folder(state)
        self.store.patch(job['id'],status='generating')
        if job['kind'] == 'demo':
            self.store.event(job['id'],'generate','info','执行本地已知 CadQuery 模板；这是参数化样例，不是 AI 识图结果。')
            (folder/'params.json').write_text(json.dumps(job['parameters']),encoding='utf-8')
            await self.cad('demo',folder)
        else:
            self.store.event(job['id'],'generate','info','图纸正在发送到你配置的建模服务器。')
            await self.remote(job,folder)
        return state

    async def cad(self,mode,folder):
        flags = {'creationflags':subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        proc = await asyncio.create_subprocess_exec(sys.executable,'-m','backend.app.cad_runner',mode,str(folder),cwd=str(PROJECT),stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.PIPE,**flags)
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(),90)
            if proc.returncode:
                raise UserError('CAD 执行或实体校验失败，请检查输入几何和参数。')
        except (asyncio.CancelledError, asyncio.TimeoutError):
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            if asyncio.current_task().cancelling():
                raise
            raise UserError('CAD 执行超过 90 秒，已停止本次执行。')

    async def remote(self,job,folder):
        input_dir = self.settings.storage/'jobs'/job['id']/'input'
        previous = ''
        if job['versions']:
            previous = (self.settings.storage/'jobs'/job['id']/'attempts'/job['versions'][-1]['attempt']/'model.py').read_text(encoding='utf-8')
        metadata = {'unit':'mm','instructions':job['instructions'],'messages':job['messages'],'previous_code':previous}
        headers = {'Authorization':f'Bearer {self.settings.remote_token}'} if self.settings.remote_token else {}
        monitor=asyncio.create_task(self.monitor_remote(job['id'],headers))
        try:
            async with httpx.AsyncClient(timeout=self.settings.remote_timeout,follow_redirects=False,trust_env=False) as client:
                with (input_dir/'preview.png').open('rb') as image:
                    async with client.stream('POST',self.settings.remote_url+'/v1/reconstruct',headers=headers,files={'drawing':('drawing.png',image,'image/png')},data={'metadata':json.dumps(metadata,ensure_ascii=False)}) as response:
                        if response.status_code != 200:
                            if response.status_code==503:
                                await response.aread()
                                try:reason=response.json().get('detail',{})
                                except ValueError:reason={}
                                if isinstance(reason,dict) and reason.get('code')=='gpu_wait_timeout':
                                    raise UserError('等待符合空闲或共享条件的 GPU 超过 5 分钟，已结束本次排队；图纸已保留，请稍后重试。')
                            raise UserError(f'建模服务器返回 HTTP {response.status_code}。请检查连接和接口协议。')
                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            data.extend(chunk)
                            if len(data)>50*1024*1024:
                                raise UserError('服务器返回文件超过 50 MB，已拒绝。')
        except (httpx.HTTPError, OSError):
            raise UserError('无法连接建模服务器，或请求超时。请检查 SSH 隧道和远端服务。')
        finally:
            monitor.cancel()
            await asyncio.gather(monitor,return_exceptions=True)
        unpack_response(bytes(data),folder)

    async def monitor_remote(self,job_id,headers):
        messages={'waiting_gpu':'正在等待空闲 GPU，或显存充足且持续低负载的共享 GPU。','loading_and_generating':'GPU 已通过准入检查，正在加载模型并生成代码；生成后自动释放显存。','idle':'模型推理已结束，继续等待 CAD 检查结果。'}
        previous=None
        async with httpx.AsyncClient(timeout=5,trust_env=False,follow_redirects=False) as client:
            while True:
                try:
                    response=await client.get(self.settings.remote_url+'/health',headers=headers)
                    state=response.json().get('model_state') if response.status_code==200 else None
                    if state in messages and state!=previous and (state!='idle' or previous is not None):
                        self.store.event(job_id,'remote','info',messages[state])
                    previous=state
                except (httpx.HTTPError,ValueError):pass
                await asyncio.sleep(5)

    async def inspect(self,state):
        job = self.store.get(state['job_id'])
        self.store.patch(job['id'],status='checking')
        self.store.event(job['id'],'inspect','info','检查 STEP 实体有效性，并生成与该实体对应的 STL 预览。')
        if job['kind'] != 'demo':
            await self.cad('inspect',self.folder(state))
        return state

    async def publish(self,state):
        folder = self.folder(state)
        report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        job = self.store.get(state['job_id'])
        from .store import now
        version = {'number':len(job['versions'])+1,'attempt':state['attempt'],'created_at':now(),'parameters':job['parameters'],'report':report}
        self.store.mutate(job['id'],lambda j: (j['versions'].append(version),j.update(status='completed',error=None)))
        self.store.event(job['id'],'complete','success',f"版本 v{version['number']} 已保存。实体有效；图纸符合性仍需核验。")
        self.store.message(job['id'],'assistant',f"已生成 v{version['number']}，可以旋转预览并下载 STEP、STL 和源码。" + ('这是本地参数化样例，可修改长、宽、厚度和中心孔径。' if job['kind']=='demo' else '几何有效性检查已通过，尚未验证与原图全部尺寸一致。'))
        return state

    async def cancel(self,job_id):
        task = self.tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
            if self.store.get(job_id)['status'] in ACTIVE:
                self.store.patch(job_id,status='cancelled')
                self.store.event(job_id,'system','warning','排队任务已取消。')

    async def close(self):
        for job_id,task in self.tasks.items():
            if not task.done():
                self.store.patch(job_id,status='interrupted')
                task.cancel()
        await asyncio.gather(*self.tasks.values(),return_exceptions=True)


class UserError(Exception):
    pass


def unpack_response(data,folder):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name,limit in [('model.step',40*1024*1024),('model.py',512*1024)]:
                matches = [x for x in z.infolist() if x.filename==name]
                if len(matches)!=1 or matches[0].file_size>limit:
                    raise UserError('服务器结果缺少 STEP/源码、包含重名文件或文件过大。')
                content = z.read(matches[0])
                if not content:
                    raise UserError('服务器返回空文件。')
                if name.endswith('.py'):
                    content.decode('utf-8')
                (folder/name).write_bytes(content)
    except (zipfile.BadZipFile,UnicodeError,RuntimeError,KeyError):
        raise UserError('服务器结果不是有效的 CAD ZIP 包，请按接口文档实现适配器。')
