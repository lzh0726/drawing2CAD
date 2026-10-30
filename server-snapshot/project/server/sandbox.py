import asyncio
import io
import json
import uuid
import zipfile

class SandboxUnavailable(RuntimeError):
    pass

class CADFailure(RuntimeError):
    pass

class DockerExecutor:
    def __init__(self,image='drawing2cad-executor:0.1',timeout=90):
        self.image,self.timeout=image,timeout

    async def available(self):
        try:
            proc=await asyncio.create_subprocess_exec('docker','image','inspect',self.image,stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
            try: return await asyncio.wait_for(proc.wait(),10)==0
            except asyncio.TimeoutError:
                proc.kill();await proc.wait();return False
        except OSError: return False

    def command(self,name):
        return ['docker','run','--rm','-i','--name',name,'--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534','--memory','2g','--memory-swap','2g','--cpus','2','--pids-limit','64','--ulimit','fsize=52428800:52428800','--tmpfs','/tmp:rw,noexec,nosuid,size=128m,uid=65534,gid=65534','--tmpfs','/work:rw,noexec,nosuid,size=128m,uid=65534,gid=65534',self.image]

    async def run(self,code):
        name='drawing2cad-'+uuid.uuid4().hex
        try:
            proc=await asyncio.create_subprocess_exec(*self.command(name),stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL)
        except OSError:
            raise SandboxUnavailable('Docker is unavailable; no host execution fallback is allowed.')
        async def collect():
            proc.stdin.write(code.encode());await proc.stdin.drain();proc.stdin.close()
            data=bytearray()
            while chunk:=await proc.stdout.read(65536):
                data.extend(chunk)
                if len(data)>50*1024*1024: raise CADFailure('CAD output exceeds size limit.')
            await proc.wait()
            if proc.returncode==125: raise SandboxUnavailable('Container could not start; check Docker image and permissions.')
            if proc.returncode: raise CADFailure('CAD execution failed. Check closed sketches, feature dimensions, face selectors and result.')
            return unpack_geometry(bytes(data))
        try:
            return await asyncio.wait_for(collect(),self.timeout)
        except asyncio.TimeoutError:
            raise CADFailure('CAD execution time limit reached. Simplify the model.')
        finally:
            cleanup=await asyncio.create_subprocess_exec('docker','rm','-f',name,stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
            try: await asyncio.wait_for(cleanup.wait(),10)
            except asyncio.TimeoutError: cleanup.kill();await cleanup.wait()
            if proc.returncode is None:
                proc.kill();await proc.wait()

def unpack_geometry(data):
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name,limit in [('model.step',40*1024*1024),('geometry.json',4096)]:
                entries=[x for x in z.infolist() if x.filename==name]
                if len(entries)!=1 or entries[0].file_size>limit: raise ValueError('Invalid result')
            step=z.read('model.step')
            report=json.loads(z.read('geometry.json'))
            if not step.startswith(b'ISO-10303-21;') or report.get('valid') is not True: raise ValueError('Invalid result')
            return step,report
    except (ValueError,KeyError,zipfile.BadZipFile):
        raise CADFailure('CAD container returned invalid artifacts.')
