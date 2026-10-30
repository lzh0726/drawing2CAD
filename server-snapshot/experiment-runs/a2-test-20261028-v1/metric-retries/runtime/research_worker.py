"""Trusted entry point; execute candidate code ONLY inside research_sandbox."""
import io
import json
import resource
import sys
import contextlib
import zipfile
from pathlib import Path

resource.setrlimit(resource.RLIMIT_AS, (6*1024**3, 6*1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
resource.setrlimit(resource.RLIMIT_FSIZE, (40*1024**2, 40*1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (128,128))
sys.path.insert(0, '/runner')
from policy import validate_code

def describe(s):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b=s.BoundingBox(); dims=[b.xlen,b.ylen,b.zlen]
    broad=Bnd_Box(); BRepBndLib.Add_s(s.wrapped,broad); a=broad.Get()
    scale=max(dims)
    signature=[sorted(round(x/scale,3) for x in dims),round(s.Volume()/scale**3,3),round(s.Area()/scale**2,3),sorted(round(f.Area()/scale**2,3) for f in s.Faces())]
    return {'valid':True,'dimensions':dims,'center':[(b.xmin+b.xmax)/2,(b.ymin+b.ymax)/2,(b.zmin+b.zmax)/2], 'volume':s.Volume(),'broad_dimensions':[a[3]-a[0],a[4]-a[1],a[5]-a[2]],'geometric_signature':json.dumps(signature),'unit':'mm'}

def main():
    request=json.loads(sys.stdin.buffer.read(4*1024**2))
    if request.get('probe'):
        import os,socket
        denied=False
        try: Path('/usr/probe-file').write_text('x')
        except OSError: denied=True
        sock=socket.socket(); sock.settimeout(.2)
        try: sock.connect(('1.1.1.1',443)); net=False
        except OSError: net=True
        print(json.dumps({'home_hidden':not Path('/home/luzhuohong/.ssh').exists(),'root_readonly':denied,'network_blocked':net,'gpu_hidden':not Path('/dev/nvidia0').exists(),'env_cleared':'SSH_AUTH_SOCK' not in os.environ}))
        return
    import cadquery as cq
    if 'prediction_b64' in request:
        import base64
        shapes=[]
        for key in ('prediction_b64','truth_b64'):
            path='/work/'+key+'.step'
            Path(path).write_bytes(base64.b64decode(request[key]))
            shape=cq.importers.importStep(path).val()
            if not shape.isValid() or len(shape.Solids())!=1: raise ValueError('invalid_comparison_solid')
            shapes.append(shape)
        pred,truth=shapes
        pg,tg=describe(pred),describe(truth)
        def iou(a,b):
            intersection=a.intersect(b).Volume()
            union=a.Volume()+b.Volume()-intersection
            return max(0,min(1,intersection/union)) if union>0 else 0
        delta=[b-a for a,b in zip(pg['center'],tg['center'])]
        absolute=[abs(a-b) for a,b in zip(pg['dimensions'],tg['dimensions'])]
        print(json.dumps({'iou_raw':iou(pred,truth),'iou_translation_aligned':iou(pred.translate(tuple(delta)),truth),'bbox_absolute_mm':absolute,'bbox_relative':[a/max(abs(b),1e-9) for a,b in zip(absolute,tg['dimensions'])],'alignment_translation_mm':delta,'scale_alignment':False}))
        return
    if 'step_b64' in request:
        import base64
        Path('/work/input.step').write_bytes(base64.b64decode(request['step_b64']))
        result=cq.importers.importStep('/work/input.step')
    else:
        code=request['code']; validate_code(code)
        def allowed_import(name,*a,**k):
            if name=='cadquery': return cq
            if name=='math': return __import__('math')
            raise ValueError('Unsupported import')
        safe={'__import__':allowed_import,'range':range,'len':len,'list':list,'tuple':tuple,'dict':dict,'enumerate':enumerate,'zip':zip,'min':min,'max':max,'abs':abs,'round':round,'sum':sum,'float':float,'int':int,'bool':bool}
        ns={'__builtins__':safe}
        if request.get('diagnose'):
            from cad_diagnostics import diagnose
            print(json.dumps(diagnose(code,ns,cq)))
            return
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): exec(compile(code,'candidate.py','exec'),ns)
        result=ns.get('result')
    if not isinstance(result,cq.Workplane): raise ValueError('result_not_workplane')
    solids=result.solids().vals()
    if len(solids)!=1 or not solids[0].isValid() or solids[0].Volume()<=0: raise ValueError('invalid_or_multiple_solids')
    shape=solids[0]; report=describe(shape)
    cq.exporters.export(shape,'/work/model.step')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        z.write('/work/model.step','model.step'); z.writestr('geometry.json',json.dumps(report))
    sys.stdout.buffer.write(stream.getvalue())

if __name__=='__main__':
    try: main()
    except Exception as e:
        print(type(e).__name__+': '+str(e)[:300],file=sys.stderr)
        sys.exit(2)
