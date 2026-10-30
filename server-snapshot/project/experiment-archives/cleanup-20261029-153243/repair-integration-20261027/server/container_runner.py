"""Runs ONLY inside the CAD container. stdin=source, stdout=ZIP."""
import contextlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from policy import validate_code, MAX_CODE

def main():
    source=sys.stdin.buffer.read(MAX_CODE+1).decode('utf-8')
    if len(source.encode())>MAX_CODE:
        raise ValueError('Code too large')
    validate_code(source)
    import cadquery as cq
    def allowed_import(name,*args,**kwargs):
        if name=='cadquery': return cq
        if name=='math': return math
        raise ValueError('Unsupported import')
    safe={'__import__':allowed_import,'range':range,'len':len,'list':list,'tuple':tuple,'dict':dict,'enumerate':enumerate,'zip':zip,'min':min,'max':max,'abs':abs,'round':round,'sum':sum,'float':float,'int':int,'bool':bool}
    namespace={'__builtins__':safe}
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        exec(compile(source,'model.py','exec'),namespace)
        result=namespace.get('result')
        if not isinstance(result,cq.Workplane):
            raise ValueError('result must be a CadQuery Workplane')
        solids=result.solids().vals()
        if len(solids)!=1 or not solids[0].isValid() or solids[0].Volume()<=0:
            raise ValueError('Expected one valid nonempty solid')
        solid=solids[0];box=solid.BoundingBox()
        cq.exporters.export(result,'/work/model.step')
        path=Path('/work/model.step')
        if path.stat().st_size>40*1024*1024:
            raise ValueError('STEP exceeds 40 MB')
        report={'dimensions':[box.xlen,box.ylen,box.zlen],'volume':solid.Volume(),'unit':'mm','valid':True,'drawing_conformance':'unverified'}
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(path,'model.step')
        z.writestr('geometry.json',json.dumps(report))
    sys.stdout.buffer.write(out.getvalue())

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        sys.stderr.write(type(error).__name__+': CAD operation failed. Check result, sketch closure, dimensions and selected faces.\n')
        sys.exit(2)
