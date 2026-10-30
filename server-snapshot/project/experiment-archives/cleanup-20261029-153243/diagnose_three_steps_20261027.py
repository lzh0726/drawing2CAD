"""Orchestrator: candidate execution is confined to the existing bwrap sandbox."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('/home/luzhuohong/drawing2CAD')
sys.path.insert(0, str(ROOT))
from server.research_sandbox import command, run
from server.policy import extract_code, validate_code

SOURCE = ROOT/'experiments/finetuned-pilot-v1-repairs0'
OUT = ROOT/'experiments'/('invalid-solids-step-diagnosis-'+time.strftime('%Y%m%d-%H%M%S'))
OUT.mkdir()
shutil.copy2(__file__, OUT/'diagnose.py')
def hashes():
    paths = list(SOURCE.glob('*')) + list((ROOT/'tools/cad-runtime').glob('*.py'))
    paths += [ROOT/'server/research_sandbox.py', ROOT/'datasets/pilot-v1/manifest.jsonl']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
before = hashes()
(OUT/'source-sha256-before.json').write_text(json.dumps(before,indent=2))
runtime = OUT/'runtime'
shutil.copytree(ROOT/'tools/cad-runtime', runtime, ignore=shutil.ignore_patterns('__pycache__'))
worker = runtime/'research_worker.py'
text = worker.read_text()
old = "        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): exec(compile(code,'candidate.py','exec'),ns)"
new = '''        import ast
        from OCP.BRepCheck import BRepCheck_Analyzer
        def detail(shape):
            record={'type':shape.ShapeType(),'valid':shape.isValid()}
            solids=shape.Solids()
            record['solids']=[{'valid':s.isValid(),'volume':s.Volume()} for s in solids]
            record['wires']=[{'valid':w.isValid(),'closed':w.IsClosed(),'edges':len(w.Edges())} for w in shape.Wires()]
            if not record['valid']:
                analyzer=BRepCheck_Analyzer(shape.wrapped)
                errors=[]
                for kind,items in [('shape',[shape]),('face',shape.Faces()),('wire',shape.Wires()),('edge',shape.Edges())]:
                    for i,item in enumerate(items):
                        try:
                            statuses=[str(s) for s in analyzer.Result(item.wrapped).Status()]
                            statuses=[s for s in statuses if not s.endswith('BRepCheck_NoError')]
                            if statuses:errors.append({'kind':kind,'index':i,'statuses':statuses})
                        except Exception as e:errors.append({'kind':kind,'index':i,'diagnostic_error':type(e).__name__})
                record['brep_errors']=errors
            return record
        for statement in ast.parse(code).body:
            record={'line':statement.lineno,'source':ast.get_source_segment(code,statement)}
            try:
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    exec(compile(ast.Module(body=[statement],type_ignores=[]),'candidate.py','exec'),ns)
                if isinstance(statement,ast.Assign):
                    for target in statement.targets:
                        if isinstance(target,ast.Name) and isinstance(ns.get(target.id),cq.Workplane):
                            record['variable']=target.id
                            record['objects']=[detail(s) for s in ns[target.id].vals() if isinstance(s,cq.Shape)]
                print('STEP_JSON '+json.dumps(record),file=sys.__stderr__,flush=True)
            except Exception as e:
                record['exception']=type(e).__name__+': '+str(e)
                print('STEP_JSON '+json.dumps(record),file=sys.__stderr__,flush=True)
                raise
'''
assert old in text
worker.write_text(text.replace(old,new))
cmd = command();cmd[cmd.index(str(ROOT/'tools/cad-runtime'))] = str(runtime)
probe = subprocess.run(cmd,input=json.dumps({'probe':True}).encode(),capture_output=True,timeout=15)
assert probe.returncode==0 and all(json.loads(probe.stdout).values())
(OUT/'sandbox-probe.json').write_bytes(probe.stdout)
rows={r['id']:r for r in map(json.loads,(ROOT/'datasets/pilot-v1/manifest.jsonl').read_text().splitlines())}
records=[]
for ident in ['test-00306373','test-00790402','test-00932166']:
    folder=OUT/ident;folder.mkdir()
    code=extract_code((SOURCE/(ident+'-attempt1.txt')).read_text());validate_code(code)
    (folder/'candidate.py').write_text(code)
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try:out,err=p.communicate(json.dumps({'code':code}).encode(),timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);out,err=p.communicate()
    (folder/'trace.stderr').write_bytes(err)
    steps=[json.loads(l[len('STEP_JSON '):]) for l in err.decode(errors='replace').splitlines() if l.startswith('STEP_JSON ')]
    invalid=[s for s in steps if any(not o['valid'] for o in s.get('objects',[]))]
    bad_solid=[s for s in steps if any(not v['valid'] for o in s.get('objects',[]) for v in o['solids'])]
    rec={'id':ident,'returncode':p.returncode,'steps':steps,'first_invalid_shape_line':invalid[0]['line'] if invalid else None,'first_invalid_solid_line':bad_solid[0]['line'] if bad_solid else None}
    reference=(ROOT/'datasets/pilot-v1'/rows[ident]['code']).read_text()
    (folder/'reference.py').write_text(reference)
    try:
        _,geometry=run({'code':reference},timeout=45);rec['reference']={'passed':True,'geometry':geometry}
    except Exception as e:rec['reference']={'passed':False,'error':str(e)}
    (folder/'diagnosis.json').write_text(json.dumps(rec,indent=2))
    records.append(rec)
    print(ident,'first_invalid_shape',rec['first_invalid_shape_line'],'first_invalid_solid',rec['first_invalid_solid_line'],flush=True)
after=hashes()
(OUT/'source-sha256-after.json').write_text(json.dumps(after,indent=2))
assert before==after
(OUT/'summary.json').write_text(json.dumps({'source_unchanged':True,'gpu_started':False,'records':records},indent=2))
print('COMPLETE',OUT,flush=True)

