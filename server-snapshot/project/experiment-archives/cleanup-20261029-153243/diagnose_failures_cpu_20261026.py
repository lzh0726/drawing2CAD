import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/luzhuohong/drawing2CAD')
sys.path.insert(0, str(ROOT))
from server.policy import extract_code, validate_code
from server.research_sandbox import run, command

OUT = ROOT / 'experiments' / ('failure-diagnosis-cpu-' + time.strftime('%Y%m%d-%H%M%S'))
OUT.mkdir(exist_ok=False)
shutil.copy2(__file__, OUT / 'diagnose.py')
SOURCE = ROOT / 'experiments/finetuned-pilot-v1-repairs0'
def hashes():
    paths = list(SOURCE.glob('*')) + list((ROOT/'experiments/finetuned-pilot-v1-repairs2').glob('*'))
    paths += [ROOT/'datasets/pilot-v1/manifest.jsonl', ROOT/'server/research_sandbox.py']
    paths += list((ROOT/'tools/cad-runtime').glob('*.py'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
before = hashes()
(OUT/'source-sha256-before.json').write_text(json.dumps(before, indent=2))
probe = run({'probe':True}, timeout=15)
assert all(probe.values()), probe
(OUT/'sandbox-probe.json').write_text(json.dumps(probe, indent=2))
runtime = OUT/'runtime'
runtime.mkdir()
for p in (ROOT/'tools/cad-runtime').glob('*.py'):
    shutil.copy2(p, runtime/p.name)
worker = (runtime/'research_worker.py').read_text()
worker = worker.replace("ns={'__builtins__':safe}", """ns={'__builtins__':safe}
        def trace(frame, event, arg):
            if frame.f_code.co_filename == 'candidate.py' and event == 'line':
                print('CANDIDATE_LINE '+str(frame.f_lineno), file=sys.__stderr__, flush=True)
            return trace
        sys.settrace(trace)""")
worker = worker.replace("result=ns.get('result')", "sys.settrace(None)\n        result=ns.get('result')")
worker = worker.replace("raise ValueError('invalid_or_multiple_solids')", "raise ValueError('invalid_or_multiple_solids '+json.dumps({'count':len(solids),'valid':[s.isValid() for s in solids],'volumes':[s.Volume() for s in solids]}))")
(runtime/'research_worker.py').write_text(worker)
def traced(code, folder):
    cmd = command()
    cmd[cmd.index(str(ROOT/'tools/cad-runtime'))] = str(runtime)
    with (folder/'trace.stderr').open('wb') as err, (folder/'trace-output.zip').open('wb') as out:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out, stderr=err, start_new_session=True)
        try:
            p.communicate(json.dumps({'code':code}).encode(), timeout=120)
            return {'returncode':p.returncode,'wall_timeout':False}
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
            return {'returncode':p.returncode,'wall_timeout':True}

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'models/drawing2cad-v1'), local_files_only=True, trust_remote_code=False)
rows = {r['id']:r for r in map(json.loads, (ROOT/'datasets/pilot-v1/manifest.jsonl').read_text().splitlines())}
failures = [json.loads(p.read_text()) for p in sorted(SOURCE.glob('test-*.json'))]
failures = [r for r in failures if r['valid_ground_truth'] and not r['valid_solid']]
assert len(failures)==10
records=[]
for original in failures:
    ident=original['id']; folder=OUT/ident;folder.mkdir()
    raw=(SOURCE/(ident+'-attempt1.txt')).read_text()
    (folder/'original-response.txt').write_text(raw)
    rec={'id':ident,'original_attempts':original['attempts'],'response_tokens_reencoded':len(tokenizer.encode(raw,add_special_tokens=False)), 'closed_code_fence':raw.rstrip().endswith('```')}
    body=re.sub(r'^```(?:python|py)?\s*\n','',raw).removesuffix('```').strip()
    rec['sketch_blocks']=len(re.findall(r'# Generating a workplane',body))
    # Diagnostic normalization removes sketch indices only, without executing code.
    lines=[re.sub(r'(wp_sketch|loop|solid)\d+',r'\1N',l) for l in body.splitlines() if l and not l.startswith('#')]
    rec['repeated_normalized_lines']=len(lines)-len(set(lines))
    try:
        code=extract_code(raw);validate_code(code)
        rec['policy']='passed'
    except (SyntaxError,ValueError) as e:
        rec['policy']=type(e).__name__+': '+str(e)
        try: ast.parse(body);rec['body_syntax']='passed'
        except SyntaxError as e:rec['body_syntax']={'message':e.msg,'line':e.lineno,'text':e.text}
        rec['classification']='incomplete_generation_with_repetition'
    else:
        (folder/'candidate.py').write_text(code)
        started=time.monotonic()
        try:
            step,geometry=run({'code':code}, timeout=45)
            (folder/'replay.step').write_bytes(step)
            rec['replay']={'valid':True,'geometry':geometry}
        except Exception as e:
            rec['replay']={'valid':False,'error':str(e)}
        rec['replay_seconds']=round(time.monotonic()-started,3)
        rec['instrumented']=traced(code,folder)
        trace=(folder/'trace.stderr').read_text(errors='replace')
        rec['trace_tail']=trace[-1800:]
        rec['classification']='execution_timeout' if 'CAD_timeout' in rec['replay'].get('error','') else ('replay_succeeded' if rec['replay']['valid'] else 'invalid_geometry')
    # Execute the matched reference program only through the same sandbox.
    reference=(ROOT/'datasets/pilot-v1'/rows[ident]['code']).read_text()
    try:
        validate_code(reference)
        _, geom=run({'code':reference},timeout=45)
        rec['reference_replay']={'valid':True,'geometry':geom}
    except Exception as e:rec['reference_replay']={'valid':False,'error':str(e)}
    (folder/'diagnosis.json').write_text(json.dumps(rec,indent=2))
    records.append(rec)
    print(json.dumps({'id':ident,'class':rec['classification'],'tokens':rec['response_tokens_reencoded'],'reference_valid':rec['reference_replay']['valid']}),flush=True)
after=hashes()
(OUT/'source-sha256-after.json').write_text(json.dumps(after,indent=2))
summary={'directory':str(OUT),'count':len(records),'source_records_unchanged':before==after,'gpu_started':False,'sandbox_probe':probe,'records':records}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
assert before==after, 'Source files unexpectedly changed'
print('COMPLETE '+str(OUT),flush=True)
