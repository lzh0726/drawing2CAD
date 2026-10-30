"""CPU-only exact-IoU retries; preserve primary records and their errors."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/home/luzhuohong/drawing2CAD')
sys.path.insert(0,str(ROOT))
from server.research_sandbox import command


def main(folder):
    spec=importlib.util.spec_from_file_location('a2_eval',folder/'evaluate_a2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out=folder/'metric-retries';out.mkdir(exist_ok=True)
    runtime=out/'runtime';runtime.mkdir(exist_ok=True)
    original=ROOT/'tools/cad-runtime'
    text=(original/'research_worker.py').read_text()
    assert 'resource.setrlimit(resource.RLIMIT_CPU, (35, 35))' in text
    (runtime/'research_worker.py').write_text(text.replace(
        'resource.setrlimit(resource.RLIMIT_CPU, (35, 35))',
        'resource.setrlimit(resource.RLIMIT_CPU, (120, 120))',1))
    (runtime/'policy.py').write_bytes((original/'policy.py').read_bytes())
    protocol=json.loads((folder/'protocol.json').read_text())
    rows={r['id']:r for r in [json.loads(l) for l in (module.DATA/'manifest.jsonl').read_text().splitlines()]}
    baseline=module.BASELINE
    commands=command()
    pos=commands.index(str(original));commands[pos]=str(runtime)
    while True:
        module.save(out/'progress.json',{'status':'running','updated_at':module.stamp()})
        for group, source in [('baseline',folder/'baseline-first-pass'),('a2',folder/'samples')]:
            for path in sorted(source.glob('test-*.json')):
                record=json.loads(path.read_text())
                if not record.get('valid_solid') or 'metrics' in record:continue
                target=out/(group+'-'+record['id']+'.json')
                if target.exists():continue
                ident=record['id'];step=(baseline if group=='baseline' else source)/(ident+'.step')
                truth=module.DATA/rows[ident]['step']
                item={'id':ident,'group':group,'cpu_seconds':120,'wall_seconds':300,
                      'prediction_sha256':module.digest(step),'truth_sha256':module.digest(truth),
                      'initial_error':record.get('metric_error'),'started_at':module.stamp()}
                request={'prediction_b64':base64.b64encode(step.read_bytes()).decode(),
                         'truth_b64':base64.b64encode(truth.read_bytes()).decode()}
                process=subprocess.Popen(commands,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,start_new_session=True)
                try:
                    output,error=process.communicate(json.dumps(request).encode(),timeout=300)
                    if process.returncode:
                        raise RuntimeError(error.decode(errors='replace')[-400:])
                    item['metrics']=json.loads(output)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.communicate()
                    item['error']='metric_retry_wall_timeout'
                except Exception as exc:
                    item['error']=type(exc).__name__+': '+str(exc)[:400]
                item['finished_at']=module.stamp();module.save(target,item)
                print(json.dumps({k:v for k,v in item.items() if k!='metrics'}),flush=True)
        if (folder/'summary.json').exists():break
        if (folder/'progress.json').exists():
            state=json.loads((folder/'progress.json').read_text())
            if state.get('status')=='stopped':
                module.save(out/'progress.json',{'status':'stopped_with_primary','updated_at':module.stamp()})
                return
        time.sleep(20)
    def effective(group,source):
        records=[]
        for ident in protocol['evaluated_ids']:
            record=json.loads((source/(ident+'.json')).read_text())
            retry=out/(group+'-'+ident+'.json')
            if retry.exists():
                resolved=json.loads(retry.read_text())
                if 'metrics' in resolved:
                    record['metrics']=resolved['metrics'];record['metric_resolution_file']=str(retry)
            records.append(record)
        return records
    baseline_records=effective('baseline',folder/'baseline-first-pass')
    results=effective('a2',folder/'samples')
    summary=json.loads((folder/'summary.json').read_text())
    summary.update(module.aggregate(results,44))
    summary['baseline']=module.aggregate(baseline_records,44)
    summary['metric_retry_protocol']={'cpu_seconds':120,'wall_seconds':300,
        'runtime_directory':str(runtime),'original_records_preserved':True,
        'note':'Longer budget applies only to trusted STEP comparison, never to generated CAD code.'}
    module.save(folder/'summary-resolved.json',summary)
    module.save(out/'progress.json',{'status':'completed','updated_at':module.stamp()})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
