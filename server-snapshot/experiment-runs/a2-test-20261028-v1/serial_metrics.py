"""Uniform serial OCC comparison, isolated from host and model generation."""
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
    out=folder/'serial-metrics';out.mkdir(exist_ok=False)
    runtime=out/'runtime';runtime.mkdir()
    original=ROOT/'tools/cad-runtime'
    source=(original/'research_worker.py').read_text()
    assert source.count('intersection=a.intersect(b).Volume()')==1
    source=source.replace('resource.setrlimit(resource.RLIMIT_CPU, (35, 35))',
                          'resource.setrlimit(resource.RLIMIT_CPU, (120, 120))',1)
    source=source.replace('intersection=a.intersect(b).Volume()',
        'from OCP.BRepAlgoAPI import BRepAlgoAPI_Common\n'
        '            args=a if isinstance(a,cq.Compound) else (a,)\n'
        '            intersection=a._bool_op(args,(b,),BRepAlgoAPI_Common(),parallel=False).Volume()',1)
    compile(source,'research_worker.py','exec')
    (runtime/'research_worker.py').write_text(source)
    (runtime/'policy.py').write_bytes((original/'policy.py').read_bytes())
    commands=command();commands[commands.index(str(original))]=str(runtime)
    rows={r['id']:r for r in [json.loads(l) for l in (module.DATA/'manifest.jsonl').read_text().splitlines()]}
    protocol=json.loads((folder/'protocol.json').read_text())
    module.save(out/'protocol.json',{'cpu_seconds':120,'wall_seconds':180,'memory_gib':6,
        'parallel':False,'formula':'Unchanged exact Boolean intersection/union; original and bbox-center translated coordinates',
        'scope':'Compare saved STEP only, never run generated CAD code',
        'worker_sha256':module.digest(runtime/'research_worker.py')})
    while True:
        for group, source_dir in [('baseline',folder/'baseline-first-pass'),('a2',folder/'samples')]:
            for path in sorted(source_dir.glob('test-*.json')):
                record=json.loads(path.read_text())
                if not record['valid_solid']:continue
                ident=record['id'];target=out/(group+'-'+ident+'.json')
                if target.exists():continue
                module.save(out/'progress.json',{'status':'running','group':group,'id':ident,'updated_at':module.stamp()})
                step=(module.BASELINE if group=='baseline' else source_dir)/(ident+'.step')
                truth=module.DATA/rows[ident]['step']
                item={'id':ident,'group':group,'parallel':False,'started_at':module.stamp(),
                      'prediction_sha256':module.digest(step),'truth_sha256':module.digest(truth)}
                request={'prediction_b64':base64.b64encode(step.read_bytes()).decode(),
                         'truth_b64':base64.b64encode(truth.read_bytes()).decode()}
                started=time.monotonic()
                process=subprocess.Popen(commands,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,start_new_session=True)
                try:
                    output,error=process.communicate(json.dumps(request).encode(),timeout=180)
                    if process.returncode:raise RuntimeError(error.decode(errors='replace')[-500:])
                    item['metrics']=json.loads(output)
                    if 'metrics' in record:
                        item['delta_from_initial']={k:item['metrics'][k]-record['metrics'][k]
                                                   for k in ['iou_raw','iou_translation_aligned']}
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.communicate();item['error']='serial_metric_timeout'
                except Exception as exc:
                    item['error']=type(exc).__name__+': '+str(exc)[:500]
                item.update(seconds=round(time.monotonic()-started,3),finished_at=module.stamp())
                module.save(target,item)
                print(json.dumps({'group':group,'id':ident,'seconds':item['seconds'],'error':item.get('error'),
                                  'metrics':item.get('metrics')}),flush=True)
        if (folder/'summary-final.json').exists():break
        if json.loads((folder/'progress.json').read_text()).get('status')=='stopped':
            raise RuntimeError('Primary evaluation stopped; final metrics not produced')
        time.sleep(10)
    summaries={}
    deltas=[]
    for group, source_dir in [('baseline',folder/'baseline-first-pass'),('a2',folder/'samples')]:
        records=[]
        for ident in protocol['evaluated_ids']:
            record=json.loads((source_dir/(ident+'.json')).read_text())
            if record['valid_solid']:
                record.pop('metrics',None)
                metric=json.loads((out/(group+'-'+ident+'.json')).read_text())
                if 'metrics' in metric:record['metrics']=metric['metrics']
                deltas.extend(abs(v) for v in metric.get('delta_from_initial',{}).values())
            records.append(record)
        summaries[group]=module.aggregate(records,44)
    summary=json.loads((folder/'summary-final.json').read_text())
    summary.update(summaries['a2'],baseline=summaries['baseline'],baseline_historical_metric_reuse_ids=[],
        metric_execution_mode='All valid A1/A2 STEP compared with serial OCC Boolean operations in Bubblewrap',
        serial_control_max_abs_delta=max(deltas,default=0),serial_control_comparison_count=len(deltas),
        metrics_finished_at=module.stamp())
    module.save(folder/'summary-serial.json',summary)
    spec=importlib.util.spec_from_file_location('a2_report',folder/'make_report.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report);report.main(folder)
    module.save(out/'progress.json',{'status':'completed','updated_at':module.stamp()})
    module.save(folder/'report-status.json',{'status':'completed','updated_at':module.stamp(),
        'report':str(folder/'REPORT.md'),'metrics':str(folder/'summary-serial.json')})


if __name__=='__main__':main(Path(sys.argv[1]))
