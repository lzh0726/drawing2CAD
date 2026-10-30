"""CPU-only audit of new successful STEP files; no model inference or host CAD execution."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/luzhuohong/drawing2CAD');sys.path.insert(0,str(ROOT))
from server.research_sandbox import command,run
EVAL=ROOT/'experiments/entity-rate-current-20261027-141244'
OLD=ROOT/'experiments/finetuned-pilot-v1-repairs0'
OUT=ROOT/'experiments'/('success-geometry-audit-'+time.strftime('%Y%m%d-%H%M%S'))
OUT.mkdir();runtime=OUT/'runtime';runtime.mkdir()
shutil.copy2(ROOT/'tools/audit_step_shapes.py',runtime/'audit_step_shapes.py')
shutil.copy2(__file__,OUT/'audit.py')
def isolated(request):
    cmd=command();cmd[cmd.index(str(ROOT/'tools/cad-runtime'))]=str(runtime);cmd[-1]='/runner/audit_step_shapes.py'
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try:output,error=p.communicate(json.dumps(request).encode(),timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);p.communicate();raise RuntimeError('Audit_wall_timeout')
    if p.returncode:raise RuntimeError('Audit_child_failed_'+str(p.returncode)+': '+error.decode(errors='replace')[-400:])
    return json.loads(output)
assert all(run({'probe':True},15).values())
save=lambda path,data:path.write_text(json.dumps(data,indent=2))
save(OUT/'selftest.json',isolated({'selftest':True}))
rows={r['id']:r for r in map(json.loads,(ROOT/'datasets/pilot-v1/manifest.jsonl').read_text().splitlines())}
samples=[json.loads(p.read_text()) for p in EVAL.glob('test-*.json')];samples=[s for s in samples if s['valid_solid']]
assert len(samples)==34
paths=list(EVAL.glob('test-*'))+list(OLD.glob('*'))+[ROOT/'datasets/pilot-v1/manifest.jsonl']
def hashes():return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
before=hashes();save(OUT/'source-before.json',before)
records=[]
for sample in sorted(samples,key=lambda s:s['id']):
    ident=sample['id'];old=json.loads((OLD/(ident+'.json')).read_text())
    try:
        report=isolated({'prediction':base64.b64encode((EVAL/(ident+'.step')).read_bytes()).decode(),
                          'truth':base64.b64encode((ROOT/'datasets/pilot-v1'/rows[ident]['step']).read_bytes()).decode()})
        report.update(id=ident,previous_iou=old['metrics'],iou_source='Previous 2048-token evaluation; new first answer is byte-identical, no new Boolean IoU computed')
        rawold=(OLD/(ident+'-attempt1.txt')).read_bytes()
        rawnew=(ROOT/'server-storage'/sample['run_id']/'attempt-1.txt').read_bytes()
        report['first_answer_identical_to_old']=rawold==rawnew
    except Exception as e:report={'id':ident,'audit_error':str(e)}
    save(OUT/(ident+'.json'),report);records.append(report)
    save(OUT/'progress.json',{'completed':len(records),'total':34})
    print(json.dumps({'completed':len(records),'id':ident,'comparison':report.get('comparison'),'error':report.get('audit_error')}),flush=True)
after=hashes();save(OUT/'source-after.json',after);assert before==after
ok=[v for v in records if 'comparison' in v]
summary={'sample_count':34,'audited_count':len(ok),'audit_errors':[v for v in records if 'audit_error' in v],
         'valid_single_solids':sum(v['prediction']['valid'] and v['prediction']['solid_count']==1 for v in ok),
         'all_shells_closed_count':sum(v['prediction']['all_shells_closed'] for v in ok),
         'bbox_xyz_within_1pct':sum(v['comparison']['bbox_max_relative_xyz']<=.01 for v in ok),
         'bbox_xyz_within_5pct':sum(v['comparison']['bbox_max_relative_xyz']<=.05 for v in ok),
         'sorted_bbox_within_1pct':sum(v['comparison']['sorted_bbox_max_relative']<=.01 for v in ok),
         'sorted_bbox_within_5pct':sum(v['comparison']['sorted_bbox_max_relative']<=.05 for v in ok),
         'volume_within_5pct':sum(v['comparison']['volume_relative_error']<=.05 for v in ok),
         'genus_comparable_count':sum(v['comparison']['surface_genus_match'] is not None for v in ok),
         'genus_mismatch_ids':[v['id'] for v in ok if v['comparison']['surface_genus_match'] is False],
         'shell_count_mismatch_ids':[v['id'] for v in ok if not v['comparison']['shell_count_match']],
         'all_new_first_answers_identical_to_old':all(v['first_answer_identical_to_old'] for v in ok),
         'old_mean_iou_raw_44':sum(v['previous_iou']['iou_raw'] for v in ok)/44,
         'old_mean_iou_aligned_44':sum(v['previous_iou']['iou_translation_aligned'] for v in ok)/44,
         'old_mean_iou_raw_successes':sum(v['previous_iou']['iou_raw'] for v in ok)/len(ok),
         'old_mean_iou_aligned_successes':sum(v['previous_iou']['iou_translation_aligned'] for v in ok)/len(ok),
         'gpu_started':False,'original_sources_unchanged':True,'directory':str(OUT)}
save(OUT/'summary.json',summary);print('COMPLETE '+json.dumps(summary),flush=True)
