"""Complete already-running preparation, then export and verify artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=ROOT/'datasets/pilot-v1/manifest.jsonl'
    deadline=time.monotonic()+7200
    while not manifest.exists():
        if time.monotonic()>deadline: raise TimeoutError('Dataset preparation did not finish')
        processes=subprocess.check_output(['pgrep','-af','tools/prepare_pilot.py'],text=True)
        if not processes.strip(): raise RuntimeError('Preparation stopped without manifest')
        time.sleep(15)
    rows=[json.loads(l) for l in manifest.read_text().splitlines()]
    groups={};images={}
    for r in rows:
        split=r['split'];group=r['group']
        assert group not in groups or groups[group]==split,'cross-split family'
        groups[group]=split
        image=hashlib.sha256((manifest.parent/r['drawing']).read_bytes()).hexdigest()
        assert image not in images or images[image]==split,'cross-split identical drawing'
        images[image]=split
        assert not r['human_reviewed'] and r['license_status']=='unresolved'
        if not r['geometry']['valid']: r['validation_status']='official_truth_invalid_or_multiple_solids'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    subprocess.run(['/home/luzhuohong/miniconda3/envs/drawing2cad-train/bin/python','tools/export_pilot_sft.py'],cwd=ROOT,check=True)
    baseline=ROOT/'experiments/a1-1k-r16/baseline-pilot-v1'
    while not (baseline/'summary.json').exists():
        if time.monotonic()>deadline: raise TimeoutError('Baseline did not finish')
        time.sleep(15)
    selected=[r for r in rows if r['split']=='test']
    eligible=[r for r in selected if r['geometry']['valid']]
    records=[json.loads((baseline/(r['id'].removeprefix('test-')+'.json')).read_text()) for r in eligible]
    metric_errors=[r['id'] for r in records if r['valid_solid'] and 'metrics' not in r]
    evaluable=[r for r in records if not r['valid_solid'] or 'metrics' in r]
    report={'fixed_test_count':len(selected),'valid_ground_truth_count':len(eligible),'invalid_ground_truth_ids':[r['id'] for r in selected if not r['geometry']['valid']],'metric_error_ids':metric_errors,'evaluable_count':len(evaluable),'valid_generated_solids_on_valid_gt':sum(r['valid_solid'] for r in records),'mean_iou_raw_evaluable':sum(r.get('metrics',{}).get('iou_raw',0) for r in evaluable)/len(evaluable) if evaluable else None,'mean_iou_translation_aligned_evaluable':sum(r.get('metrics',{}).get('iou_translation_aligned',0) for r in evaluable)/len(evaluable) if evaluable else None,'generation_failure_score':0,'invalid_gt_and_metric_errors_excluded':True}
    (baseline/'reviewed-summary.json').write_text(json.dumps(report,indent=2))
    (ROOT/'experiments/a1-1k-r16/pilot-completion.json').write_text(json.dumps({'data_export_complete':True,'baseline_complete':True,'training_started':False,'license_status':'unresolved','human_reviewed':False,'checks':['cross-split family','cross-split identical image','review status']},indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
