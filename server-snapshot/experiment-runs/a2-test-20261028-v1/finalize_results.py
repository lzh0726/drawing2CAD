"""Finish the report when evaluation and isolated metric retries complete."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def data_hash(path):
    text=path.read_text()
    assert 'DATA;' in text
    return hashlib.sha256(text[text.index('DATA;'):].encode()).hexdigest()


def main(folder):
    spec=importlib.util.spec_from_file_location('a2_eval',folder/'evaluate_a2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    while not (folder/'summary-resolved.json').exists():
        if (folder/'progress.json').exists():
            state=json.loads((folder/'progress.json').read_text())
            if state['status']=='stopped':
                module.save(folder/'report-status.json',{'status':'primary_stopped','updated_at':module.stamp()})
                return
        time.sleep(20)
    summary=json.loads((folder/'summary-resolved.json').read_text())
    baseline_records=[]
    proof_dir=folder/'metric-cache-equivalence';proof_dir.mkdir(exist_ok=True)
    for path in sorted((folder/'baseline-first-pass').glob('test-*.json')):
        record=json.loads(path.read_text()); ident=record['id']
        retry=folder/'metric-retries'/('baseline-'+ident+'.json')
        if retry.exists():
            result=json.loads(retry.read_text())
            if 'metrics' in result:record['metrics']=result['metrics']
        if record['valid_solid'] and 'metrics' not in record:
            old=module.ROOT/'experiments/finetuned-pilot-v1-repairs0'
            current_step=module.BASELINE/(ident+'.step');old_step=old/(ident+'.step')
            old_record=old/(ident+'.json')
            if old_step.exists() and old_record.exists() and data_hash(current_step)==data_hash(old_step):
                cached=json.loads(old_record.read_text())
                if 'metrics' in cached:
                    proof={'id':ident,'metrics':cached['metrics'],'source':str(old_record),
                        'source_record_sha256':module.digest(old_record),
                        'step_data_section_sha256':data_hash(current_step),
                        'current_step':str(current_step),'historical_step':str(old_step),
                        'method':'Historical metric reuse after exact STEP DATA-section equivalence; not a new measurement',
                        'initial_and_extended_timeout_records_preserved':True}
                    module.save(proof_dir/('baseline-'+ident+'.json'),proof)
                    record['metrics']=cached['metrics']
        baseline_records.append(record)
    summary['baseline']=module.aggregate(baseline_records,44)
    summary['baseline_historical_metric_reuse_ids']=[json.loads(f.read_text())['id'] for f in proof_dir.glob('*.json')]
    module.save(folder/'summary-final.json',summary)
    spec=importlib.util.spec_from_file_location('a2_report',folder/'make_report.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    report.main(folder)
    module.save(folder/'report-status.json',{'status':'completed','updated_at':module.stamp(),
        'report':str(folder/'REPORT.md'),'metrics':str(folder/'summary-final.json')})


if __name__=='__main__':main(Path(sys.argv[1]))
