"""Independent A2 evaluation. Candidate CAD executes only through Bubblewrap."""
import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/home/luzhuohong/drawing2CAD')
TRAIN = Path('/data/luzhuohong/drawing2CAD-experiments/a2-10k-r16/output')
DATA = ROOT / 'datasets/pilot-v1'
BASELINE = ROOT / 'experiments/entity-rate-current-20261027-141244'
sys.path.insert(0, str(ROOT))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def aggregate(records, expected):
    assert len({r['id'] for r in records}) == len(records)
    valid = [r for r in records if r['valid_solid']]
    scored = [r for r in valid if 'metrics' in r]
    errors = [r['id'] for r in valid if 'metrics' not in r]
    complete = len(records) == expected
    result = {'completed_count': len(records), 'test_count': expected,
              'valid_solid_count': len(valid), 'invalid_count': len(records)-len(valid),
              'pending_count': expected-len(records),
              'valid_solid_rate': len(valid)/expected if complete else None,
              'metric_error_ids': errors,
              'bbox_xyz_within_1pct_count': sum(max(r['metrics']['bbox_relative']) <= .01 for r in scored),
              'bbox_xyz_within_5pct_count': sum(max(r['metrics']['bbox_relative']) <= .05 for r in scored)}
    for metric in ['iou_raw', 'iou_translation_aligned']:
        values = [r['metrics'][metric] for r in scored]
        result['mean_'+metric+'_successful_comparable'] = statistics.mean(values) if values else None
        result['mean_'+metric+'_all'] = sum(values)/expected if complete and not errors else None
    return result


def self_test():
    bad = {'id':'b', 'valid_solid':False}
    good = {'id':'a', 'valid_solid':True, 'metrics':{
        'iou_raw':.4, 'iou_translation_aligned':.8, 'bbox_relative':[0, 0, 0]}}
    assert aggregate([good], 2)['valid_solid_rate'] is None
    result = aggregate([good, bad], 2)
    assert result['valid_solid_rate'] == .5 and result['mean_iou_raw_all'] == .2
    result = aggregate([{'id':'a', 'valid_solid':True}, bad], 2)
    assert result['mean_iou_raw_all'] is None and result['metric_error_ids'] == ['a']
    try:
        aggregate([bad, bad], 2)
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate samples accepted')
    print('SUMMARY_SELF_TEST_PASSED', flush=True)


def merge_cpu(folder):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel
    torch.set_num_threads(4)
    assert not torch.cuda.is_available(), 'CPU export must not see CUDA'
    base = ROOT/'models/qwen3-vl-8b-instruct'
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base), device_map='cpu', torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(model, str(TRAIN), is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    assert next(model.parameters()).device.type == 'cpu'
    target = folder/'merged-model'
    target.mkdir(exist_ok=False)
    model.save_pretrained(target, max_shard_size='4GB', safe_serialization=True)
    AutoProcessor.from_pretrained(str(base), trust_remote_code=False).save_pretrained(target)
    save(folder/'merge-complete.json', {'completed_at':stamp(), 'device':'cpu',
        'adapter_sha256':digest(TRAIN/'adapter_model.safetensors'),
        'model':str(target), 'gpu_started':False})


def main(folder):
    from server.demand_model import DemandModel, GPUWaitTimeout
    from server.policy import extract_code, validate_code
    from server.research_sandbox import run
    folder = folder.resolve()
    assert folder.parent == Path('/data/luzhuohong/drawing2CAD-experiments')
    folder.mkdir(exist_ok=True)
    lock = (ROOT/'outputs/entity-evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    own_lock = (folder/'run.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = folder/'progress.json'
    state = {'status':'running', 'started_at':stamp(), 'pid':os.getpid(), 'phase':'preflight'}

    def progress(**changes):
        state.update(changes, updated_at=stamp())
        save(state_path, state)

    try:
        progress()
        rows_all = [json.loads(l) for l in (DATA/'manifest.jsonl').read_text().splitlines()]
        tests = [r for r in rows_all if r['split']=='test']
        rows = [r for r in tests if r['geometry']['valid']]
        assert len(tests)==50 and len(rows)==44
        expanded = [json.loads(l) for l in Path('/data/luzhuohong/drawing2CAD-datasets/pilot-10k-v1/manifest.jsonl').read_text().splitlines()]
        training = [r for r in expanded if r['split']=='train']
        assert len(training)==10000
        assert not ({r['id'] for r in tests} & {r['id'] for r in training})
        assert not ({r['group'] for r in tests} & {r['group'] for r in training})
        assert digest(TRAIN/'adapter_model.safetensors') == digest(TRAIN/'checkpoint-1200/adapter_model.safetensors')
        probe = run({'probe':True})
        assert probe and all(probe.values()), 'Isolation probe failed'
        baseline_protocol = json.loads((BASELINE/'protocol.json').read_text())
        assert baseline_protocol['max_new_tokens']==16384 and baseline_protocol['generation_max_seconds']==600
        assert baseline_protocol['evaluated_ids']==[r['id'] for r in rows]
        sources = [DATA/'manifest.jsonl', TRAIN/'adapter_model.safetensors', TRAIN/'adapter_config.json']
        sources += [ROOT/n for n in ['server/model.py','server/demand_model.py','server/gpu_sharing.py',
                    'server/research_sandbox.py','tools/cad-runtime/research_worker.py','server/policy.py']]
        for row in rows:
            sources.extend([DATA/row['drawing'], DATA/row['step'], BASELINE/(row['id']+'.json')])
            cached = json.loads((BASELINE/(row['id']+'.json')).read_text())
            assert not cached.get('operational_failure')
            if cached['first_pass_valid']:
                assert cached['repair_count']==0
                sources.append(BASELINE/(row['id']+'.step'))
        before = {str(p):digest(p) for p in sources}
        protocol = {'created_at':stamp(), 'model':'A2 10k rank16, best checkpoint1200',
            'adapter':str(TRAIN), 'max_new_tokens':16384, 'generation_max_seconds':600,
            'max_repairs':0, 'image_max_pixels':262144, 'do_sample':False,
            'cad_wall_seconds':90, 'metric_wall_seconds':90, 'cad_cpu_seconds':35,
            'fixed_test_count':50, 'denominator':44, 'evaluated_ids':[r['id'] for r in rows],
            'excluded_invalid_ground_truth_ids':[r['id'] for r in tests if not r['geometry']['valid']],
            'input_policy':'Only original drawing and instructions; no ground truth code/STEP or repair',
            'gpu_policy':'Existing DemandModel admission and shared-memory protection; release after each generation',
            'metric_policy':'Exact Boolean IoU raw and bbox-center-translation; no rotation or scale normalization',
            'baseline_policy':'Reuse A1 current-budget first-pass outcomes and STEP only; ignore repair outcomes; recompute metrics with the same CPU executor',
            'baseline_source':str(BASELINE), 'isolation_probe':probe, 'source_sha256':before,
            'note':'The 44-case set is a diagnosed regression set, not a fresh generalization test.'}
        if (folder/'protocol.json').exists():
            previous = json.loads((folder/'protocol.json').read_text())
            assert previous['source_sha256']==before, 'Source drift; refuse mixed run'
        else:
            save(folder/'protocol.json', protocol)
        if not (folder/'merge-complete.json').exists():
            progress(phase='cpu_merge')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1')
            with (folder/'merge.log').open('a') as log:
                process = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--merge-cpu',str(folder)],
                    stdout=log, stderr=subprocess.STDOUT, env=env)
                while process.poll() is None:
                    progress(phase='cpu_merge', cpu_merge_pid=process.pid)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                assert process.returncode==0, 'CPU model export failed'
        model = DemandModel(str(folder/'merged-model'), None, wait_seconds=300, inference_seconds=720)
        model.load()

        def metrics(step, row):
            return run({'prediction_b64':base64.b64encode(step).decode(),
                        'truth_b64':base64.b64encode((DATA/row['step']).read_bytes()).decode()}, timeout=90)

        baseline_folder = folder/'baseline-first-pass'
        baseline_folder.mkdir(exist_ok=True)
        baseline_records=[]
        for row in rows:
            path=baseline_folder/(row['id']+'.json')
            progress(phase='baseline_cpu_metrics', current_id=row['id'], baseline_completed=len(baseline_records))
            if path.exists():
                record=json.loads(path.read_text())
            else:
                old=json.loads((BASELINE/(row['id']+'.json')).read_text())
                record={'id':row['id'], 'valid_solid':old['first_pass_valid'], 'source_run_id':old['run_id']}
                if record['valid_solid']:
                    try:
                        record['metrics']=metrics((BASELINE/(row['id']+'.step')).read_bytes(),row)
                    except Exception as exc:
                        record['metric_error']=type(exc).__name__+': '+str(exc)[:300]
                save(path,record)
            baseline_records.append(record)
        baseline_summary=aggregate(baseline_records,44)
        save(folder/'baseline-summary.json',baseline_summary)
        results=[]
        result_folder=folder/'samples'
        result_folder.mkdir(exist_ok=True)
        for row in rows:
            ident=row['id']; path=result_folder/(ident+'.json')
            if path.exists():
                results.append(json.loads(path.read_text()))
                continue
            began=time.monotonic(); model_errors=0; wait_started=time.monotonic()
            while True:
                progress(phase='waiting_or_generating', current_id=ident,
                         current_started_at=stamp(), **aggregate(results,44))
                try:
                    raw=model.generate((DATA/row['drawing']).read_bytes(),
                        {'instructions':row['instructions'],'messages':[]})
                    break
                except GPUWaitTimeout:
                    progress(phase='waiting_eligible_gpu',current_id=ident)
                    if time.monotonic()-wait_started > 6*3600:
                        raise RuntimeError('GPU resource wait exceeded six hours; sample was not scored')
                    time.sleep(10)
                except RuntimeError:
                    model_errors+=1
                    progress(phase='model_operational_retry', current_id=ident, model_error_retries=model_errors)
                    if model_errors>=3:
                        raise RuntimeError('Model operational failure; pending sample was not scored')
                    time.sleep(15)
            (result_folder/(ident+'-attempt1.txt')).write_text(raw,encoding='utf-8')
            record={'id':ident,'valid_ground_truth':True,'valid_solid':False,
                    'peak_gpu_memory_mb':model.last_peak_memory_mb,'max_repairs':0,
                    'model_error_retries':model_errors,'attempts':[{'number':1}]}
            progress(phase='isolated_cad_execution', current_id=ident)
            try:
                if raw.strip().startswith('NEEDS_CLARIFICATION:'):
                    record['attempts'][0].update(status='needs_clarification')
                else:
                    code=extract_code(raw); validate_code(code)
                    (result_folder/(ident+'.py')).write_text(code,encoding='utf-8')
                    step,geometry=run({'code':code},timeout=90)
                    (result_folder/(ident+'.step')).write_bytes(step)
                    record.update(valid_solid=True,geometry=geometry)
                    record['attempts'][0].update(status='passed')
            except (ValueError,SyntaxError,RuntimeError) as exc:
                record['attempts'][0].update(status='failed',error=type(exc).__name__+': '+str(exc)[:500])
            if record['valid_solid']:
                progress(phase='isolated_iou', current_id=ident)
                try:
                    record['metrics']=metrics(step,row)
                except Exception as exc:
                    record['metric_error']=type(exc).__name__+': '+str(exc)[:300]
            record['seconds']=round(time.monotonic()-began,3)
            save(path,record); results.append(record)
            progress(phase='sample_completed',current_id=ident,**aggregate(results,44))
            print(json.dumps({'completed':len(results),'total':44,'id':ident,'valid':record['valid_solid'],
                  'valid_count':state['valid_solid_count'],'seconds':record['seconds']},ensure_ascii=False),flush=True)
        summary=aggregate(results,44)
        after={str(p):digest(p) for p in sources}
        save(folder/'source-sha256-after.json',after)
        summary.update(status='completed',finished_at=stamp(),sources_unchanged=before==after,
            baseline=baseline_summary, protocol_file=str(folder/'protocol.json'),
            newly_valid_ids=[r['id'] for r,b in zip(results,baseline_records) if r['valid_solid'] and not b['valid_solid']],
            regressed_ids=[r['id'] for r,b in zip(results,baseline_records) if not r['valid_solid'] and b['valid_solid']],
            mean_seconds=statistics.mean(r['seconds'] for r in results))
        assert summary['sources_unchanged'], 'Original source changed during evaluation'
        save(folder/'summary.json',summary)
        progress(phase='completed',**summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as exc:
        progress(status='stopped',failure_type=type(exc).__name__,failure=str(exc)[:400])
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--merge-cpu',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    elif args.merge_cpu:
        merge_cpu(args.merge_cpu)
    else:
        assert args.output is not None
        main(args.output)
