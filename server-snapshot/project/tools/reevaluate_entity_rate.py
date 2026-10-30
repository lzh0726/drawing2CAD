"""Fixed-test evaluation through the deployed Agent; never executes CAD on host."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile

ROOT = Path('/home/luzhuohong/drawing2CAD')
RUN_OUTPUT = None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def summarize(records, ids):
    assert len({r['id'] for r in records}) == len(records)
    assert all(r['id'] in ids for r in records)
    first = sum(r['first_pass_valid'] for r in records)
    final = sum(r['valid_solid'] for r in records)
    complete = len(records) == len(ids)
    return {'status': 'completed' if complete else 'running', 'completed_count': len(records),
            'valid_ground_truth_count': len(ids), 'first_pass_valid_count': first,
            'repair_added_valid_count': final - first, 'final_valid_count': final,
            'first_pass_rate': first / len(ids) if complete else None,
            'final_valid_rate': final / len(ids) if complete else None,
            'baseline_valid_count': 34, 'baseline_denominator': 44,
            'pending_ids': [i for i in ids if i not in {r['id'] for r in records}],
            'operational_failure_ids': [r['id'] for r in records if r.get('operational_failure')],
            'updated_at': stamp()}


def parse_outcome(status, body, trace):
    attempts = trace.get('attempts', [])
    first = bool(attempts and attempts[0].get('status') == 'passed')
    if status == 200:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            generation = json.loads(archive.read('generation.json'))
            step = archive.read('model.step')
            code = archive.read('model.py')
        assert trace.get('status') == 'completed'
        assert generation['geometry']['valid'] is True and step.startswith(b'ISO-10303-21;')
        return {'valid_solid': True, 'first_pass_valid': first, 'attempts': attempts,
                'repair_count': generation['repair_count'], 'geometry': generation['geometry']}, step, code
    detail = json.loads(body).get('detail', {})
    assert status == 422 and isinstance(detail, dict)
    assert detail.get('code') in ('cad_generation_failed', 'needs_clarification')
    return {'valid_solid': False, 'first_pass_valid': False, 'attempts': attempts,
            'failure': detail['code']}, None, None


def main():
    global RUN_OUTPUT
    import fcntl
    import httpx
    from dotenv import dotenv_values
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    assert folder.parent == (ROOT/'experiments').resolve()
    folder.mkdir(exist_ok=True)
    RUN_OUTPUT = folder
    lock = (ROOT/'outputs/entity-evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = dotenv_values(ROOT/'server/server.env')
    assert cfg.get('CAD_MAX_REPAIRS') == '1'
    headers = {'Authorization': 'Bearer ' + cfg['CAD_SERVER_TOKEN']}
    source = ROOT/'experiments/a1-1k-r16/finetuned-pilot-v1-repairs0'
    data = ROOT/'datasets/pilot-v1'
    manifest = data/'manifest.jsonl'
    all_rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    tests = [r for r in all_rows if r['split'] == 'test']
    rows = [r for r in tests if r['geometry']['valid']]
    assert len(tests) == 50 and len(rows) == 44
    ids = [r['id'] for r in rows]
    def hashes():
        files = list(source.glob('*')) + [manifest]
        files += [data/r['drawing'] for r in rows]
        files += [ROOT/n for n in ['server/model.py','server/app.py','server/demand_model.py',
                  'server/gpu_sharing.py','server/bubblewrap_executor.py','server/research_sandbox.py',
                  'tools/cad-runtime/research_worker.py','tools/cad-runtime/cad_diagnostics.py']]
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    before = hashes()
    protocol = {'started_at': stamp(), 'fixed_test_count': 50, 'evaluated_ids': ids,
                'excluded_invalid_ground_truth_ids': [r['id'] for r in tests if not r['geometry']['valid']],
                'max_new_tokens': 16384, 'generation_max_seconds': 600, 'max_repairs': 1,
                'cad_wall_seconds': 90, 'diagnostic_wall_seconds': 45,
                'model': cfg.get('CAD_MODEL'), 'endpoint': 'http://127.0.0.1:8001',
                'gpu_policy': 'Existing per-generation shared/idle admission; explicit user authorization',
                'input_policy': 'Only original drawing and original instructions; no reference code, STEP, or manual repair',
                'metric': 'Valid single solid accepted by deployed isolated executor; not drawing correctness',
                'comparison_note': 'Old 34/44 used 2048 tokens and a shorter generation/CAD budget; this run measures the changed pipeline, not a new checkpoint',
                'source_sha256': before}
    if (folder/'protocol.json').exists():
        old = json.loads((folder/'protocol.json').read_text())
        assert old['source_sha256'] == before, 'Protocol drift: do not mix runs'
    else:
        save(folder/'protocol.json', protocol)
    records = [json.loads(p.read_text()) for p in sorted(folder.glob('test-*.json'))]
    progress = summarize(records, ids)
    save(folder/'progress.json', progress)
    with httpx.Client(timeout=2400, trust_env=False, follow_redirects=False) as client:
        for row in rows:
            if any(r['id'] == row['id'] for r in records):
                continue
            ident = row['id']; started = time.monotonic(); network_retries = 0; model_errors = 0
            while True:
                progress = summarize(records, ids)
                progress.update(current_id=ident, current_started_at=stamp(), phase='requesting_agent')
                save(folder/'progress.json', progress)
                metadata = {'unit':'mm','instructions':row['instructions'],'messages':[]}
                try:
                    response = client.post('http://127.0.0.1:8001/v1/reconstruct', headers=headers,
                        files={'drawing':('drawing.png',(data/row['drawing']).read_bytes(),'image/png')},
                        data={'metadata':json.dumps(metadata)})
                except httpx.HTTPError:
                    network_retries += 1
                    progress.update(phase='waiting_connection', network_retries=network_retries)
                    save(folder/'progress.json', progress)
                    if network_retries >= 3: raise RuntimeError('Connection unavailable; pending sample was not scored')
                    time.sleep(30); continue
                detail = response.json().get('detail', {}) if response.status_code != 200 else {}
                if not isinstance(detail, dict): detail = {}
                reason = detail.get('code')
                if response.status_code == 429 or reason == 'gpu_wait_timeout':
                    progress.update(phase='waiting_gpu_or_service', last_status=response.status_code)
                    save(folder/'progress.json', progress)
                    if time.monotonic()-started > 6*3600: raise RuntimeError('Resource wait exceeded six hours; sample not scored')
                    time.sleep(30); continue
                run_id = response.headers.get('X-CAD-Run-ID') or detail.get('run_id')
                trace = {}
                if run_id and re.fullmatch('[0-9a-f]{32}', run_id):
                    trace_path = ROOT/'server-storage'/run_id/'trace.json'
                    if trace_path.exists(): trace = json.loads(trace_path.read_text())
                if response.status_code == 503 and reason == 'model_error':
                    model_errors += 1
                    save(folder/f'{ident}-operational-{model_errors}.trace', trace)
                    if model_errors < 3:
                        time.sleep(30); continue
                    record = {'valid_solid':False,'first_pass_valid':False,'operational_failure':True,
                              'failure':'model_error_after_3_requests','attempts':trace.get('attempts',[])}
                else:
                    if response.status_code not in (200,422):
                        raise RuntimeError(f'Agent unavailable: HTTP {response.status_code}; sample not scored')
                    record,step,code = parse_outcome(response.status_code,response.content,trace)
                    if step:
                        (folder/f'{ident}.step').write_bytes(step)
                        (folder/f'{ident}.py').write_bytes(code)
                record.update(id=ident,run_id=run_id,seconds=round(time.monotonic()-started,2),
                              model_error_retries=model_errors,trace=trace)
                save(folder/f'{ident}.json',record)
                records.append(record)
                progress=summarize(records,ids);save(folder/'progress.json',progress)
                print(json.dumps({'completed':len(records),'total':44,'id':ident,'first_pass':record['first_pass_valid'],
                                  'valid':record['valid_solid'],'first_pass_count':progress['first_pass_valid_count'],
                                  'final_count':progress['final_valid_count']},ensure_ascii=False),flush=True)
                break
    summary=summarize(records,ids)
    after=hashes();save(folder/'source-sha256-after.json',after)
    summary['sources_unchanged']=before==after
    baseline={json.loads(p.read_text())['id']:json.loads(p.read_text())['valid_solid'] for p in source.glob('test-*.json')}
    summary['newly_valid_ids']=[r['id'] for r in records if r['valid_solid'] and not baseline[r['id']]]
    summary['regressed_ids']=[r['id'] for r in records if not r['valid_solid'] and baseline[r['id']]]
    save(folder/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        # Do not log HTTP headers, bodies, credentials or environment values.
        if RUN_OUTPUT is not None and (RUN_OUTPUT/'progress.json').exists():
            progress=json.loads((RUN_OUTPUT/'progress.json').read_text())
            progress.update(status='stopped',failure_type=type(exc).__name__,updated_at=stamp())
            save(RUN_OUTPUT/'progress.json',progress)
        print('EVALUATION_STOPPED '+type(exc).__name__,flush=True)
        raise SystemExit(1)
