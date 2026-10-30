"""Extend the immutable A2 10k set by 20k CPU-checked samples for A4."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import ast
import base64
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile

ROOT = Path('/home/luzhuohong/drawing2CAD')
HERE = Path(__file__).resolve().parent
OUT = Path('/data/luzhuohong/drawing2CAD-datasets/pilot-30k-v1')
OLD = Path('/data/luzhuohong/drawing2CAD-datasets/pilot-10k-v1')
sys.path.insert(0, str(ROOT))
from server.research_sandbox import command, run
from server.policy import validate_code
from tools.prepare_pilot import fingerprints
from server.model import SYSTEM, task_prompt


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class Sandbox:
    def __init__(self):
        self.process = None
        self.count = 0

    def close(self):
        if self.process:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
                try: self.process.wait(3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL); self.process.wait()
            self.process.stdin.close(); self.process.stdout.close(); self.process.stderr.close()
            self.process = None

    def call(self, request):
        if self.process is None or self.count >= 64 or self.process.poll() is not None:
            self.close()
            cmd = command()
            cmd[cmd.index(str(ROOT / 'tools/cad-runtime'))] = str(HERE / 'runtime')
            cmd[-1] = '/runner/batch_worker.py'
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            self.count = 0
        self.process.stdin.write(json.dumps(request).encode() + b'\n'); self.process.stdin.flush()
        self.count += 1
        if not select.select([self.process.stdout], [], [], 70)[0]:
            self.close(); raise RuntimeError('sandbox_transport_timeout')
        line = self.process.stdout.readline()
        if not line:
            self.close(); raise RuntimeError('sandbox_transport_closed')
        response = json.loads(line)
        if not response['ok']: raise ValueError(response['error'])
        data = base64.b64decode(response['data'])
        if request.get('probe'): return json.loads(data)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return z.read('model.step'), json.loads(z.read('geometry.json'))


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    sandbox = Sandbox()
    try:
        assert all(sandbox.call({'probe': True}).values())
        # Check exact same geometry as the original executor and isolation after failure.
        code = "import cadquery as cq\nresult=cq.Workplane('XY').box(8,6,2)\n"
        _, original = run({'code': code})
        _, batched = sandbox.call({'code': code})
        assert original == batched
        try: sandbox.call({'code': "import os\nresult=1\n"})
        except ValueError: pass
        else: raise AssertionError('Unsafe code accepted')
        _, again = sandbox.call({'code': code})
        assert again == original
        print('CPU_SANDBOX_SMOKE_PASSED', flush=True)
    finally: sandbox.close()
    if args.smoke: return
    OUT.mkdir(parents=True, exist_ok=False)
    for name in ('images', 'codes', 'steps', 'sft'): (OUT/name).mkdir()
    report = {'status': 'running', 'target_train': 30000, 'retained_original_train': 10000, 'target_new_train': 20000, 'seed': 42,
              'cutoff_len': 4096, 'image_max_pixels': 262144, 'workers': 4, 'gpu_started': False,
              'human_reviewed': False, 'license_status': 'unresolved', 'exclusions': {}, 'started_unix': time.time()}
    previous = json.loads((OLD/'preparation-report.json').read_text())
    assert previous['status']=='completed'
    sources = [OLD/'manifest.jsonl', OLD/'sft/train.jsonl', OLD/'sft/val.jsonl', OLD/'preparation-report.json', ROOT/'datasets/raw/ortho_train_data.zip', ROOT/'datasets/raw/inference.zip']
    report['source_sha256'] = {str(p): sha(p) for p in sources}
    for name in ('ortho_train_data.zip','inference.zip'):
        key=str(ROOT/'datasets/raw'/name)
        assert report['source_sha256'][key]==previous['source_sha256'][key], 'Original archive changed'
    report['continuation_source_index']=previous['last_source_index']
    report['phase']='copying_existing_data'
    write_json(OUT/'progress.json', report)
    old_rows = [json.loads(l) for l in (OLD/'manifest.jsonl').read_text().splitlines()]
    assert Counter(r['split'] for r in old_rows) == {'train': 10000, 'val': 100, 'test': 50}
    manifest = []; seen_ids = set(); seen_exact = set(); blocked_families = set(); geometry_seen = set()
    source_file_hashes = {}
    for row in old_rows:
        for key in ('drawing', 'code', 'step'):
            src = OLD/row[key]; source_file_hashes[str(src)] = sha(src); shutil.copy2(src, OUT/row[key])
        manifest.append(row); seen_ids.add(row['id'])
        raw = (OLD/row['code']).read_text().removesuffix('\nresult = solid\n')
        exact, family = fingerprints(raw); assert family == row['group']
        seen_exact.add(exact)
        if row['split'] != 'train': blocked_families.add(family)
        if row['geometry'].get('valid'): geometry_seen.add(row['geometry']['geometric_signature'])
    write_json(OUT/'original-files-sha256.json', source_file_hashes)
    with (OLD/'sft/train.jsonl').open() as f:
        assert json.loads(next(f))['messages'][0]['content'] == SYSTEM, 'System prompt drift'
    # Reserve all 100 official test examples, including the 50 not in the pilot.
    sandbox = Sandbox()
    try:
        with zipfile.ZipFile(ROOT/'datasets/raw/inference.zip') as z:
            official = [json.loads(l) for l in z.read('inference/deepcad_test_data_subset100.jsonl').decode().splitlines()]
            for row in official:
                exact, family = fingerprints(row['ground_truth']); seen_exact.add(exact); blocked_families.add(family)
                stem = Path(row['image']).stem
                if 'test-'+stem in seen_ids: continue
                try:
                    _, g = sandbox.call({'step_b64': base64.b64encode(z.read(f'inference/test100_gt_steps_deepcad/{stem}.step')).decode()})
                    geometry_seen.add(g['geometric_signature'])
                except ValueError: pass
        report['official_test_reserved'] = len(official)
    finally: sandbox.close()
    from transformers import AutoProcessor, AutoTokenizer
    from llamafactory.hparams import DataArguments
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    model = str(ROOT/'models/qwen3-vl-8b-instruct')
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False, local_files_only=True)
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=False, local_files_only=True)
    processor.image_max_pixels = 262144
    data_args = DataArguments(template='qwen3_vl_nothink', cutoff_len=131072)
    encoder = SupervisedDatasetProcessor(get_template_and_fix_tokenizer(tokenizer, data_args), tokenizer, processor, data_args)
    counts = Counter(r['split'] for r in manifest); exclusions = Counter()
    manifest_stream = (OUT/'manifest.inprogress.jsonl').open('w')
    for row in manifest: manifest_stream.write(json.dumps(row)+'\n')
    manifest_stream.flush()
    train = (OUT/'sft/train.inprogress.jsonl').open('w')
    for split in ('train', 'val'):
        records = [json.loads(l) for l in (OLD/f'sft/{split}.jsonl').read_text().splitlines()]
        for rec in records:
            rec['images'] = [str(OUT/'images'/Path(p).name) for p in rec['images']]
        text = ''.join(json.dumps(r)+'\n' for r in records)
        if split == 'train': train.write(text); train.flush()
        else: (OUT/'sft/val.jsonl').write_text(text)
    thread_state = threading.local(); workers = []; pool = ThreadPoolExecutor(max_workers=4)
    def checked(item):
        if not hasattr(thread_state, 'sandbox'):
            thread_state.sandbox = Sandbox(); workers.append(thread_state.sandbox)
        try: return item, thread_state.sandbox.call({'code': item['code']}), None
        except (ValueError, RuntimeError) as exc: return item, None, str(exc)
    pending = deque()
    try:
        with zipfile.ZipFile(ROOT/'datasets/raw/ortho_train_data.zip') as z:
            rows = json.loads(z.read('ortho_train_data/cadquery_ortho_train_data.json')); random.Random(42).shuffle(rows)
            report.update(phase='selecting_new_data',counts=dict(counts),new_accepted=0)
            write_json(OUT/'progress.json',report)
            print('EXISTING_10000_RETAINED_STARTING_NEW_SELECTION',flush=True)
            def candidates():
                for index, r in enumerate(rows):
                    if index <= previous['last_source_index']: continue
                    sid = 'train-source-'+str(r['id'])
                    if sid in seen_ids: continue
                    if not all(c.isalnum() or c in '_-' for c in sid): raise ValueError('Unsafe source ID')
                    raw = r['conversations'][1]['value']
                    try: exact, family = fingerprints(raw)
                    except SyntaxError: exclusions['syntax'] += 1; continue
                    if family in blocked_families: exclusions['holdout_family'] += 1; continue
                    if int(family[:8],16)%10 == 0: exclusions['validation_family_partition'] += 1; continue
                    if exact in seen_exact: exclusions['duplicate_code'] += 1; continue
                    seen_exact.add(exact)
                    code = raw+'\nresult = solid\n'
                    try: validate_code(code)
                    except (ValueError, SyntaxError): exclusions['code_policy'] += 1; continue
                    yield {'id': sid, 'index': index, 'record': r, 'family': family, 'code': code}
            iterator = iter(candidates()); exhausted = False
            while counts['train'] < 30000:
                while len(pending) < 8 and not exhausted:
                    try: pending.append(pool.submit(checked, next(iterator)))
                    except StopIteration: exhausted = True
                if not pending: raise RuntimeError('Not enough qualifying training samples')
                item, evaluated, error = pending.popleft().result()
                report['last_source_index'] = item['index']
                if error: exclusions['cad_failure'] += 1; continue
                step, g = evaluated
                if any(abs(a-b)/max(abs(b),1e-7) > .005 for a,b in zip(g['broad_dimensions'],g['dimensions'])):
                    exclusions['bbox_annotation_disagreement'] += 1; continue
                if g['geometric_signature'] in geometry_seen: exclusions['geometry_signature_duplicate'] += 1; continue
                image = z.read('ortho_train_data/'+item['record']['image'])
                from PIL import Image
                with Image.open(io.BytesIO(image)) as picture:
                    if picture.format!='PNG' or picture.width*picture.height>9_000_000:
                        exclusions['image_format_or_size'] += 1; continue
                    picture.load()
                temporary = OUT/'candidate.png'; temporary.write_bytes(image)
                instructions = 'Generate CadQuery code for the single part shown in this drawing. Units are mm.'
                prompt = [{'role':'user','content':'<image>\n'+task_prompt({'instructions':instructions,'messages':[]})}]
                response = [{'role':'assistant','content':'```python\n'+item['code'].rstrip()+'\n```'}]
                ids, labels = encoder._encode_data_example(prompt,response,SYSTEM,None,[str(temporary)],[],[])
                if len(ids)>4096: exclusions['full_sequence_over_4096'] += 1; continue
                if not any(x!=-100 for x in labels): raise RuntimeError('No supervised assistant tokens')
                if shutil.disk_usage(OUT).free < 15*1024**3: raise RuntimeError('Data disk reserve below 15 GiB')
                sid = item['id']; image_path = 'images/'+sid+'.png'
                temporary.replace(OUT/image_path)
                (OUT/'codes'/f'{sid}.py').write_text(item['code'])
                (OUT/'steps'/f'{sid}.step').write_bytes(step)
                row = {'id':sid,'group':item['family'],'split':'train','drawing':image_path,'code':'codes/'+sid+'.py','step':'steps/'+sid+'.step','source':'Ortho2CAD training/'+item['record']['image'],'license':'not declared in upstream dataset','license_status':'unresolved','human_reviewed':False,'validation_status':'automated_cad_checked','geometry':g,'instructions':instructions,'full_tokens':len(ids),'source_shuffle_index':item['index']}
                manifest.append(row); seen_ids.add(sid); geometry_seen.add(g['geometric_signature']); counts['train'] += 1
                manifest_stream.write(json.dumps(row)+'\n'); manifest_stream.flush()
                train.write(json.dumps({'images':[str(OUT/image_path)],'messages':[{'role':'system','content':SYSTEM}]+prompt+response})+'\n'); train.flush()
                report.update(counts=dict(counts),new_accepted=counts['train']-10000,exclusions=dict(exclusions),elapsed_seconds=round(time.time()-report['started_unix'],1),max_new_full_tokens=max(report.get('max_new_full_tokens',0),len(ids)))
                if counts['train'] % 100 == 0:
                    write_json(OUT/'progress.json',report); print(json.dumps({'accepted':counts['train'],'elapsed_seconds':report['elapsed_seconds'],'source_index':item['index'],'excluded':dict(exclusions)}),flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for worker in workers: worker.close()
        manifest_stream.close(); train.close(); (OUT/'candidate.png').unlink(missing_ok=True)
    assert counts == {'train':30000,'val':100,'test':50}
    assert len(seen_ids) == 30150
    assert counts['train']-sum(r['split']=='train' for r in old_rows)==20000
    groups = {split:{r['group'] for r in manifest if r['split']==split} for split in counts}
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    assert not(groups['train']&blocked_families)
    assert report['source_sha256'] == {str(p):sha(p) for p in sources}
    assert source_file_hashes == {name:sha(Path(name)) for name in source_file_hashes}
    assert {r['id'] for r in old_rows if r['split']=='train'} <= {r['id'] for r in manifest if r['split']=='train'}
    for split in ('train','val','test'):
        if split!='train': assert [r for r in manifest if r['split']==split] == [r for r in old_rows if r['split']==split]
    old_by_id={r['id']:r for r in old_rows}
    assert all(row==old_by_id[row['id']] for row in manifest if row['id'] in old_by_id)
    exported=[json.loads(l) for l in (OUT/'sft/train.inprogress.jsonl').read_text().splitlines()]
    original_sft=[json.loads(l) for l in (OLD/'sft/train.jsonl').read_text().splitlines()]
    assert len(exported)==30000 and len(original_sft)==10000
    assert all(a['messages']==b['messages'] for a,b in zip(exported,original_sft))
    assert all(len(r['images'])==1 and Path(r['images'][0]).is_file() for r in exported)
    assert all(sum(m['content'].count('<image>') for m in r['messages'])==1 for r in exported)
    assert report['new_accepted']==20000
    (OUT/'manifest.inprogress.jsonl').replace(OUT/'manifest.jsonl')
    (OUT/'sft/train.inprogress.jsonl').replace(OUT/'sft/train.jsonl')
    shutil.copy2(OLD/'sft/dataset_info.json', OUT/'sft/dataset_info.json')
    report.update(status='completed',phase='completed',source_unchanged=True,original_training_subset_preserved=True,holdouts_unchanged=True,completed_unix=time.time())
    write_json(OUT/'preparation-report.json',report); write_json(OUT/'progress.json',report)
    print('COMPLETED '+str(OUT),flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException as exc:
        print('FAILED '+type(exc).__name__+': '+str(exc),flush=True)
        if (OUT/'progress.json').exists():
            progress=json.loads((OUT/'progress.json').read_text())
            progress.update(status='failed',failed_unix=time.time(),error_type=type(exc).__name__)
            write_json(OUT/'progress.json',progress)
        if OUT.exists(): write_json(OUT/'failure.json',{'type':type(exc).__name__,'message':str(exc),'time':time.time()})
        raise
