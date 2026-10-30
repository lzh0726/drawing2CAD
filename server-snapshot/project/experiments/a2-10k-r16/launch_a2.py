"""One-shot A2 admission check; scheduling is managed by the Codex heartbeat."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/luzhuohong/drawing2CAD')
STAGE = ROOT/'experiments/a2-10k-r16'
DATA = Path('/data/luzhuohong/drawing2CAD-datasets/pilot-10k-v1')
RUN = Path('/data/luzhuohong/drawing2CAD-experiments/a2-10k-r16')
CONFIG = STAGE/'train-a2-10k.yaml'
STATE = STAGE/'launch.json'
CLI = '/home/luzhuohong/miniconda3/envs/drawing2cad-train/bin/llamafactory-cli'
PYTHON = '/home/luzhuohong/miniconda3/envs/drawing2cad-train/bin/python'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible_from_text(info):
    result = []
    for line in info.splitlines():
        index, uuid, memory, util = [s.strip() for s in line.split(',')]
        if int(memory) < 5000 and int(util) < 10:
            result.append((int(index), uuid))
    return sorted(result)


def snapshot():
    info = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True, timeout=10)
    return eligible_from_text(info), info


def stable_gpu(history):
    common=set(history[0][0])
    for candidates,_ in history[1:]:common.intersection_update(candidates)
    for candidate in sorted(common):
        used=[]
        for _,info in history:
            row=next(line.split(',') for line in info.splitlines() if line.split(',')[1].strip()==candidate[1])
            used.append(int(row[2].strip()))
        if max(used)-min(used)<=512:return candidate
    return None


def validate():
    import yaml
    report_path = DATA/'preparation-report.json'
    if not report_path.exists(): return 'dataset_not_ready'
    report = json.loads(report_path.read_text())
    if report.get('status') != 'completed': return 'dataset_not_ready'
    for key in ('source_unchanged','holdouts_unchanged','original_training_subset_preserved'):
        assert report.get(key) is True, key
    manifest = [json.loads(line) for line in (DATA/'manifest.jsonl').read_text().splitlines()]
    assert Counter(r['split'] for r in manifest) == {'train':10000,'val':100,'test':50}
    groups = {s:{r['group'] for r in manifest if r['split']==s} for s in ('train','val','test')}
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    old = [json.loads(line) for line in (ROOT/'datasets/pilot-v1/manifest.jsonl').read_text().splitlines()]
    assert {r['id'] for r in old if r['split']=='train'} <= {r['id'] for r in manifest if r['split']=='train'}
    for split in ('val','test'):
        assert [r for r in old if r['split']==split] == [r for r in manifest if r['split']==split]
    for row in manifest:
        for key in ('drawing','code','step'): assert (DATA/row[key]).is_file()
        if 'full_tokens' in row: assert row['full_tokens'] <= 4096
    for split, count in [('train',10000),('val',100)]:
        records = [json.loads(line) for line in (DATA/f'sft/{split}.jsonl').read_text().splitlines()]
        assert len(records)==count
        for record in records:
            assert len(record['images'])==1 and Path(record['images'][0]).is_file()
            assert [m['role'] for m in record['messages']]==['system','user','assistant']
    assert digest(ROOT/'configs/train-pilot-v1.yaml')==digest(STAGE/'a1-config.yaml'), 'A1 config drift'
    original = yaml.safe_load((STAGE/'a1-config.yaml').read_text())
    expected = dict(original, dataset_dir=str(DATA/'sft'), output_dir=str(RUN/'output'))
    assert yaml.safe_load(CONFIG.read_text())==expected, 'Unexpected A2 configuration changes'
    assert original['lora_rank']==16 and original['finetuning_type']=='lora'
    assert original['freeze_vision_tower'] and original['freeze_multi_modal_projector']
    assert 'adapter_name_or_path' not in original, 'A2 must start from base model'
    assert not (RUN/'output').exists() or not any((RUN/'output').iterdir()), 'Existing output; do not overwrite'
    if shutil.disk_usage(RUN).free < 10*1024**3: return 'insufficient_data_disk_space'
    assert Path(CLI).is_file()
    return None


def supervise(fd):
    # The inherited open file description keeps the project lock held throughout training.
    lock = os.fdopen(fd, 'a')
    state = json.loads(STATE.read_text()); state.update(pid=os.getpid(), status='starting')
    save(STATE,state)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=state['gpu_uuid'], TOKENIZERS_PARALLELISM='false',
               HF_DATASETS_CACHE=str(RUN/'cache'), TMPDIR=str(RUN/'tmp'),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    try:
        with (RUN/'train.log').open('x') as log:
            process = subprocess.Popen([PYTHON,str(STAGE/'train_with_budget.py'),'train',str(CONFIG)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running', train_pid=process.pid); save(STATE,state)
            while process.poll() is None:
                try:
                    free=int(subprocess.check_output(['nvidia-smi','--id='+state['gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=10).strip())
                    pressure=free<8192
                except (ValueError,OSError,subprocess.SubprocessError):pressure=True
                if pressure:
                    import signal
                    state['stop_reason']='gpu_memory_reserve_below_8GiB_or_monitor_unavailable'
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(30)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    break
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
            code = process.returncode
        state.update(status='completed' if code==0 else 'failed', returncode=code, finished_unix=time.time())
    except Exception as exc:
        state.update(status='failed', error_type=type(exc).__name__, finished_unix=time.time())
    finally:
        save(STATE,state); lock.close()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--supervise-fd',type=int)
    args=parser.parse_args()
    if args.supervise_fd is not None: supervise(args.supervise_fd); return
    if STATE.exists():
        state=json.loads(STATE.read_text())
        print(json.dumps(dict(state, duplicate_launch_prevented=True)))
        return
    reason=validate()
    if reason: print(json.dumps({'status':'waiting','reason':reason})); return
    with (ROOT/'outputs/training.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'status':'waiting','reason':'project_lock_busy'})); return
        if STATE.exists():
            print(json.dumps({'status':'already_submitted'})); return
        first, initial=snapshot()
        if not first:
            print(json.dumps({'status':'waiting','reason':'no_gpu_below_10percent_and_5000MiB'})); return
        history=[(first,initial)]
        for _ in range(2):time.sleep(10);history.append(snapshot())
        chosen=stable_gpu(history);latest=history[-1][1]
        if chosen is None:
            print(json.dumps({'status':'waiting','reason':'gpu_occupancy_changed'})); return
        if args.check_only:
            print(json.dumps({'status':'ready','gpu_index':chosen[0],'training_started':False})); return
        record={'status':'submitted','gpu_index':chosen[0],'gpu_uuid':chosen[1],
                'config':str(CONFIG),'config_sha256':digest(CONFIG),'train_sha256':digest(DATA/'sft/train.jsonl'),
                'val_sha256':digest(DATA/'sft/val.jsonl'),'started_unix':time.time(),'gpu_snapshot':latest,
                'admission':'utilization<10%, memory.used<5000MiB for three samples 10s apart; memory range<=512MiB',
                'pytorch_memory_budget_gib':34,'minimum_free_memory_mib':8192}
        save(STATE,record)
        try:
            with (RUN/'supervisor.log').open('x') as log:
                process=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--supervise-fd',str(lock.fileno())],
                    cwd=ROOT,pass_fds=(lock.fileno(),),stdout=log,stderr=log,start_new_session=True)
        except Exception:
            record.update(status='launch_failed'); save(STATE,record); raise
        print(json.dumps({'status':'submitted','pid':process.pid,'gpu_index':chosen[0],'record':str(STATE)}))


if __name__=='__main__': main()
