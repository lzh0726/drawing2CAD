"""Explicit single-stage launcher; fails closed when every GPU is occupied."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/luzhuohong/miniconda3/envs/drawing2cad-train/bin/python'
CLI='/home/luzhuohong/miniconda3/envs/drawing2cad-train/bin/llamafactory-cli'

def idle_gpus():
    info=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    idle=[]
    for line in info.splitlines():
        index,uuid,memory,util=[x.strip() for x in line.split(',')]
        if uuid not in busy and int(memory)<256 and int(util)==0:
            idle.append((index,uuid))
    return idle,info

def main():
    import yaml
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['smoke','train'])
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    config=ROOT/'configs'/('smoke-pilot-v1.yaml' if args.stage=='smoke' else 'train-pilot-v1.yaml')
    settings=yaml.safe_load(config.read_text())
    output=ROOT/settings['output_dir']
    if output.exists() and any(output.iterdir()): raise RuntimeError('Output exists; inspect before resuming or replacing')
    if args.stage=='train':
        smoke=ROOT/'outputs/smoke-pilot-v1'
        if not (smoke/'adapter_model.safetensors').exists(): raise RuntimeError('20-step smoke adapter missing')
        metrics=json.loads((smoke/'train_results.json').read_text())
        import math
        if not math.isfinite(metrics['train_loss']): raise RuntimeError('Smoke loss is not finite')
        state=json.loads((smoke/'trainer_state.json').read_text())
        if state['global_step']!=20: raise RuntimeError('Smoke run did not complete 20 steps')
    if shutil.disk_usage(ROOT).free<4*1024**3: raise RuntimeError('Less than 4 GiB free for training outputs')
    # Lock protects this project's launchers; other users require scheduler coordination.
    with (ROOT/'outputs/training.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle,first=idle_gpus()
        if not idle:
            print(json.dumps({'status':'blocked','reason':'no_idle_gpu','gpu_snapshot':first}),flush=True)
            return 3
        time.sleep(2)
        second,latest=idle_gpus()
        selected=next((x for x in idle if x in second),None)
        if selected is None: raise RuntimeError('GPU occupancy changed; no job launched')
        if args.check_only:
            print(json.dumps({'status':'ready','gpu':selected,'config':str(config),'training_started':False}));return 0
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=selected[1]
        env['TOKENIZERS_PARALLELISM']='false'
        record={'stage':args.stage,'gpu_index':selected[0],'gpu_uuid':selected[1],'config':str(config),'started_unix':time.time(),'gpu_snapshot':latest}
        (ROOT/'experiments'/f'{args.stage}-launch.json').write_text(json.dumps(record,indent=2))
        with (ROOT/'logs'/f'{args.stage}-pilot-v1.log').open('x') as log:
            result=subprocess.run([CLI,'train',str(config)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode,finished_unix=time.time())
        (ROOT/'experiments'/f'{args.stage}-launch.json').write_text(json.dumps(record,indent=2))
        return result.returncode

if __name__=='__main__': raise SystemExit(main())
