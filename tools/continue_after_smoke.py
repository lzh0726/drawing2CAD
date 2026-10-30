"""Run the authorized training/export/evaluation stages, stopping on failure."""
import json
import math
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/home/luzhuohong/miniconda3/envs/drawing2cad-train/bin/python'

def main():
    state={'stage':'waiting_for_smoke','training_complete':False,'export_complete':False,'evaluation_complete':False,'page_integration_complete':False}
    status=ROOT/'experiments/phase9-12-status.json'
    def save():status.write_text(json.dumps(state,indent=2))
    save()
    try:
        deadline=time.monotonic()+3600
        while True:
            launch=json.loads((ROOT/'experiments/smoke-launch.json').read_text())
            if 'returncode' in launch:break
            if time.monotonic()>deadline:raise TimeoutError('Smoke launcher has not completed within one hour')
            time.sleep(10)
        if launch['returncode']!=0:raise RuntimeError('Smoke training failed; inspect logs/smoke-pilot-v1.log')
        folder=ROOT/'outputs/smoke-pilot-v1'
        trainer=json.loads((folder/'trainer_state.json').read_text())
        assert trainer['global_step']==20
        for record in trainer['log_history']:
            for key in ('loss','eval_loss','grad_norm'):
                if key in record and not math.isfinite(record[key]):raise RuntimeError('Nonfinite smoke '+key)
        from safetensors import safe_open
        with safe_open(folder/'adapter_model.safetensors',framework='pt',device='cpu') as f:
            keys=list(f.keys())
        if not keys or any('lora_' not in k or '.visual.' in k for k in keys):
            raise RuntimeError('Unexpected adapter parameters; inspect before formal training')
        state.update(smoke_complete=True,smoke_steps=20,adapter_tensor_count=len(keys));save()
        stages=[('training',['tools/launch_training.py','train']),('export',[ 'tools/export_checked.py']),('evaluation_zero_repair',['tools/evaluate_finetuned.py','--repairs','0']),('evaluation_two_repairs',['tools/evaluate_finetuned.py','--repairs','2'])]
        for name,args in stages:
            state['stage']=name;save()
            with (ROOT/'logs'/(name+'-pipeline.log')).open('x') as log:
                subprocess.run([PYTHON,'-u',*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            if name=='training':state['training_complete']=True
            if name=='export':state['export_complete']=True
            if name=='evaluation_two_repairs':state['evaluation_complete']=True
            save()
        state['stage']='offline_complete_page_integration_pending';save()
    except Exception as e:
        state.update(status='stopped',error=type(e).__name__+': '+str(e)[:600]);save();raise

if __name__=='__main__':main()
