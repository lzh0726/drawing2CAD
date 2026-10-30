"""Merge the validated training output on CPU, retaining base and adapter."""
import json
from pathlib import Path
import shutil
import subprocess
import yaml
ROOT=Path(__file__).resolve().parents[1]
CLI='/home/luzhuohong/miniconda3/envs/drawing2cad-train/bin/llamafactory-cli'

def main():
    config=ROOT/'configs/export-pilot-v1.yaml'
    settings=yaml.safe_load(config.read_text())
    adapter=ROOT/settings['adapter_name_or_path']
    if not (adapter/'adapter_model.safetensors').exists(): raise RuntimeError('Training adapter missing')
    state=json.loads((adapter/'trainer_state.json').read_text())
    if not state.get('best_model_checkpoint'): raise RuntimeError('No validation-selected checkpoint recorded')
    target=ROOT/settings['export_dir']
    if target.exists() and any(target.iterdir()): raise RuntimeError('Export directory already contains files')
    size=sum(p.stat().st_size for p in (ROOT/settings['model_name_or_path']).glob('*.safetensors'))
    free=shutil.disk_usage(ROOT).free
    if free<size+3*1024**3: raise RuntimeError('Insufficient disk: need model size plus 3 GiB headroom')
    available=int(next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')).split()[1])*1024
    if available<2*size+4*1024**3: raise RuntimeError('Insufficient available CPU RAM for merge')
    subprocess.run([CLI,'export',str(config)],cwd=ROOT,check=True)
    from transformers import AutoConfig,AutoProcessor
    AutoConfig.from_pretrained(target,trust_remote_code=False)
    AutoProcessor.from_pretrained(target,trust_remote_code=False)
    (ROOT/'experiments/export-validation.json').write_text(json.dumps({'config_and_processor_loadable':True,'weights_generation_test_pending':True,'best_checkpoint':state['best_model_checkpoint']},indent=2))

if __name__=='__main__': main()
