"""Prepare independent smoke/train/export configurations without loading weights."""
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
base=yaml.safe_load((ROOT/'configs/lora-pilot-v1.yaml').read_text())
smoke=dict(base,output_dir='./outputs/smoke-pilot-v1',max_samples=32,max_steps=20,gradient_accumulation_steps=1,logging_steps=1,save_steps=20,eval_steps=20)
train=dict(base,save_steps=25,eval_steps=25,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False)
export={'model_name_or_path':base['model_name_or_path'],'adapter_name_or_path':base['output_dir'],'template':base['template'],'finetuning_type':'lora','trust_remote_code':False,'export_dir':'./models/drawing2cad-v1','export_size':4,'export_device':'cpu','export_legacy_format':False}
for name,data in [('smoke-pilot-v1.yaml',smoke),('train-pilot-v1.yaml',train),('export-pilot-v1.yaml',export)]:
    path=ROOT/'configs'/name
    if path.exists(): raise RuntimeError(f'Preserve existing config: {path}')
    path.write_text(yaml.safe_dump(data,sort_keys=False))
    print(path)
