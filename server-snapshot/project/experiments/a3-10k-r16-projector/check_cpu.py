"""Validate the actual installed adapter construction without allocating CUDA memory."""
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, Qwen3VLForConditionalGeneration
from peft import get_peft_model_state_dict
from llamafactory.hparams import ModelArguments, FinetuningArguments
from llamafactory.model.adapter import init_adapter
import yaml
from train_with_budget import PROJECTORS, EXPECTED, audit_parameters

stage = Path(__file__).parent
config = yaml.safe_load((stage / 'train-a3-10k.yaml').read_text())
model_args = ModelArguments(model_name_or_path=config['model_name_or_path'])
fields = FinetuningArguments.__dataclass_fields__
tuning = FinetuningArguments(**{k: v for k, v in config.items() if k in fields})
base = AutoConfig.from_pretrained(model_args.model_name_or_path, local_files_only=True)
with init_empty_weights():
    model = Qwen3VLForConditionalGeneration(base)
    model = init_adapter(base, model, model_args, tuning, True)
counts = audit_parameters(model)
weights = get_peft_model_state_dict(model)
assert sum(p.numel() for p in weights.values()) == sum(EXPECTED.values())
assert all(any(m + '.' in k for k in weights) for m in PROJECTORS)
assert all(p.device.type == 'meta' for p in model.parameters())
result = {'status': 'passed', 'trainable_parameters': counts, 'total_trainable': sum(counts.values()),
          'projectors': PROJECTORS, 'saved_parameter_count': sum(p.numel() for p in weights.values()),
          'gpu_used': False, 'gradient_check': 'pending_actual_training'}
(stage / 'cpu-audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
