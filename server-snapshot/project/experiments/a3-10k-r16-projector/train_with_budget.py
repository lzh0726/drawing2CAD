"""A3-only entry point: language LoRA + full projectors, with resource audits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECTORS = ["model.visual.merger"] + [f"model.visual.deepstack_merger_list.{i}" for i in range(3)]
EXPECTED = {"lora": 43646976, "projectors": 160496896}
RUN = Path('/data/luzhuohong/drawing2CAD-experiments/a3-10k-r16-projector')
PROJECTOR_LR = 1e-5
BUDGET_GIB = 36
MIN_FREE_MIB = 42 * 1024


def classify(name):
    if any(p + '.modules_to_save.default.' in name for p in PROJECTORS):
        return 'projectors'
    if 'language_model.' in name and ('.lora_A.' in name or '.lora_B.' in name):
        return 'lora'
    raise AssertionError('Unexpected trainable parameter: ' + name)


def audit_parameters(model):
    counts = dict.fromkeys(EXPECTED, 0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            counts[classify(name)] += param.numel()
    assert counts == EXPECTED, counts
    return counts


def event(kind, **values):
    with (RUN / 'audit.jsonl').open('a') as out:
        out.write(json.dumps(dict(event=kind, unix=time.time(), **values)) + '\n')


def main():
    import yaml
    uuid = os.environ['CUDA_VISIBLE_DEVICES']
    row = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
        '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True, timeout=10)
    used, free, util = map(int, row.split(','))
    if free <= MIN_FREE_MIB or util >= 10:
        raise RuntimeError('GPU occupancy changed before loading; no model loaded')
    import torch
    torch.cuda.set_per_process_memory_fraction(BUDGET_GIB * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    from transformers import TrainerCallback
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    from llamafactory.train.tuner import run_exp

    def create_optimizer(self):
        if self.optimizer is None:
            counts = audit_parameters(self.model)
            groups = {'lora': [], 'projectors': []}
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    groups[classify(name)].append(p)
            cls, kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
            assert cls is torch.optim.AdamW, cls
            self.optimizer = cls([
                {'params': groups['lora'], 'lr': self.args.learning_rate, 'weight_decay': self.args.weight_decay},
                {'params': groups['projectors'], 'lr': PROJECTOR_LR, 'weight_decay': self.args.weight_decay},
            ], **kwargs)
            event('optimizer', counts=counts, lora_lr=self.args.learning_rate, projector_lr=PROJECTOR_LR)
        return self.optimizer

    CustomSeq2SeqTrainer.create_optimizer = create_optimizer

    class Audit(TrainerCallback):
        def memory(self, kind, state):
            event(kind, step=state.global_step,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3)

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            event('parameters', counts=audit_parameters(model), gpu_uuid=uuid, budget_gib=BUDGET_GIB)

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            if state.global_step != 0:
                return
            norms = {}
            for module in PROJECTORS:
                params = [p for n, p in model.named_parameters()
                          if module + '.modules_to_save.default.' in n and p.requires_grad]
                assert params and all(p.grad is not None for p in params), 'Missing gradients: ' + module
                norm = torch.stack([p.grad.detach().norm() for p in params]).norm().item()
                assert 0 < norm < float('inf'), 'Invalid/zero gradients: ' + module
                norms[module] = norm
            event('projector_gradients_verified', norms=norms)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in (1, 5, 25) or state.global_step % 100 == 0:
                self.memory('step_memory', state)

        def on_evaluate(self, args, state, control, **kwargs):
            self.memory('validation_memory', state)

        def on_save(self, args, state, control, **kwargs):
            from safetensors import safe_open
            path = Path(args.output_dir) / f'checkpoint-{state.global_step}' / 'adapter_model.safetensors'
            with safe_open(path, framework='pt', device='cpu') as saved:
                keys = list(saved.keys())
                assert all(any(m + '.' in k for k in keys) for m in PROJECTORS), 'Projectors missing in checkpoint'
                import math
                count = sum(math.prod(saved.get_slice(k).get_shape()) for k in keys)
                assert count == sum(EXPECTED.values()), count
            event('checkpoint_verified', step=state.global_step, saved_parameters=count)

        def on_train_end(self, args, state, control, **kwargs):
            self.memory('training_end', state)

    config = yaml.safe_load(Path(sys.argv[-1]).read_text())
    assert config['freeze_vision_tower'] and not config['freeze_multi_modal_projector']
    assert config['additional_target'].split(',') == PROJECTORS
    assert 'adapter_name_or_path' not in config
    run_exp(config, callbacks=[Audit()])


if __name__ == '__main__':
    main()
