"""Independent A3 inference worker. Merge on CPU, verify projectors, then generate."""
import base64
import ctypes
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/home/luzhuohong/drawing2CAD')
TRAIN = Path('/data/luzhuohong/drawing2CAD-experiments/a3-10k-r16-projector/output')
sys.path.insert(0, str(ROOT))
from server.model import TransformersModel


class A3Model(TransformersModel):
    def load(self, cpu_only=False):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from peft import PeftModel
        from safetensors import safe_open
        torch.set_num_threads(4)
        if cpu_only:
            assert not torch.cuda.is_available()
        else:
            assert torch.cuda.is_available()
            if os.environ.get('CAD_GPU_MEMORY_LIMIT_MIB'):
                limit = int(os.environ['CAD_GPU_MEMORY_LIMIT_MIB']) * 1024**2
                torch.cuda.set_per_process_memory_fraction(min(limit / torch.cuda.get_device_properties(0).total_memory, 1.0))
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.name, device_map='cpu', torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            low_cpu_mem_usage=True, trust_remote_code=False, local_files_only=True)
        model = PeftModel.from_pretrained(model, str(TRAIN), is_trainable=False)
        model = model.merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16).eval()
        parameters = dict(model.named_parameters())
        checked = 0
        with safe_open(TRAIN / 'adapter_model.safetensors', framework='pt', device='cpu') as adapter:
            for key in adapter.keys():
                if '.visual.merger.' not in key and '.visual.deepstack_merger_list.' not in key:
                    continue
                name = key.removeprefix('base_model.model.')
                assert torch.equal(parameters[name].detach(), adapter.get_tensor(key).to(torch.bfloat16)), name
                checked += parameters[name].numel()
        assert checked == 160496896, checked
        self.processor = AutoProcessor.from_pretrained(self.name, trust_remote_code=False, local_files_only=True)
        self.processor.image_processor.size = {'longest_edge': 262144, 'shortest_edge': 4096}
        self.model = model if cpu_only else model.to('cuda:0')
        with (Path(__file__).parent / 'model-load-audit.jsonl').open('a') as log:
            log.write(json.dumps({'unix': time.time(), 'pid': os.getpid(), 'cpu_only': cpu_only,
                                  'projector_parameters_verified': checked, 'merged_and_cast_bf16': True}) + '\n')


def main():
    if '--cpu-check' in sys.argv:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        model = A3Model(str(ROOT / 'models/qwen3-vl-8b-instruct'), None)
        model.load(cpu_only=True)
        print(json.dumps({'cpu_model_load': 'passed', 'projector_parameters_verified': 160496896, 'gpu_started': False}))
        return
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise RuntimeError('Parent-death signal unavailable')
    if os.getppid() != int(os.environ['CAD_CONTROLLER_PID']):
        raise RuntimeError('Controller exited')
    request = json.load(sys.stdin)
    model = A3Model(request['model'], request['revision'])
    model.load()
    text = model.generate(base64.b64decode(request['image']), request['metadata'], request['previous_code'], request['error'])
    print(json.dumps({'text': text, 'peak_memory_mb': model.last_peak_memory_mb}))


if __name__ == '__main__':
    main()
