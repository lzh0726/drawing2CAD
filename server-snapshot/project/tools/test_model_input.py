"""Exercise real multimodal preprocessing with a CPU generation stub."""
import io
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def test_system_message_works_with_installed_processor(monkeypatch):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from server.model import TransformersModel
    model=TransformersModel(str(ROOT/'models/qwen3-vl-8b-instruct'),None)
    model.processor=AutoProcessor.from_pretrained(model.name,trust_remote_code=False)
    model.processor.image_processor.size={'longest_edge':262144,'shortest_edge':4096}
    expected='import cadquery as cq\nresult=cq.Workplane("XY").box(1,2,3)'
    class Stub:
        device='cpu'
        def generate(self,**kwargs):
            suffix=torch.tensor([model.processor.tokenizer.encode(expected,add_special_tokens=False)])
            return torch.cat([kwargs['input_ids'],suffix],dim=1)
    model.model=Stub()
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    image=io.BytesIO();Image.new('RGB',(100,100),'white').save(image,format='PNG')
    assert model.generate(image.getvalue(),{'instructions':'Generate CAD.','messages':[]})==expected
