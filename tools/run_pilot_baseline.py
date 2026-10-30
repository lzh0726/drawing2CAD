"""Fixed official test pilot, original weights, greedy decoding, zero repairs."""
import base64
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    # Check all cards immediately before model load; never select an occupied GPU.
    listing=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    candidates=[]
    for line in listing.splitlines():
        index,memory,util=[int(x.strip()) for x in line.split(',')]
        if memory<256 and util==0: candidates.append(index)
    if not candidates: raise RuntimeError('No idle GPU; baseline not started')
    gpu=candidates[0]
    processes=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if processes: raise RuntimeError('GPU has compute processes; baseline not started')
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    from server.model import TransformersModel
    from server.policy import extract_code
    from server.research_sandbox import run
    folder=ROOT/'experiments/baseline-pilot-v1-tokens16384';folder.mkdir(exist_ok=True)
    model=TransformersModel(str(ROOT/'models/qwen3-vl-8b-instruct'),None,16384)
    model.load()
    model.processor.image_processor.size={'longest_edge':262144,'shortest_edge':4096}
    protocol={'gpu':gpu,'gpu_before_load':listing,'model':'qwen3-vl-8b-instruct','repairs':0,'seed':42,'samples':50,'max_new_tokens':16384,'image_max_pixels':262144,'alignment':'raw plus translation only; no scale fitting'}
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[]
    with zipfile.ZipFile(ROOT/'datasets/raw/inference.zip') as z:
        rows=[json.loads(l) for l in z.read('inference/deepcad_test_data_subset100.jsonl').decode().splitlines()]
        random.Random(42).shuffle(rows)
        for r in rows[:50]:
            sid=Path(r['image']).stem;result_path=folder/(sid+'.json')
            if result_path.exists():
                results.append(json.loads(result_path.read_text()));continue
            started=time.monotonic();item={'id':sid,'valid_solid':False}
            try:
                text=model.generate(z.read('inference/test100_images_deepcad/'+r['image']),{'instructions':'Generate CadQuery code for the single part shown in this drawing. Units are mm.','messages':[]})
                (folder/(sid+'.txt')).write_text(text)
                item['peak_gpu_memory_mb']=model.last_peak_memory_mb
                if text.lstrip().startswith('NEEDS_CLARIFICATION:'): raise ValueError('needs_clarification')
                step,geometry=run({'code':extract_code(text)})
                item['valid_solid']=True
                (folder/(sid+'.step')).write_bytes(step)
                truth=z.read('inference/test100_gt_steps_deepcad/'+sid+'.step')
                item['metrics']=run({'prediction_b64':base64.b64encode(step).decode(),'truth_b64':base64.b64encode(truth).decode()})
            except Exception as e: item['failure']=type(e).__name__+': '+str(e)[:500]
            item['seconds']=round(time.monotonic()-started,3)
            result_path.write_text(json.dumps(item,indent=2));results.append(item)
            print(json.dumps(item),flush=True)
    summary={'n':len(results),'valid_solids':sum(r['valid_solid'] for r in results),'metric_completed':sum('metrics' in r for r in results),'mean_iou_raw_all':sum(r.get('metrics',{}).get('iou_raw',0) for r in results)/len(results),'mean_iou_translation_aligned_all':sum(r.get('metrics',{}).get('iou_translation_aligned',0) for r in results)/len(results),'mean_seconds':sum(r['seconds'] for r in results)/len(results),'failed_metrics_counted_as_zero':True}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
