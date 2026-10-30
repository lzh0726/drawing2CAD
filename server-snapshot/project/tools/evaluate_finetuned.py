"""Compare the merged model on the unchanged pilot test set."""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.launch_training import idle_gpus

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repairs',type=int,choices=[0,2],required=True)
    args=parser.parse_args()
    candidates,_=idle_gpus();time.sleep(2);current,snapshot=idle_gpus()
    gpu=next((x for x in candidates if x in current),None)
    if gpu is None: raise RuntimeError('No idle GPU; evaluation not started')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu[1]
    from server.model import TransformersModel
    from server.policy import extract_code,validate_code
    from server.research_sandbox import run
    model=TransformersModel(str(ROOT/'models/drawing2cad-v1'),None,16384)
    model.load();model.processor.image_processor.size={'longest_edge':262144,'shortest_edge':4096}
    folder=ROOT/'experiments'/f'finetuned-pilot-v1-repairs{args.repairs}-tokens16384';folder.mkdir(exist_ok=True)
    protocol={'gpu':gpu,'snapshot':snapshot,'repairs':args.repairs,'max_new_tokens':16384,'image_max_pixels':262144,'model':model.name,'test_manifest':'datasets/pilot-v1/manifest.jsonl','checkpoint_selection':'validation loss only'}
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2))
    data=ROOT/'datasets/pilot-v1'
    rows=[json.loads(l) for l in (data/'manifest.jsonl').read_text().splitlines()]
    results=[]
    for row in rows:
        if row['split']!='test':continue
        path=folder/(row['id']+'.json')
        if path.exists(): results.append(json.loads(path.read_text()));continue
        started=time.monotonic();record={'id':row['id'],'valid_ground_truth':row['geometry']['valid'],'valid_solid':False,'attempts':[]}
        image=(data/row['drawing']).read_bytes();code='';error=''
        try:
            for i in range(args.repairs+1):
                text=model.generate(image,{'instructions':row['instructions'],'messages':[]},code,error)
                (folder/f'{row["id"]}-attempt{i+1}.txt').write_text(text)
                attempt={'number':i+1};record['attempts'].append(attempt)
                if text.strip().startswith('NEEDS_CLARIFICATION:'):
                    attempt['status']='needs_clarification';break
                try:
                    code=extract_code(text);validate_code(code)
                    step,geometry=run({'code':code})
                except (ValueError,SyntaxError,RuntimeError) as e:
                    error='Invalid Python syntax. Return a complete CadQuery program.' if isinstance(e,SyntaxError) else str(e)[:500]
                    attempt.update(status='failed',error=error);continue
                record['valid_solid']=True;attempt['status']='passed'
                (folder/(row['id']+'.step')).write_bytes(step)
                if record['valid_ground_truth']:
                    record['metrics']=run({'prediction_b64':base64.b64encode(step).decode(),'truth_b64':base64.b64encode((data/row['step']).read_bytes()).decode()})
                break
            record['peak_gpu_memory_mb']=model.last_peak_memory_mb
        except Exception as e:record['failure']=type(e).__name__+': '+str(e)[:500]
        record['seconds']=round(time.monotonic()-started,3)
        path.write_text(json.dumps(record,indent=2));results.append(record);print(json.dumps(record),flush=True)
    valid=[r for r in results if r['valid_ground_truth']]
    evaluable=[r for r in valid if not r['valid_solid'] or 'metrics' in r]
    summary={'fixed_test_count':len(results),'valid_ground_truth_count':len(valid),'evaluable_count':len(evaluable),'valid_generated_solids_on_valid_gt':sum(r['valid_solid'] for r in valid),'metric_error_ids':[r['id'] for r in valid if r['valid_solid'] and 'metrics' not in r],'mean_iou_raw_evaluable':sum(r.get('metrics',{}).get('iou_raw',0) for r in evaluable)/len(evaluable),'mean_iou_translation_aligned_evaluable':sum(r.get('metrics',{}).get('iou_translation_aligned',0) for r in evaluable)/len(evaluable),'mean_seconds':sum(r['seconds'] for r in results)/len(results)}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
