"""Export explicitly automated research candidates, preserving review status."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from server.model import SYSTEM,task_prompt

def main():
    from transformers import AutoProcessor,AutoTokenizer
    from llamafactory.hparams import DataArguments
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    model=str(ROOT/'models/qwen3-vl-8b-instruct')
    tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=False)
    processor=AutoProcessor.from_pretrained(model,trust_remote_code=False)
    processor.image_max_pixels=262144
    args=DataArguments(template='qwen3_vl_nothink',cutoff_len=131072)
    template=get_template_and_fix_tokenizer(tokenizer,args)
    encoder=SupervisedDatasetProcessor(template,tokenizer,processor,args)
    source=ROOT/'datasets/pilot-v1'
    output=ROOT/'datasets/sft-pilot-v1';output.mkdir(exist_ok=True)
    rows=[json.loads(l) for l in (source/'manifest.jsonl').read_text().splitlines()]
    registry={};report={'license_status':'unresolved','human_reviewed':False,'cutoff_len':4096,'image_max_pixels':262144,'accepted':{},'excluded':[],'max_full_tokens':0}
    for split in ('train','val'):
        records=[]
        for r in rows:
            if r['split']!=split: continue
            code=(source/r['code']).read_text()
            user='<image>\n'+task_prompt({'instructions':r['instructions'],'messages':[]})
            answer='```python\n'+code.rstrip()+'\n```'
            image=str((source/r['drawing']).resolve())
            prompt=[{'role':'user','content':user}];response=[{'role':'assistant','content':answer}]
            ids,labels=encoder._encode_data_example(prompt,response,SYSTEM,None,[image],[],[])
            report['max_full_tokens']=max(report['max_full_tokens'],len(ids))
            if len(ids)>4096:
                report['excluded'].append({'id':r['id'],'reason':'full_sequence_over_4096','tokens':len(ids)});continue
            records.append({'images':[image],'messages':[{'role':'system','content':SYSTEM}]+prompt+response})
        (output/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
        report['accepted'][split]=len(records)
        registry['drawing_'+split]={'file_name':split+'.jsonl','formatting':'sharegpt','columns':{'messages':'messages','images':'images'},'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant','system_tag':'system'}}
    (output/'dataset_info.json').write_text(json.dumps(registry,indent=2))
    (output/'export-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
    if not all(report['accepted'].values()): raise RuntimeError('Empty training or validation split')

if __name__=='__main__': main()
