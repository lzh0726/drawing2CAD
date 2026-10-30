"""Export only human-reviewed training labels. Never automatically train on model output."""
import argparse
import json
from pathlib import Path
from .model import SYSTEM

def export(manifest,target):
    manifest=Path(manifest);target=Path(target)
    if target.exists(): raise ValueError('Target exists')
    rows=[json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    groups={}
    for row in rows:
        if not row.get('group'): raise ValueError('group is required')
        groups.setdefault(row['group'],set()).add(row['split'])
    if any(len(s)>1 for s in groups.values()): raise ValueError('Group leakage between dataset splits')
    selected=[r for r in rows if r['split']=='train' and r.get('human_reviewed') is True]
    if not selected: raise ValueError('No human-reviewed training labels; no output written')
    result=[]
    for row in selected:
        if not row.get('source') or not row.get('license'): raise ValueError('Source and license are required')
        code=(manifest.parent/row['code']).read_text(encoding='utf-8')
        from .policy import validate_code
        validate_code(code)
        result.append({'id':row['id'],'group':row['group'],'images':[str((manifest.parent/row['drawing']).resolve())],'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':'<image>\n'+row.get('instructions','Reconstruct this drawing as CadQuery code.')},{'role':'assistant','content':code}]})
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf-8')
    return len(result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(f'Exported {export(args.manifest,args.output)} reviewed training samples.')
