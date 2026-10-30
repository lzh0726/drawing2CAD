"""Deterministic research pilot. Never asserts human review or license clearance."""
import ast
import base64
from collections import Counter
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from server.research_sandbox import run

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

class Family(ast.NodeTransformer):
    def visit_Constant(self,node):
        return ast.copy_location(ast.Constant(value=0),node) if isinstance(node.value,(int,float)) else node
    def visit_Name(self,node):
        return ast.copy_location(ast.Name(id='v',ctx=node.ctx),node)
    def visit_UnaryOp(self,node):
        return self.visit(node.operand) if isinstance(node.op,(ast.USub,ast.UAdd)) else self.generic_visit(node)

def fingerprints(code):
    tree=ast.parse(code)
    exact=digest(ast.dump(tree,include_attributes=False))
    family=digest(ast.dump(Family().visit(tree),include_attributes=False))
    return exact,family

def checked(items):
    def evaluate(item):
        try: return item,run({'code':item[4]})
        except Exception: return item,None
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending=deque()
        for item in items:
            pending.append(pool.submit(evaluate,item))
            if len(pending)>=8: yield pending.popleft().result()
        while pending: yield pending.popleft().result()

def main():
    out=ROOT/'datasets/pilot-v1'
    out.mkdir(exist_ok=True)
    if (out/'manifest.jsonl').exists():
        raise RuntimeError('Existing pilot manifest: preserve it; inspect before replacing.')
    for name in ('images','codes','steps'): (out/name).mkdir(exist_ok=True)
    report={'seed':42,'license_status':'unresolved','human_reviewed':False,'exclusions':Counter(),'grouping':'AST structure ignoring numeric parameters and variable names; not a complete geometry equivalence proof'}
    probe=run({'probe':True})
    if not all(probe.values()): raise RuntimeError(str(probe))
    report['sandbox_probe']=probe
    manifest=[]; blocked_exact=set(); blocked_family=set(); geometry_splits={}
    def save(sid,split,image,code,step,geometry,group,source):
        (out/'images'/f'{sid}.png').write_bytes(image)
        (out/'steps'/f'{sid}.step').write_bytes(step)
        (out/'codes'/f'{sid}.py').write_text(code,encoding='utf-8')
        row={'id':sid,'group':group,'split':split,'drawing':f'images/{sid}.png','code':f'codes/{sid}.py','step':f'steps/{sid}.step','source':source,'license':'not declared in upstream dataset','license_status':'unresolved','human_reviewed':False,'validation_status':'automated_cad_checked','geometry':geometry,'instructions':'Generate CadQuery code for the single part shown in this drawing. Units are mm.'}
        manifest.append(row)
    with zipfile.ZipFile(ROOT/'datasets/raw/inference.zip') as z:
        rows=[json.loads(x) for x in z.read('inference/deepcad_test_data_subset100.jsonl').decode().splitlines()]
        for r in rows:
            exact,family=fingerprints(r['ground_truth']);blocked_exact.add(exact);blocked_family.add(family)
        random.Random(42).shuffle(rows)
        for i,r in enumerate(rows):
            stem=Path(r['image']).stem
            step=z.read(f'inference/test100_gt_steps_deepcad/{stem}.step')
            try:
                _,g=run({'step_b64':base64.b64encode(step).decode()})
                geometry_splits[g['geometric_signature']]='test'
            except Exception as e:
                g={'valid':False,'error':str(e)[:300]}
                report['exclusions']['official_test_invalid_gt']+=1
            if i<50:
                save('test-'+stem,'test',z.read('inference/test100_images_deepcad/'+r['image']),r['ground_truth']+'\nresult = solid\n',step,g,fingerprints(r['ground_truth'])[1],'Ortho2CAD official DeepCAD test/'+r['image'])
        report['official_test_parts_reserved']=len(rows)
    with zipfile.ZipFile(ROOT/'datasets/raw/ortho_train_data.zip') as z:
        rows=json.loads(z.read('ortho_train_data/cadquery_ortho_train_data.json'))
        random.Random(42).shuffle(rows)
        seen=set(); counts=Counter()
        def candidates():
            for i,r in enumerate(rows):
                code=r['conversations'][1]['value']
                exact,family=fingerprints(code)
                if exact in blocked_exact or family in blocked_family:
                    report['exclusions']['official_test_code_or_family']+=1;continue
                if exact in seen:
                    report['exclusions']['duplicate_code']+=1;continue
                seen.add(exact)
                split='val' if int(family[:8],16)%10==0 else 'train'
                if counts[split]>={'train':1000,'val':100}[split]: continue
                yield i,r,family,split,code+'\nresult = solid\n'
        for (i,r,family,split,code),evaluated in checked(candidates()):
            if counts[split]>={'train':1000,'val':100}[split]: continue
            if evaluated is None:
                report['exclusions']['cad_failure']+=1;continue
            step,g=evaluated
            if any(abs(a-b)/max(abs(b),1e-7)>.005 for a,b in zip(g['broad_dimensions'],g['dimensions'])):
                report['exclusions']['bbox_annotation_disagreement']+=1;continue
            signature=g['geometric_signature']
            if signature in geometry_splits:
                report['exclusions']['geometry_signature_duplicate']+=1;continue
            geometry_splits[signature]=split
            sid='train-source-'+str(r['id'])
            save(sid,split,z.read('ortho_train_data/'+r['image']),code,step,g,family,'Ortho2CAD training/'+r['image'])
            counts[split]+=1
            if sum(counts.values())%25==0: print(json.dumps({'examined':i+1,'accepted':dict(counts),'exclusions':report['exclusions']}),flush=True)
            if counts['train']==1000 and counts['val']==100: break
        report['counts']=dict(Counter(r['split'] for r in manifest))
    groups={s:{r['group'] for r in manifest if r['split']==s} for s in ('train','val','test')}
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (out/'preparation-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
