"""Read ZIP data and AST only. Never execute dataset code or extract arbitrary paths."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import sys
from zipfile import ZipFile

root = Path(sys.argv[1]).resolve()
out = root / 'experiments' / 'dataset-audit-20261024'
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))
from server.policy import validate_code, extract_code

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def static(code):
    try:
        code = extract_code(code)
        validate_code(code)
        return 'pass'
    except Exception as e:
        return type(e).__name__ + ': ' + str(e)

report = {'execution': 'No dataset Python executed. AST pass is not geometry validation.', 'archives': {}, 'train': {}, 'test': {}}
train_ids = set()
sample_rows = []
for archive in ['ortho_train_data.zip', 'inference.zip']:
    path = root/'datasets/raw'/archive
    with ZipFile(path) as z:
        names = [i.filename for i in z.infolist() if not i.is_dir()]
        report['archives'][archive] = {
            'bytes': path.stat().st_size, 'sha256': digest(path),
            'files': len(names), 'extensions': dict(Counter(PurePosixPath(n).suffix for n in names)),
            'duplicate_zip_members': len(names)-len(set(names)),
            'unsafe_paths': [n for n in names if PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts],
            'license_files': [n for n in names if any(k in PurePosixPath(n).name.lower() for k in ['license','copyright','readme'])],
            'crc_first_bad_member': z.testzip(),
        }
        if archive == 'ortho_train_data.zip':
            rows = json.loads(z.read('ortho_train_data/cadquery_ortho_train_data.json'))
            name_set = set(names)
            missing, malformed, raw_errors, adapted_errors = [], [], Counter(), Counter()
            images, ids, hashes, calls, imports = Counter(), Counter(), Counter(), Counter(), Counter()
            no_solid = 0
            picks = set(random.Random(42).sample(range(len(rows)), 12)) | {0}
            png_sizes = Counter()
            for idx, row in enumerate(rows):
                image = row.get('image','')
                images[image] += 1
                ids[str(row.get('id'))] += 1
                train_ids.add(PurePosixPath(image).stem)
                member = 'ortho_train_data/' + image
                if member not in name_set: missing.append({'row':idx,'image':image})
                answers = [m.get('value','') for m in row.get('conversations',[]) if m.get('from') == 'gpt']
                if len(answers) != 1:
                    malformed.append(idx)
                    continue
                code = answers[0]
                hashes[hashlib.sha256(code.encode()).hexdigest()] += 1
                raw_errors[static(code)] += 1
                try:
                    tree = ast.parse(extract_code(code))
                    bound = {n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
                    for n in ast.walk(tree):
                        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute): calls[n.func.attr] += 1
                        if isinstance(n,ast.Import): imports.update(a.name for a in n.names)
                    adapted = code + '\nresult = solid\n' if 'solid' in bound else code
                    if 'solid' not in bound: no_solid += 1
                    adapted_errors[static(adapted)] += 1
                except SyntaxError:
                    adapted_errors['syntax_error'] += 1
                if idx in picks and member in name_set:
                    folder = out/'samples'/f'train-{idx:06d}'
                    folder.mkdir(parents=True,exist_ok=True)
                    data = z.read(member)
                    (folder/'drawing.png').write_bytes(data)
                    (folder/'original.py.txt').write_text(code,encoding='utf-8')
                    if data[:8] == b'\x89PNG\r\n\x1a\n':
                        png_sizes[str((int.from_bytes(data[16:20],'big'),int.from_bytes(data[20:24],'big')))] += 1
                    sample_rows.append({'row':idx,'id':row.get('id'),'image':image,'sample_folder':str(folder.relative_to(out))})
            report['train'] = {
                'rows':len(rows),'missing_images':missing[:50],'missing_image_count':len(missing),
                'malformed_rows':malformed[:50],'malformed_count':len(malformed),
                'unique_images':len(images),'duplicate_image_rows':sum(v-1 for v in images.values()),
                'duplicate_ids':sum(v-1 for v in ids.values()),
                'exact_code_duplicate_rows':sum(v-1 for v in hashes.values()),
                'unreferenced_png_count':sum(n.endswith('.png') for n in names)-len(images),
                'raw_policy':dict(raw_errors),'with_result_equals_solid_policy':dict(adapted_errors),
                'missing_solid_assignment':no_solid,'method_calls':dict(calls.most_common()),
                'imports':dict(imports),'sample_png_sizes':dict(png_sizes),'samples':sample_rows,
            }
            (out/'train_part_ids.json').write_text(json.dumps(sorted(train_ids)),encoding='utf-8')
        else:
            parent_counts = Counter(str(PurePosixPath(n).parent) for n in names)
            report['test']['directories'] = dict(parent_counts)
            datasets = [('deepcad','deepcad_test_data_subset100.jsonl','test100_images_deepcad','test100_gt_steps_deepcad'),('f360rec','f360rec_test_data_subset100.jsonl','test100_images_f360','test100_gt_steps_f360'),('f360rec_mdim','f360rec_test_data_subset100_mdim.jsonl','test100_images_f360_mdim','test100_gt_steps_f360'),('zerotocad1m','zerotocad1m_test_data_subset100.jsonl','test100_images_zerotocad1m','test100_gt_steps_zerotocad1m')]
            for label,jname,idir,sdir in datasets:
                rows=[json.loads(s) for s in z.read('inference/'+jname).decode().splitlines() if s.strip()]
                missing_i,missing_s, units, code_policy = [],[],Counter(),Counter()
                for idx,row in enumerate(rows):
                    im='inference/'+idir+'/'+row['image']
                    step='inference/'+sdir+'/'+row.get('step',PurePosixPath(row['image']).stem+'.step')
                    if im not in names: missing_i.append(im)
                    if step not in names: missing_s.append(step)
                    else:
                        s=z.read(step).decode(errors='replace')
                        units.update(re.findall(r'(?m)^.*(?:SI_UNIT|CONVERSION_BASED_UNIT).*$',s))
                    if row.get('ground_truth'): code_policy[static(row['ground_truth']+'\nresult = solid\n')] += 1
                    if idx < 3:
                        folder=out/'samples'/f'{label}-{idx:03d}'
                        folder.mkdir(parents=True,exist_ok=True)
                        if im in names: (folder/'drawing.png').write_bytes(z.read(im))
                        if step in names: (folder/'truth.step').write_bytes(z.read(step))
                        if row.get('ground_truth'): (folder/'original.py.txt').write_text(row['ground_truth'],encoding='utf-8')
                overlaps=sorted({PurePosixPath(r['image']).stem for r in rows}&train_ids) if label=='deepcad' else []
                report['test'][label]={'rows':len(rows),'nonempty_code_labels':sum(bool(r.get('ground_truth')) for r in rows),'missing_images':missing_i,'missing_steps':missing_s,'unit_definitions':dict(units),'adapted_code_policy':dict(code_policy),'train_part_id_overlap':overlaps}
            svg=next((n for n in names if n.endswith('.svg')),None)
            if svg: (out/'sample_manually_dimensioned.svg').write_bytes(z.read(svg))
(out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='test'},ensure_ascii=False,indent=2))
print('TEST', json.dumps({k:{a:b for a,b in v.items() if a!='unit_definitions'} for k,v in report['test'].items() if isinstance(v,dict)},ensure_ascii=False))
print('REPORT',str(out/'audit.json'))
