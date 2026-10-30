"""Fixed-test evaluation. The model receives drawing/instructions only, never GT."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import time
import zipfile
import httpx
from dotenv import load_dotenv

def geometry_metrics(prediction,truth):
    import cadquery as cq
    pred=cq.importers.importStep(str(prediction)).val()
    gt=cq.importers.importStep(str(truth)).val()
    if not pred.isValid() or pred.Volume()<=0:
        return {'geometry_valid':False}
    pb,gb=pred.BoundingBox(),gt.BoundingBox()
    p=[pb.xlen,pb.ylen,pb.zlen];g=[gb.xlen,gb.ylen,gb.zlen]
    errors=[abs(a-b) for a,b in zip(p,g)]
    result={'geometry_valid':True,'bbox_error_mm_xyz':errors,'bbox_max_error_mm':max(errors),'bbox_within_0_1mm':max(errors)<=.1}
    try:
        intersection=pred.intersect(gt).Volume()
        union=pred.Volume()+gt.Volume()-intersection
        result['iou_same_coordinate_system']=max(0.,min(1.,intersection/union)) if union>0 else None
    except Exception: result['iou_same_coordinate_system']=None
    return result

def evaluate(manifest,output,url,token,limit=0):
    manifest=Path(manifest);output=Path(output)
    if output.exists(): raise ValueError('Output already exists; use a new experiment directory.')
    all_rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    groups={}
    for r in all_rows:
        group=r.get('group')
        if not group: raise ValueError('Every sample requires a group identifier.')
        groups.setdefault(group,set()).add(r.get('split'))
    if any(len(splits)>1 for splits in groups.values()): raise ValueError('Group leakage across splits detected.')
    rows=[r for r in all_rows if r.get('split')=='test']
    if limit: rows=rows[:limit]
    if not rows: raise ValueError('No test samples')
    output.mkdir(parents=True)
    results=[]
    with httpx.Client(timeout=600,trust_env=False,follow_redirects=False) as client:
        for index,row in enumerate(rows):
            started=time.monotonic();record={'id':row['id'],'success':False}
            folder=output/f'{index:04d}';folder.mkdir()
            try:
                image=(manifest.parent/row['drawing']).read_bytes()
                if row.get('drawing_sha256') and hashlib.sha256(image).hexdigest()!=row['drawing_sha256']: raise ValueError('Drawing checksum changed')
                # Never forward ground truth, code, dimensions, or source filename to inference.
                metadata={'unit':'mm','instructions':row.get('instructions','Reconstruct the dimensioned part.'),'messages':[],'previous_code':''}
                response=client.post(url.rstrip('/')+'/v1/reconstruct',headers={'Authorization':'Bearer '+token},files={'drawing':('drawing.png',image,'image/png')},data={'metadata':json.dumps(metadata)})
                record['http_status']=response.status_code
                if response.status_code==200:
                    if len(response.content)>50*1024*1024: raise ValueError('Oversized response')
                    from backend.app.worker import unpack_response
                    unpack_response(response.content,folder)
                    record.update(geometry_metrics(folder/'model.step',manifest.parent/row['step']))
                    record['success']=record['geometry_valid']
                    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                        if 'generation.json' in archive.namelist() and archive.getinfo('generation.json').file_size<=16384:
                            info=json.loads(archive.read('generation.json'))
                            record.update(attempt_count=info.get('attempt_count'),repair_count=info.get('repair_count'),model=info.get('model'),revision=info.get('revision'))
            except Exception as error:
                record['error_type']=type(error).__name__
            record['elapsed_seconds']=round(time.monotonic()-started,2);results.append(record)
            (output/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results),encoding='utf-8')
    successful=[r for r in results if r['success']]
    ious=[r['iou_same_coordinate_system'] for r in successful if r.get('iou_same_coordinate_system') is not None]
    summary={'total':len(results),'valid_count':len(successful),'valid_rate':len(successful)/len(results),'bbox_pass_count':sum(r.get('bbox_within_0_1mm',False) for r in results),'iou_comparable_count':len(ious),'mean_iou_of_comparable':statistics.mean(ious) if ious else None,'mean_elapsed_seconds':statistics.mean(r['elapsed_seconds'] for r in results),'coordinate_policy':'No rotation, scale normalization or alignment. Compare mm in the same XYZ coordinate frame.','manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    load_dotenv(Path(__file__).resolve().parents[1]/'server'/'server.env')
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True);parser.add_argument('--url',default='http://127.0.0.1:8001');parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    token=os.getenv('CAD_SERVER_TOKEN') or os.getenv('CAD_REMOTE_TOKEN')
    if not token: raise SystemExit('Set CAD_SERVER_TOKEN or CAD_REMOTE_TOKEN without putting it on the command line.')
    print(json.dumps(evaluate(args.manifest,args.output,args.url,token,args.limit),indent=2))
