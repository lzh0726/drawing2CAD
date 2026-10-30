"""Diagnostic rigid axis rotations inside bwrap. No scaling or reflections."""
import base64
import itertools
import json
import math
from pathlib import Path
import random
import resource
import sys

resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(35,35))
resource.setrlimit(resource.RLIMIT_FSIZE,(40*1024**2,40*1024**2))
import cadquery as cq

request=json.loads(sys.stdin.buffer.read(6*1024**2))
shapes=[]
for key in ['prediction','truth']:
    path=Path('/work')/(key+'.step');path.write_bytes(base64.b64decode(request[key]))
    shapes.append(cq.importers.importStep(str(path)).solids().val())
pred,truth=shapes
def bounds(shape):
    b=shape.BoundingBox()
    return [b.xlen,b.ylen,b.zlen],[(b.xmin+b.xmax)/2,(b.ymin+b.ymax)/2,(b.zmin+b.zmax)/2]
pd,pc=bounds(pred);td,tc=bounds(truth)
rotations=[]
for perm in itertools.permutations(range(3)):
    inversions=sum(perm[i]>perm[j] for i in range(3) for j in range(i+1,3))
    for signs in itertools.product([-1,1],repeat=3):
        if (-1)**inversions*signs[0]*signs[1]*signs[2]!=1:continue
        matrix=[[signs[i] if j==perm[i] else 0 for j in range(3)] for i in range(3)]
        error=max(abs(pd[perm[i]]-td[i])/td[i] for i in range(3))
        rotations.append((error,matrix,perm))
best_error=min(x[0] for x in rotations)
candidates=[x for x in rotations if x[0]<=best_error+1e-7]
if request.get('rotation_matrix'):
    candidates=[x for x in candidates if x[1]==request['rotation_matrix']]
results=[]
for _,matrix,perm in candidates:
    lengths=[max(pd[perm[i]],td[i]) for i in range(3)]
    rng=random.Random(42);intersection=union=0
    for _ in range(1024):
        point=[rng.uniform(-x/2,x/2) for x in lengths]
        pp=[sum(matrix[i][j]*point[i] for i in range(3))+pc[j] for j in range(3)]
        tp=[point[i]+tc[i] for i in range(3)]
        a,b=pred.isInside(cq.Vector(*pp),1e-9),truth.isInside(cq.Vector(*tp),1e-9)
        intersection+=a and b;union+=a or b
    results.append({'rotation_matrix':matrix,'estimated_iou':intersection/union if union else 0,
                    'union_samples':union,'intersection_samples':intersection,'samples':1024})
best=max(results,key=lambda x:x['estimated_iou'])
report={'method':'24 proper axis rotations, closest bbox candidates, bbox-center translation; no scaling/reflections',
        'candidate_count':len(candidates),'bbox_relative_error_after_axis_permutation':best_error,
        'best':best,'all_candidates':results}
print(json.dumps(report),flush=True)
try:
    import numpy as np
    rotation=np.array(best['rotation_matrix'],dtype=float)
    angle=math.acos(max(-1,min(1,(np.trace(rotation)-1)/2)))
    rotated=pred.translate(tuple(-x for x in pc))
    if angle>1e-9:
        axis=np.array([rotation[2,1]-rotation[1,2],rotation[0,2]-rotation[2,0],rotation[1,0]-rotation[0,1]])
        if np.linalg.norm(axis)<1e-8:
            values,vectors=np.linalg.eig(rotation)
            axis=np.real(vectors[:,np.argmin(abs(values-1))])
        axis=axis/np.linalg.norm(axis)
        rotated=rotated.rotate((0,0,0),tuple(float(x) for x in axis),math.degrees(angle))
    rotated=rotated.translate(tuple(tc))
    intersection=rotated.intersect(truth).Volume()
    union=rotated.Volume()+truth.Volume()-intersection
    report['exact_iou_after_rigid_axis_alignment']=max(0,min(1,intersection/union)) if union>0 else None
except Exception as exc:report['exact_error_type']=type(exc).__name__
print(json.dumps(report),flush=True)
