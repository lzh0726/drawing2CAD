"""Offline comparison only; invoked inside the same bwrap sandbox."""
import base64
import json
from pathlib import Path
import random
import resource
import sys

resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(35,35))
resource.setrlimit(resource.RLIMIT_FSIZE,(40*1024**2,40*1024**2))
import cadquery as cq

request=json.loads(sys.stdin.buffer.read(4*1024**2))
solids=[]
for name in ('prediction_b64','truth_b64'):
    path=Path('/work')/(name+'.step')
    path.write_bytes(base64.b64decode(request[name]))
    shapes=cq.importers.importStep(str(path)).solids().vals()
    assert len(shapes)==1 and shapes[0].isValid()
    solids.append(shapes[0])
a,b=solids
aa,bb=a.BoundingBox(),b.BoundingBox()
low=[min(getattr(aa,k),getattr(bb,k)) for k in ('xmin','ymin','zmin')]
high=[max(getattr(aa,k),getattr(bb,k)) for k in ('xmax','ymax','zmax')]
samples=int(request.get('samples',8192))
assert 1<=samples<=8192
rng=random.Random(42);intersection=union=mismatch=0
for _ in range(samples):
    point=cq.Vector(*[rng.uniform(lo,hi) for lo,hi in zip(low,high)])
    ina,inb=a.isInside(point,1e-9),b.isInside(point,1e-9)
    intersection+=ina and inb;union+=ina or inb;mismatch+=ina!=inb
print(json.dumps({'method':'seeded uniform volume sampling; approximate, not Boolean IoU','seed':42,'samples':samples,'intersection_samples':intersection,'union_samples':union,'mismatch_samples':mismatch,'estimated_iou':intersection/union if union else None,'volume_relative_error':abs(a.Volume()-b.Volume())/b.Volume(),'bbox_absolute_mm':[abs(getattr(aa,k)-getattr(bb,k)) for k in ('xlen','ylen','zlen')]}))
