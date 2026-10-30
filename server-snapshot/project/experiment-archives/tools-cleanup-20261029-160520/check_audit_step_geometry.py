"""Import trusted-format STEP samples; never execute dataset Python labels."""
from pathlib import Path
import json
import sys
import cadquery as cq
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib

root = Path(sys.argv[1])
results = []
for path in sorted((root / 'inspection').glob('*/truth.step')):
    workplane = cq.importers.importStep(str(path))
    if len(workplane.objects) != 1:
        raise ValueError(f'Unexpected multiple imported objects: {path}')
    shape = workplane.val()
    box = shape.BoundingBox()
    broad = Bnd_Box()
    BRepBndLib.Add_s(shape.wrapped, broad)
    a = broad.Get()
    results.append({
        'sample': path.parent.name,
        'valid': shape.isValid(), 'solids': len(shape.Solids()),
        'bbox_mm': [box.xlen, box.ylen, box.zlen],
        'bounds_mm': [box.xmin, box.ymin, box.zmin, box.xmax, box.ymax, box.zmax],
        'volume_mm3': shape.Volume(),
        'generator_style_broad_bbox_mm': [a[3]-a[0], a[4]-a[1], a[5]-a[2]],
    })
(root / 'step-geometry.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(f'Imported {len(results)} STEP instances; labels were not executed.')
