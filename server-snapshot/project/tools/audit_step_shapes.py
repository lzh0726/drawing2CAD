"""Trusted STEP geometry audit. Run only inside the existing bwrap sandbox."""
import base64
from collections import Counter
import json
from pathlib import Path
import resource
import sys

resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(35,35))
resource.setrlimit(resource.RLIMIT_FSIZE,(40*1024**2,40*1024**2))


def mesh_topology(shape):
    scale=max(shape.BoundingBox().xlen,shape.BoundingBox().ylen,shape.BoundingBox().zlen)
    vertices,triangles=shape.tessellate(scale/1000,0.15)
    weld=scale*1e-8
    mapping={};remap=[]
    for v in vertices:
        key=tuple(round(x/weld) for x in v.toTuple())
        if key not in mapping:mapping[key]=len(mapping)
        remap.append(mapping[key])
    faces=set()
    for triangle in triangles:
        face=tuple(remap[v] for v in triangle)
        if len(set(face))==3:faces.add(tuple(sorted(face)))
    edges=Counter();adj={}
    for a,b,c in faces:
        for x,y in [(a,b),(b,c),(c,a)]:
            edges[tuple(sorted((x,y)))]+=1
            adj.setdefault(x,set()).add(y);adj.setdefault(y,set()).add(x)
    remaining=set(adj);components=0
    while remaining:
        components+=1;stack=[remaining.pop()]
        while stack:
            for other in adj[stack.pop()]:
                if other in remaining:remaining.remove(other);stack.append(other)
    chi=len(adj)-len(edges)+len(faces)
    boundary=sum(count==1 for count in edges.values())
    nonmanifold=sum(count>2 for count in edges.values())
    closed=bool(faces) and not boundary and not nonmanifold
    genus=(2*components-chi)/2 if closed else None
    return {'method':'welded triangular boundary mesh; corroborating topology, not feature equivalence',
            'vertices':len(adj),'edges':len(edges),'triangles':len(faces),'boundary_components':components,
            'boundary_edges':boundary,'nonmanifold_edges':nonmanifold,'euler_characteristic':chi,
            'closed_2manifold_edges':closed,'total_surface_genus':genus,
            'mesh_tolerance':scale/1000,'weld_tolerance':weld}


def describe(shape):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    b=shape.BoundingBox();shells=shape.Shells();cylinders=[];types=Counter()
    for face in shape.Faces():
        surface=BRepAdaptor_Surface(face.wrapped,True)
        kind=str(surface.GetType()).split('.')[-1];types[kind]+=1
        if kind=='GeomAbs_Cylinder':
            cylinder=surface.Cylinder();axis=cylinder.Axis().Direction()
            cylinders.append({'radius':cylinder.Radius(),'axis':[axis.X(),axis.Y(),axis.Z()]})
    record={'valid':shape.isValid(),'solid_count':len(shape.Solids()),'shell_count':len(shells),
            'all_shells_closed':all(s.Closed() for s in shells),'volume':shape.Volume(),'area':shape.Area(),
            'bbox_dimensions':[b.xlen,b.ylen,b.zlen],
            'bbox_center':[(b.xmin+b.xmax)/2,(b.ymin+b.ymax)/2,(b.zmin+b.zmax)/2],
            'faces':len(shape.Faces()),'edges':len(shape.Edges()),'vertices':len(shape.Vertices()),
            'surface_types':dict(types),'cylindrical_surfaces':cylinders}
    try:record['mesh_topology']=mesh_topology(shape)
    except Exception as exc:record['mesh_topology']={'error_type':type(exc).__name__}
    return record


def main():
    import cadquery as cq
    request=json.loads(sys.stdin.buffer.read(6*1024**2))
    if request.get('selftest'):
        box=cq.Workplane('XY').box(2,3,1).val()
        ring=cq.Workplane('XY').circle(1).circle(.5).extrude(.2).val()
        a,b=mesh_topology(box),mesh_topology(ring)
        assert a['total_surface_genus']==0 and b['total_surface_genus']==1,(a,b)
        print(json.dumps({'box_genus':0,'ring_genus':1,'passed':True}));return
    report={}
    for key in ['prediction','truth']:
        path=Path('/work')/(key+'.step');path.write_bytes(base64.b64decode(request[key]))
        report[key]=describe(cq.importers.importStep(str(path)).val())
    p,t=report['prediction'],report['truth']
    pd,td=p['bbox_dimensions'],t['bbox_dimensions']
    xyz=[abs(a-b)/max(abs(b),1e-12) for a,b in zip(pd,td)]
    sorted_error=[abs(a-b)/max(abs(b),1e-12) for a,b in zip(sorted(pd),sorted(td))]
    pg,tg=p['mesh_topology'].get('total_surface_genus'),t['mesh_topology'].get('total_surface_genus')
    report['comparison']={'bbox_relative_xyz':xyz,'bbox_max_relative_xyz':max(xyz),
        'bbox_absolute_xyz':[abs(a-b) for a,b in zip(pd,td)],
        'sorted_bbox_max_relative':max(sorted_error),
        'volume_relative_error':abs(p['volume']-t['volume'])/t['volume'],
        'bbox_center_translation':[b-a for a,b in zip(p['bbox_center'],t['bbox_center'])],
        'surface_genus_match':pg==tg if pg is not None and tg is not None else None,
        'shell_count_match':p['shell_count']==t['shell_count'],
        'brep_face_count_match':p['faces']==t['faces'],
        'cylindrical_surface_count_match':len(p['cylindrical_surfaces'])==len(t['cylindrical_surfaces'])}
    print(json.dumps(report))


if __name__=='__main__':main()
