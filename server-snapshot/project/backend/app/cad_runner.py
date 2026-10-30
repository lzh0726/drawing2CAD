"""Execute trusted templates or import STEP. Never execute generated Python."""
import json
import sys
from pathlib import Path


def template(p):
    # Only validated numeric values are interpolated into this downloadable source.
    return f'''import cadquery as cq
length, width, thickness, hole = {p['length']}, {p['width']}, {p['thickness']}, {p['hole']}
result = cq.Workplane("XY").box(length, width, thickness)
result = result.edges("|Z").fillet(3)
result = result.faces(">Z").workplane().hole(hole)
points = [(x, y) for x in (-length/2+10, length/2-10) for y in (-width/2+10, width/2-10)]
result = result.faces(">Z").workplane().pushPoints(points).hole(6)
'''


def drawing(p, target):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new('RGB', (1100, 760), '#faf9f6')
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype('arial.ttf', 22)
        small = ImageFont.truetype('arial.ttf', 17)
    except OSError:
        f = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=17)
    ink, muted = '#253142', '#697384'
    d.text((48, 30), 'PARAMETRIC SAMPLE  /  MOUNTING PLATE', fill=ink, font=f)
    d.text((48, 67), 'Reference diagram - known template, not AI reconstruction. Unit: mm', fill=muted, font=small)
    scale = min(4.8, 420 / p['length'], 300 / p['width'])
    l, w, t, h = (p[k] * scale for k in ('length', 'width', 'thickness', 'hole'))
    x, y = 90, 145
    d.rounded_rectangle((x,y,x+l,y+w), radius=3*scale, outline=ink, width=3)
    cx, cy = x+l/2, y+w/2
    d.ellipse((cx-h/2,cy-h/2,cx+h/2,cy+h/2),outline=ink,width=3)
    for dx in (10*scale,l-10*scale):
        for dy in (10*scale,w-10*scale):
            d.ellipse((x+dx-3*scale,y+dy-3*scale,x+dx+3*scale,y+dy+3*scale),outline=ink,width=2)
    d.line((x,y-22,x+l,y-22),fill=muted,width=1)
    d.text((x+l/2-30,y-52),f"{p['length']:g}",fill=ink,font=small)
    d.text((x+l+18,y+w/2),f"{p['width']:g}",fill=ink,font=small)
    d.text((x,y+w+18),f"TOP   center D{p['hole']:g}   4 x D6",fill=ink,font=small)
    fy = 570
    d.rectangle((x,fy,x+l,fy+t),outline=ink,width=3)
    for xx in (cx-h/2,cx+h/2):
        for yy in range(fy, int(fy+t), 10): d.line((xx,yy,xx,min(yy+5,fy+t)),fill=muted,width=1)
    d.text((x,fy+t+18),f"FRONT   thickness {p['thickness']:g}",fill=ink,font=small)
    sx = 650
    d.rectangle((sx,fy,sx+w,fy+t),outline=ink,width=3)
    d.text((sx,fy+t+18),'RIGHT',fill=ink,font=small)
    d.text((650,155),'Drawing2CAD',fill=ink,font=f)
    d.text((650,198),'LOCAL CAD WORKBENCH',fill=muted,font=small)
    d.text((650,240),'Plate / fillet R3 / 5 through holes',fill=muted,font=small)
    im.save(target)


def run(mode, folder):
    import cadquery as cq
    folder = Path(folder)
    if mode == 'demo':
        from backend.app.schemas import Parameters
        p = Parameters.model_validate_json((folder/'params.json').read_text()).model_dump()
        result = cq.Workplane('XY').box(p['length'],p['width'],p['thickness'])
        result = result.edges('|Z').fillet(3)
        result = result.faces('>Z').workplane().hole(p['hole'])
        points = [(x,y) for x in (-p['length']/2+10,p['length']/2-10) for y in (-p['width']/2+10,p['width']/2-10)]
        result = result.faces('>Z').workplane().pushPoints(points).hole(6)
        (folder/'model.py').write_text(template(p),encoding='utf-8')
        cq.exporters.export(result,str(folder/'model.step'))
        drawing(p,folder/'drawing.png')
    else:
        result = cq.importers.importStep(str(folder/'model.step'))
    solids = result.solids().vals()
    if len(solids) != 1 or not solids[0].isValid() or solids[0].Volume() <= 0:
        raise ValueError('Expected one valid nonempty solid')
    solid = solids[0]
    bb = solid.BoundingBox()
    cq.exporters.export(result,str(folder/'model.stl'),tolerance=0.05,angularTolerance=0.1)
    checks = [{'name':'实体有效性','status':'passed','detail':'OpenCascade 检查通过，1 个实体'},
              {'name':'实体体积','status':'passed','detail':f'{solid.Volume():.2f} mm³'},
              {'name':'图纸符合性','status':'unknown','detail':'未进行自动图纸尺寸与拓扑一致性验证'}]
    if mode == 'demo':
        dims = (bb.xlen,bb.ylen,bb.zlen)
        ok = all(abs(a-b)<0.01 for a,b in zip(dims,(p['length'],p['width'],p['thickness'])))
        checks.insert(1,{'name':'样例外形尺寸','status':'passed' if ok else 'failed','detail':'包围盒与指定长宽厚比较，误差限 0.01 mm'})
    report = {'dimensions':[round(bb.xlen,3),round(bb.ylen,3),round(bb.zlen,3)],'volume':round(solid.Volume(),2),'checks':checks,'unit':'mm'}
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8')


if __name__ == '__main__':
    run(sys.argv[1],sys.argv[2])
