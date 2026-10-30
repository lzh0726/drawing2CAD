"""CPU-only checks for split fingerprints and isolated CAD metrics."""
import base64
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('prepare',ROOT/'tools/prepare_pilot.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
from server.research_sandbox import run

def test_family_excludes_numeric_variants():
    a='import cadquery as cq\nx=cq.Workplane("XY").box(1,2,3)'
    b='import cadquery as cq\ny=cq.Workplane("XY").box(4,5,6)'
    assert prepare.fingerprints(a)[0]!=prepare.fingerprints(b)[0]
    assert prepare.fingerprints(a)[1]==prepare.fingerprints(b)[1]
    assert prepare.fingerprints(a)[1]!=prepare.fingerprints(b.replace('box','sphere'))[1]

def test_comment_deduplication():
    assert prepare.fingerprints('x=1')[0]==prepare.fingerprints('# comment\nx = 1\n')[0]

def test_isolation_and_metrics():
    assert all(run({'probe':True}).values())
    cube='import cadquery as cq\nresult=cq.Workplane("XY").box(2,3,4)'
    a,_=run({'code':cube})
    b,_=run({'code':cube+'.translate((10,0,0))'})
    metrics=run({'prediction_b64':base64.b64encode(a).decode(),'truth_b64':base64.b64encode(b).decode()})
    assert metrics['iou_raw']==0
    assert abs(metrics['iou_translation_aligned']-1)<1e-6
    c,_=run({'code':cube.replace('2,3,4','4,6,8')})
    metrics=run({'prediction_b64':base64.b64encode(a).decode(),'truth_b64':base64.b64encode(c).decode()})
    assert abs(metrics['iou_translation_aligned']-.125)<1e-6
