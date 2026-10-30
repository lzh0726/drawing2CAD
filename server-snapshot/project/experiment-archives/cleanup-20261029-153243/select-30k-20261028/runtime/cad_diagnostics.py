"""Trusted diagnostics, called only inside the CAD isolation runtime."""
import ast
import contextlib
import io


def diagnose(code, namespace, cq):
    from OCP.BRepCheck import BRepCheck_Analyzer
    hints = {
        'BRepCheck_IntersectingWires': 'Check touching or overlapping inner/outer boundaries. An open notch must not be represented as an internal hole.',
        'BRepCheck_SelfIntersectingWire': 'Check arc endpoints and contour traversal order; rebuild a non-self-intersecting boundary.',
        'BRepCheck_UnorientableShape': 'Check whether the profile bounds a valid planar region.',
    }
    for statement in ast.parse(code).body:
        record = {'line': statement.lineno, 'operation': ast.get_source_segment(code, statement)[:240]}
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exec(compile(ast.Module(body=[statement], type_ignores=[]), 'candidate.py', 'exec'), namespace)
            if not isinstance(statement, ast.Assign):
                continue
            for target in statement.targets:
                if not isinstance(target, ast.Name):
                    continue
                value = namespace.get(target.id)
                if not isinstance(value, cq.Workplane):
                    continue
                for shape in value.vals():
                    if not isinstance(shape, cq.Shape) or shape.isValid():
                        continue
                    analyzer = BRepCheck_Analyzer(shape.wrapped)
                    statuses = set()
                    for item in [shape] + shape.Faces() + shape.Wires() + shape.Edges():
                        try:
                            statuses.update(str(s).split('.')[-1] for s in analyzer.Result(item.wrapped).Status())
                        except Exception:
                            pass
                    statuses.discard('BRepCheck_NoError')
                    solids = shape.Solids()
                    return dict(record, status='invalid_geometry', variable=target.id,
                                solid_count=len(solids), kernel_errors=sorted(statuses),
                                hints=[hints[s] for s in sorted(statuses) if s in hints])
        except Exception as exc:
            return dict(record, status='operation_exception', exception_type=type(exc).__name__)
    result = namespace.get('result')
    solids = result.solids().vals() if isinstance(result, cq.Workplane) else []
    valid = len(solids) == 1 and solids[0].isValid() and solids[0].Volume() > 0
    return {'status': 'passed' if valid else 'invalid_final_result', 'solid_count': len(solids)}
