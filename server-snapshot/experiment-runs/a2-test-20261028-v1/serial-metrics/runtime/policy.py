"""A narrow modeling-language filter. Docker remains the isolation boundary."""
import ast
import re

MAX_CODE = 32_000
FORBIDDEN_CALLS = {'exec','eval','open','compile','input','getattr','setattr','delattr','globals','locals','vars','breakpoint','help','dir','type','object','memoryview','exit','quit'}
FORBIDDEN_ATTRS = {'exporters','importers','export','exportStep','exportStl','exportBrep','exportBin','importStep','importBrep','read','write','system','popen','remove','unlink','mkdir','rmdir'}

def extract_code(text):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,flags=re.S)
    if len(blocks)>1:
        raise ValueError('Return exactly one Python code block.')
    code=(blocks[0] if blocks else text).strip()
    if not code or len(code.encode('utf-8'))>MAX_CODE:
        raise ValueError('Code is empty or exceeds 32 KB.')
    return code+'\n'

def validate_code(code):
    tree=ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            if any(a.name not in ('cadquery','math') for a in node.names):
                raise ValueError('Only cadquery and math imports are supported.')
        if isinstance(node,(ast.ImportFrom,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef,ast.With,ast.AsyncWith,ast.Try,ast.Raise,ast.Global,ast.Nonlocal,ast.Delete,ast.Lambda,ast.While)):
            raise ValueError('Use simple modeling assignments, loops and expressions only.')
        if isinstance(node,ast.Name) and (node.id.startswith('_') or node.id in FORBIDDEN_CALLS):
            raise ValueError('Unsupported name in modeling code.')
        if isinstance(node,ast.Attribute) and (node.attr.startswith('_') or node.attr in FORBIDDEN_ATTRS):
            raise ValueError('File I/O and introspection are not modeling operations.')
    if not any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id=='result' for n in ast.walk(tree)):
        raise ValueError('Assign the final CadQuery Workplane to result.')
