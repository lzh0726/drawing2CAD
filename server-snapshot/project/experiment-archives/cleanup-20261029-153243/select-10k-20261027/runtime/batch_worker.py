"""CPU dataset checks: import CAD once, fork a fresh child for each sample.

This entry point is mounted read-only and launched ONLY by research_sandbox.command.
Candidate code executes only in children inside that Bubblewrap namespace.
"""
import base64
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import sys

resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
resource.setrlimit(resource.RLIMIT_FSIZE, (40 * 1024**2, 40 * 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
sys.path.insert(0, '/runner')
import cadquery  # Trusted library preloaded; never load a VLM or CUDA.


def child(request, connection, folder):
    os.chdir(folder)
    output = io.BytesIO()
    sys.stdin = io.TextIOWrapper(io.BytesIO(json.dumps(request).encode()), encoding='utf-8')
    sys.stdout = io.TextIOWrapper(output, encoding='utf-8')
    sys.stderr = io.StringIO()
    try:
        # Import in the child so the original per-candidate resource limits apply.
        import research_worker
        research_worker.main()
        sys.stdout.flush()
        data = output.getvalue()
        if len(data) > 45 * 1024**2:
            raise ValueError('CAD_output_too_large')
        response = {'ok': True, 'data': base64.b64encode(data).decode()}
    except BaseException as exc:
        response = {'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)[:300]}
    connection.send(response)
    connection.close()


def main():
    context = mp.get_context('fork')
    for index, line in enumerate(sys.stdin):
        if index >= 64:
            break
        request = json.loads(line)
        folder = Path('/work') / ('sample-' + str(index))
        folder.mkdir()
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(target=child, args=(request, writer, str(folder)))
        process.start()
        writer.close()
        if reader.poll(45):
            try:
                result = reader.recv()
            except EOFError:
                result = {'ok': False, 'error': 'CAD_child_exited'}
        else:
            result = {'ok': False, 'error': 'CAD_timeout'}
        process.join(1)
        if process.is_alive():
            process.kill()
            process.join()
        reader.close()
        # Each worker is serial; the original worker writes this fixed sandbox path.
        for name in ('model.step', 'input.step'):
            (Path('/work') / name).unlink(missing_ok=True)
        shutil.rmtree(folder)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
