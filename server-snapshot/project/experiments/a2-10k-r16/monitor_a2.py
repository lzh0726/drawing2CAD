"""One-shot server-side admission watcher for this explicitly authorized training."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

STAGE=Path('/home/luzhuohong/drawing2CAD/experiments/a2-10k-r16')
PYTHON='/home/luzhuohong/miniconda3/envs/drawing2cad-train/bin/python'
STATE=STAGE/'monitor-state.json'


def save(value):
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(STATE)


def main():
    with (STAGE/'monitor.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        state={'pid':os.getpid(),'status':'monitoring','started_unix':time.time(),'interval_seconds':300}
        save(state)
        while True:
            try:
                result=subprocess.run([PYTHON,str(STAGE/'launch_a2.py')],capture_output=True,text=True,timeout=120)
                if result.returncode:
                    state.update(status='error',reason='launcher_failed',returncode=result.returncode,last_check_unix=time.time())
                    # Traceback is source-only; no environment or credential values are printed.
                    print(result.stderr[-2000:],flush=True);save(state);return
                record=json.loads(result.stdout.strip().splitlines()[-1])
                state.update(last_check_unix=time.time(),last_result=record)
                status=record.get('status')
                if status=='waiting':
                    state.update(status='waiting',next_check_unix=time.time()+300);save(state)
                    print(json.dumps({'time':time.time(),'result':record}),flush=True)
                    time.sleep(300);continue
                if status in ('submitted','starting','running','completed'):
                    launch=STAGE/'launch.json'
                    for _ in range(30):
                        current=json.loads(launch.read_text())
                        if current['status']=='running':
                            assert (Path('/proc')/str(current['pid'])).exists()
                            cmd=(Path('/proc')/str(current['train_pid'])/'cmdline').read_bytes()
                            assert b'train_with_budget.py' in cmd
                            state.update(status='training_started',launch=current,finished_unix=time.time());save(state);return
                        if current['status'] in ('failed','launch_failed','completed'):break
                        time.sleep(1)
                    if current['status']=='completed':
                        state.update(status='training_completed',launch=current,finished_unix=time.time());save(state);return
                    state.update(status='attention_required',launch=current,finished_unix=time.time());save(state);return
                state.update(status='attention_required',last_result=record,finished_unix=time.time());save(state);return
            except Exception as exc:
                state.update(status='error',error_type=type(exc).__name__,finished_unix=time.time());save(state);raise


if __name__=='__main__':main()
