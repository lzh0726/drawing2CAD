"""Conservative admission for inference only; no GPU work is started here."""
import subprocess

SHARED_FREE_MIB = 36 * 1024
SHARED_LIMIT_MIB = 28 * 1024
MIN_RESERVE_MIB = 6 * 1024

def snapshot():
    info = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=5)
    apps = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=5)
    processes = {}
    for line in apps.splitlines():
        uuid, pid = (v.strip() for v in line.split(','))
        processes.setdefault(uuid, set()).add(int(pid))
    result = {}
    for line in info.splitlines():
        index, uuid, total, free, used, util = (v.strip() for v in line.split(','))
        result[uuid] = dict(index=index, uuid=uuid, total=int(total), free=int(free),
                            used=int(used), util=int(util), pids=processes.get(uuid, set()))
    return result

def select_gpu(history):
    """Samples are (monotonic time, snapshot); prefer two-sample idle cards."""
    if len(history) < 2 or history[-1][0] - history[-2][0] < 1:
        return None
    eligible = []
    for uuid, last in history[-1][1].items():
        recent = [s.get(uuid) for _, s in history[-2:]]
        if all(g and not g['pids'] and g['used'] < 256 and g['util'] == 0
               and g['free'] >= SHARED_FREE_MIB for g in recent):
            eligible.append((False, last))
            continue
        if len(history) < 6 or history[-1][0] - history[-6][0] < 10:
            continue
        recent = [s.get(uuid) for _, s in history[-6:]]
        if not all(g and g['free'] >= SHARED_FREE_MIB and g['util'] <= 20 for g in recent):
            continue
        if any(g['pids'] != last['pids'] for g in recent):
            continue
        if max(g['free'] for g in recent) - min(g['free'] for g in recent) > 512:
            continue
        eligible.append((True, last))
    if not eligible:
        return None
    shared, gpu = min(eligible, key=lambda item: (item[0], -item[1]['free'], item[1]['util'], int(item[1]['index'])))
    return dict(gpu, shared=shared)
