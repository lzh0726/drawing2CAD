import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from server.gpu_sharing import select_gpu, snapshot

def gpu(index='1', free=40000, util=10, used=9000, pids=None):
    return dict(index=index,uuid='GPU-'+index,total=49140,free=free,used=used,util=util,pids={10} if pids is None else pids)

def history(g):
    return [(i*2,{g['uuid']:dict(g)}) for i in range(6)]

class SharingTests(unittest.TestCase):
    def test_shared_requires_full_stable_window(self):
        h=history(gpu())
        self.assertIsNone(select_gpu(h[:5]))
        self.assertTrue(select_gpu(h)['shared'])

    def test_rejects_insufficient_memory_busy_and_fluctuating_cards(self):
        for g in (gpu(free=36000),gpu(util=21)):
            self.assertIsNone(select_gpu(history(g)))
        h=history(gpu());h[0][1]['GPU-1']['free']=39000
        self.assertIsNone(select_gpu(h))
        h=history(gpu());h[0][1]['GPU-1']['pids']={99}
        self.assertIsNone(select_gpu(h))

    def test_idle_preferred_over_shared(self):
        h=history(gpu())
        for _,s in h:s['GPU-2']=gpu(index='2',free=49000,used=100,util=0,pids=set())
        selected=select_gpu(h)
        self.assertEqual(selected['uuid'],'GPU-2');self.assertFalse(selected['shared'])

    def test_missing_gpu_and_invalid_metrics_fail_closed(self):
        h=history(gpu());h[2][1].clear()
        self.assertIsNone(select_gpu(h))
        with patch('server.gpu_sharing.subprocess.check_output',side_effect=['1, GPU-1, 49140, 40000, 9000, N/A','GPU-1, 10']):
            with self.assertRaises(ValueError):snapshot()

    def test_snapshot_tracks_compute_processes(self):
        with patch('server.gpu_sharing.subprocess.check_output',side_effect=['1, GPU-1, 49140, 40000, 9000, 10','GPU-1, 10\nGPU-1, 11']):
            self.assertEqual(snapshot()['GPU-1']['pids'],{10,11})

if __name__=='__main__':unittest.main()
