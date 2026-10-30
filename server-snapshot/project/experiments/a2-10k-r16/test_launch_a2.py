import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('a2',Path(__file__).with_name('launch_a2.py'))
a2=importlib.util.module_from_spec(spec);spec.loader.exec_module(a2)

class AdmissionTests(unittest.TestCase):
    def test_occupied_low_load_gpu_is_allowed(self):
        self.assertEqual(a2.eligible_from_text('0, GPU-test, 4999, 9'),[(0,'GPU-test')])
    def test_strict_thresholds(self):
        self.assertEqual(a2.eligible_from_text('0, GPU-a, 5000, 0\n1, GPU-b, 4, 10'),[])
    def test_xorg_memory_is_allowed(self):
        self.assertEqual(a2.eligible_from_text('2, GPU-c, 4, 0'),[(2,'GPU-c')])
    def test_unknown_metrics_fail_closed(self):
        with self.assertRaises(ValueError):a2.eligible_from_text('0, GPU-a, N/A, 0')
    def test_memory_growth_is_not_stable(self):
        h=[([(0,'GPU-a')],'0, GPU-a, 1000, 0'), ([(0,'GPU-a')],'0, GPU-a, 2000, 0'), ([(0,'GPU-a')],'0, GPU-a, 2500, 0')]
        self.assertIsNone(a2.stable_gpu(h))
    def test_gpu_must_qualify_in_every_sample(self):
        h=[([(0,'GPU-a')],'0, GPU-a, 1000, 0'), ([], '0, GPU-a, 1000, 12'), ([(0,'GPU-a')],'0, GPU-a, 1000, 0')]
        self.assertIsNone(a2.stable_gpu(h))

if __name__=='__main__':unittest.main()
