import json
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from server.demand_model import DemandModel

class DemandTests(unittest.TestCase):
    def test_no_gpu_does_not_spawn(self):
        model=DemandModel('unused',None,wait_seconds=0)
        with patch('tools.launch_training.idle_gpus',return_value=([],'')),patch('server.demand_model.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError,'No idle GPU'):model.generate(b'',{})
            spawn.assert_not_called()
        self.assertEqual(model.state,'idle')

    def test_cancel_wait(self):
        model=DemandModel('unused',None);errors=[]
        def work():
            try:model.generate(b'',{})
            except RuntimeError as e:errors.append(str(e))
        with patch('tools.launch_training.idle_gpus',return_value=([],'')):
            thread=threading.Thread(target=work);thread.start();time.sleep(.1);model.cancel();thread.join(3)
        self.assertFalse(thread.is_alive());self.assertIn('cancelled',errors[0]);self.assertEqual(model.state,'idle')

    def test_timeout_terminates_only_child(self):
        model=DemandModel('unused',None,inference_seconds=.05)
        real_popen=subprocess.Popen;children=[]
        def spawn(*args,**kwargs):
            p=real_popen([sys.executable,'-c','import time;time.sleep(60)'],**kwargs);children.append(p);return p
        with patch('tools.launch_training.idle_gpus',return_value=([('1','GPU-test')],'')),patch('server.demand_model.subprocess.Popen',side_effect=spawn):
            with self.assertRaisesRegex(RuntimeError,'timeout'):model.generate(b'',{})
        self.assertIsNotNone(children[0].poll());self.assertEqual(model.state,'idle')

if __name__=='__main__':unittest.main()
