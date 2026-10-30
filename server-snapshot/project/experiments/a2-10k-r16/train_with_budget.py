"""A2 training entry point with a PyTorch allocation budget on the selected GPU."""
import os
import subprocess

uuid=os.environ['CUDA_VISIBLE_DEVICES']
row=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
used,util=[int(x.strip()) for x in row.split(',')]
if used>=5000 or util>=10:raise RuntimeError('GPU occupancy changed before model loading')
import torch
total=torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(34*1024**3/total,1.0),device=0)
from llamafactory.cli import main
main()
