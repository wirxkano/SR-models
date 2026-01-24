import sys
import time
import torch
from pathlib import Path

from torch.profiler import profile, ProfilerActivity
# sys.path.append(str(Path(__file__).parent.parent))
from models import create_model
import options as DCLSR_option
from models import networks as DLCS_network

device = torch.device("cuda")

path = "/root/quoc-huy/DCLS-SR/codes/config/DCLS/options/setting1/test/test_setting1_x4.yml"
opt = DCLSR_option.parse(path, is_train=False)
opt = DCLSR_option.dict_to_nonedict(opt)
model = create_model(opt)
model = DLCS_network.define_G(opt).to(device)
model.eval()

  
batch_size = 1
channels = 3
height = 256
width = 256 
inputs = torch.randn(batch_size, channels, height, width).to(device)

### Warm-up
with torch.no_grad():
    for _ in range(5):
        model(inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize()

with torch.no_grad():
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True
    ) as prof:
        if device.type == 'cuda':
                torch.cuda.synchronize()
        start = time.perf_counter_ns()
        model(inputs)
        
        if device.type == 'cuda':
                torch.cuda.synchronize()
        end = time.perf_counter_ns()

# print(prof.key_averages().table(sort_by="flops", row_limit=10))

total_flops = sum([event.flops for event in prof.key_averages()])
print(f"\n#### Total FLOPs: {total_flops:,}")
print(f"#### Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
print(f"#### Execution Time: {(end - start) / 1e6:.2f} ms")
print(f"#### Throughput: {total_flops / (end - start):.2f} GFLOPS")
