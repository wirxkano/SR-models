import sys
import time
import yaml
import torch
from pathlib import Path
import models.networks as networks
import options.options as option

from torch.profiler import profile, ProfilerActivity
# sys.path.append(str(Path(__file__).parent.parent))

MODEL_NAME = "DAN"
device = torch.device("cuda")
  
batch_size = 1
channels = 3
height = 256
width = 256 
inputs = torch.randn(batch_size, channels, height, width).to(device)
blur_map = torch.ones(1, 1, 1)
seg_map = torch.ones(1, 1, 1)

path = "/root/quoc-huy/CMOS/codes/options/test/NYUv2_BSR/test_stage3.yml"
opt = option.parse(path, is_train=False)
opt = option.dict_to_nonedict(opt)
netG = networks.define_G(opt).to(device)

### Warm-up
with torch.no_grad():
    for _ in range(5):
        netG.eval()
        with torch.no_grad():
            netG(inputs, blur_map, seg_map)
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
        netG.eval()
        with torch.no_grad():
            netG(inputs, blur_map, seg_map)
        
        if device.type == 'cuda':
                torch.cuda.synchronize()
        end = time.perf_counter_ns()

# print(prof.key_averages().table(sort_by="flops", row_limit=10))
# prof.export_chrome_trace("trace.json")

total_flops = sum([event.flops for event in prof.key_averages()])
print(f"\n#### Total FLOPs: {total_flops:,}")
print(f"#### Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
print(f"#### Execution Time: {(end - start) / 1e6:.2f} ms")
print(f"#### Throughput: {total_flops / (end - start):.2f} GFLOPS")
