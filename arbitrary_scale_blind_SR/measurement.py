import yaml
import time
import torch
from torch.profiler import profile, ProfilerActivity

import models
from models.controller import *
from utils import make_coord

device = torch.device("cuda")

yaml_path = "/root/quoc-huy/arbitrary_scale_blind_SR/configs/train-div2k/train_SR.yaml"
with open(yaml_path, "r") as f:
    modelconfig = yaml.load(f, Loader=yaml.FullLoader)
model = models.make(modelconfig['model'], args={'config': modelconfig}).to(device)
model.eval()
batch_size = 1
channels = 3
height = 256
width = 256
target_height = height * 4
target_width = width * 4
inputs = torch.randn(batch_size, channels, height, width).to(device)
coord = make_coord((target_height, target_width), ranges=None, flatten=False)  # [H, W, 2]
coord = coord.view(-1, coord.shape[-1])  # Flatten to [H*W, 2]
coord = coord.unsqueeze(0).to(device)  # Add batch dimension [1, H*W, 2]

# Create cell - must match the number of query points
num_points = target_height * target_width
cell = torch.ones(batch_size, num_points, 2).to(device)  # [1, H*W, 2]
cell[:, :, 0] = 2.0 / target_height
cell[:, :, 1] = 2.0 / target_width

### Warm-up
with torch.no_grad():
    for _ in range(5):
        model(inputs, coord, cell)
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
        model(inputs, coord, cell)
        
        if device.type == 'cuda':
                torch.cuda.synchronize()
        end = time.perf_counter_ns()

# print(prof.key_averages().table(sort_by="flops", row_limit=10))

total_flops = sum([event.flops for event in prof.key_averages()])
print(f"\n#### Total FLOPs: {total_flops:,}")
print(f"#### Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
print(f"#### Execution Time: {(end - start) / 1e6:.2f} ms")
print(f"#### Throughput: {total_flops / (end - start):.2f} GFLOPS")
