import sys
import time
import yaml
import torch
from pathlib import Path

from torch.profiler import profile, ProfilerActivity
sys.path.append(str(Path(__file__).parent.parent))
from BSRGAN.models.network_rrdbnet import RRDBNet as net
# from arbitrary_scale_blind_SR import models
# from arbitrary_scale_blind_SR.models.controller import *
from DASR.option import args
from DASR import model, utility
from DAN.codes.config.DANv1.models import create_model
from DAN.codes.config.DANv1 import options as DAN_option
from DAN.codes.config.DANv1.models import networks as DANnetworks
from MANet.codes.models import create_model as MANet_create_model
from MANet.codes.models import networks as MANetnetworks
from MANet.codes.options import options as MANet_option

MODEL_NAME = "DAN"
device = torch.device("cuda")

if MODEL_NAME == "BSRNet":
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4).to(device)
elif MODEL_NAME == "DASR":
    checkpoint = utility.checkpoint(args)
    model = model.Model(args, checkpoint)
    model.eval()
elif MODEL_NAME == "DAN":
    path = "/root/quoc-huy/DAN/codes/config/DANv1/options/setting1/test/test_setting1_x4.yml"
    opt = DAN_option.parse(path, is_train=False)
    opt = DAN_option.dict_to_nonedict(opt)
    model = create_model(opt)
    model = DANnetworks.define_G(opt).to(device)
    model.eval()
elif MODEL_NAME == "MANet":
    path = "/root/quoc-huy/MANet/codes/options/test/test_stage3.yml"
    opt = MANet_option.parse(path, is_train=False)
    opt = MANet_option.dict_to_nonedict(opt)
    model = MANet_create_model(opt)
    model = MANetnetworks.define_G(opt).to(device)
    model.eval()
elif MODEL_NAME == "ASBSR":
    pass
    yaml_path = "/root/quoc-huy/arbitrary_scale_blind_SR/configs/train-div2k/train_SR.yaml"
    with open(yaml_path, "r") as f:
        modelconfig = yaml.load(f, Loader=yaml.FullLoader)
    # model = models.make(modelconfig['model'], args={'config': modelconfig}).to(device)
  
batch_size = 1
channels = 3
height = 256
width = 256 
inputs = torch.randn(batch_size, channels, height, width).to(device)

### Warm-up
with torch.no_grad():
    for _ in range(5):
        if MODEL_NAME == "MANet":
            kernel = torch.randn(1, 1, 1, 1).to(device)
            model(inputs, kernel)
        else:
            _ = model(inputs)
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
        if MODEL_NAME == "MANet":
            kernel = torch.randn(1, 1, 1, 1).to(device)
            model(inputs, kernel)
        else:
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
