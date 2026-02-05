import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms

class bicubic(nn.Module):
    def __init__(self):
        super(bicubic, self).__init__()

    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32).cuda()
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32).cuda()

        u0 = x0 / scale + 0.5 * (1 - 1 / scale)
        u1 = x1 / scale + 0.5 * (1 - 1 / scale)

        left0 = torch.floor(u0 - kernel_width / 2)
        left1 = torch.floor(u1 - kernel_width / 2)

        P = np.ceil(kernel_width) + 2

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).cuda()

        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0)
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale)
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2))
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))

        indice0 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice0), torch.FloatTensor([in_size[0]]).cuda()).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]).cuda(), indice1), torch.FloatTensor([in_size[1]]).cuda()).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0]
        kill1 = torch.eq(weight1, 0)[0][0]

        weight0 = weight0[:, :, kill0 == 0]
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0]
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input, scale=1/4):
        b, c, h, w = input.shape

        weight0, weight1, indice0, indice1 = self.contribute([h, w], [int(h * scale), int(w * scale)], scale)
        weight0 = weight0[0]
        weight1 = weight1[0]

        indice0 = indice0[0].long()
        indice1 = indice1[0].long()

        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = (torch.sum(out, dim=3))
        A = out.permute(0, 1, 3, 2)

        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = out.sum(3).permute(0, 1, 3, 2)

        return out

def degrade_matlab(hr_dir, scale=1/4):
    device = torch.device("cuda")
    downsampling = bicubic()
    
    times = []
    for file in sorted(os.listdir(hr_dir)):
        hr_img = Image.open(os.path.join(hr_dir, file))
        hr = transforms.ToTensor()(hr_img).unsqueeze(0).to(device)
        
        start = time.perf_counter()
        lr = downsampling(hr, scale).squeeze()
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
    
    return times

def degrade_opencv(hr_dir, scale=1/4):
    times = []
    for file in sorted(os.listdir(hr_dir)):
        hr_img = cv2.imread(os.path.join(hr_dir, file))
        h, w = hr_img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        start = time.perf_counter()
        lr_img = cv2.resize(hr_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        end = time.perf_counter()
        
        times.append(end - start)
    
    return times

def degrade_opencv_gpu(hr_dir, scale=1/4):
    """OpenCV CUDA version - requires opencv-contrib-python compiled with CUDA"""
    times = []
    
    if not cv2.cuda.getCudaEnabledDeviceCount():
        print("WARNING: OpenCV CUDA not available, falling back to CPU")
        return
    
    for file in sorted(os.listdir(hr_dir)):
        hr_img = cv2.imread(os.path.join(hr_dir, file))
        h, w = hr_img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(hr_img)
        
        start = time.perf_counter()
        gpu_lr = cv2.cuda.resize(gpu_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        lr_img = gpu_lr.download()
        end = time.perf_counter()
        
        times.append(end - start)
    
    return times

def main():
    datasetname = "Urban100"
    hr_dir = f"/root/media/quoc-huy/eval-wir/srbenchmarks/{datasetname}/HR"
    
    print("OpenCV version:", cv2.__version__)
    print("CUDA devices in OpenCV:", cv2.cuda.getCudaEnabledDeviceCount())
    
    print("Warming up...")
    degrade_matlab(hr_dir, scale=1/4)
    degrade_opencv(hr_dir, scale=1/4)
    
    print("\n" + "="*60)
    print("MATLAB Bicubic (PyTorch/CUDA)")
    print("="*60)
    matlab_times = degrade_matlab(hr_dir, scale=1/4)
    matlab_mean = np.mean(matlab_times)
    matlab_std = np.std(matlab_times)
    matlab_total = np.sum(matlab_times)
    print(f"Per-image: {matlab_mean*1000:.2f} ± {matlab_std*1000:.2f} ms")
    print(f"Total time: {matlab_total:.3f} s")
    print(f"Images processed: {len(matlab_times)}")
    
    print("\n" + "="*60)
    print("OpenCV Bicubic (CPU)")
    print("="*60)
    opencv_times = degrade_opencv(hr_dir, scale=1/4)
    opencv_mean = np.mean(opencv_times)
    opencv_std = np.std(opencv_times)
    opencv_total = np.sum(opencv_times)
    print(f"Per-image: {opencv_mean*1000:.2f} ± {opencv_std*1000:.2f} ms")
    print(f"Total time: {opencv_total:.3f} s")
    print(f"Images processed: {len(opencv_times)}")
    
    if cv2.cuda.getCudaEnabledDeviceCount():
        print("\n" + "="*60)
        print("OpenCV Bicubic (GPU/CUDA)")
        print("="*60)
        opencv_gpu_times = degrade_opencv_gpu(hr_dir, scale=1/4)
        opencv_gpu_mean = np.mean(opencv_gpu_times)
        opencv_gpu_std = np.std(opencv_gpu_times)
        opencv_gpu_total = np.sum(opencv_gpu_times)
        print(f"Per-image: {opencv_gpu_mean*1000:.2f} ± {opencv_gpu_std*1000:.2f} ms")
        print(f"Total time: {opencv_gpu_total:.3f} s")
        print(f"Images processed: {len(opencv_gpu_times)}")
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    speedup = opencv_mean / matlab_mean
    print(f"MATLAB is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than OpenCV")
    print(f"Time difference per image: {abs(matlab_mean - opencv_mean)*1000:.2f} ms")
    
    if cv2.cuda.getCudaEnabledDeviceCount():
        speedup_gpu = opencv_gpu_mean / matlab_mean
        print(f"MATLAB vs OpenCV GPU: {speedup_gpu:.2f}x {'faster' if speedup_gpu > 1 else 'slower'}")
    
    print("\nDone!!!!")

if __name__ == "__main__":
    main()
    