import os
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

        return out.clamp(0, 1)
    
def degrade(hr_dir, save_dir, scale=1/4):
    device = torch.device("cuda")
    downsampling = bicubic()
    for file in sorted(os.listdir(hr_dir)):
        hr_img = Image.open(os.path.join(hr_dir, file))
        hr = transforms.ToTensor()(hr_img).unsqueeze(0).to(device)
        lr = downsampling(hr, scale).squeeze()
        lr_img = transforms.ToPILImage()(lr)
        lr_img.save(os.path.join(save_dir, file))
        
    print("Done!!!!")
    
def main():
    datasetname = "Set14"
    hr_dir = f"/root/media/quoc-huy/eval-wir/srbenchmarks/{datasetname}/HR"
    save_dir = f"/root/media/quoc-huy/eval-wir/srbenchmarks/{datasetname}/LR-bicubic-matlab-v2"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    degrade(hr_dir, save_dir, scale=1/4)
    
if __name__ == "__main__":
    main()
    