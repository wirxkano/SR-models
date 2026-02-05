import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import torch
import pyiqa

def img2tensor(img, device):
    # img: HWC, BGR (cv2), uint8 [0,255]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0           # [0,1]
    img = torch.from_numpy(img).permute(2, 0, 1)   # CHW
    img = img.unsqueeze(0).to(device)               # BCHW
    return img


def rgb_to_ycrcb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

def calculate_metrics_y_channel(hr_image, lr_image):
    h, w = hr_image.shape[:2]
    lr_upsampled = cv2.resize(lr_image, (w, h), interpolation=cv2.INTER_CUBIC)
    
    hr_ycrcb = rgb_to_ycrcb(hr_image)
    lr_ycrcb = rgb_to_ycrcb(lr_upsampled)
    
    hr_y = hr_ycrcb[:, :, 0]
    lr_y = lr_ycrcb[:, :, 0]
    
    psnr_value = psnr(hr_y, lr_y, data_range=255)
    
    ssim_value = ssim(hr_y, lr_y, data_range=255)
    
    return psnr_value, ssim_value, lr_upsampled

def process_image_folders(lr_folder, hr_folder, sr_folder):
    device = torch.device("cuda")
    lpips = pyiqa.create_metric('lpips', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    fid_metric = pyiqa.create_metric('fid', device=device)
    lr_path = Path(lr_folder)
    hr_path = Path(hr_folder)
    sr_path = Path(sr_folder)
    sr_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    lr_images = sorted([f for f in lr_path.iterdir() if f.suffix.lower() in image_extensions])
    hr_images = sorted([f for f in hr_path.iterdir() if f.suffix.lower() in image_extensions])
    
    if len(lr_images) != len(hr_images):
        print(f"Warning: Number of LR images ({len(lr_images)}) != Number of HR images ({len(hr_images)})")
    
    results = []
    print(f"Processing {len(lr_images)} image pairs...\n")
    
    for lr_file, hr_file in zip(lr_images, hr_images):
        lr_img = cv2.imread(str(lr_file))
        hr_img = cv2.imread(str(hr_file))
        
        if lr_img is None or hr_img is None:
            print(f"Error reading: {lr_file.name} or {hr_file.name}")
            continue
        
        psnr_val, ssim_val, sr_img = calculate_metrics_y_channel(hr_img, lr_img)
        
        sr_save_path = sr_path / lr_file.name
        cv2.imwrite(str(sr_save_path), sr_img)
        
        hr_tensor = img2tensor(hr_img, device)
        sr_tensor = img2tensor(sr_img, device)

        with torch.no_grad():
            lpips_score = lpips(sr_tensor, hr_tensor).item()
            niqe_score = niqe_metric(img2tensor(sr_img, device)).item()
        
        results.append({
            'lr_name': lr_file.name,
            'hr_name': hr_file.name,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'lpips': lpips_score,
            'niqe': niqe_score
        })
        
        print(f"{lr_file.name} <-> {hr_file.name}")
        print(f"  PSNR (Y): {psnr_val:.4f} dB")
        print(f"  SSIM (Y): {ssim_val:.4f}")
        print(f"  LPIPS:   {lpips_score:.4f}")
        print(f"  NIQE:    {niqe_score:.4f}\n")
        
    with torch.no_grad():
        fid_score = fid_metric(str(sr_path), str(hr_path)).item()
    
    if results:
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        avg_lpips = np.mean([r['lpips'] for r in results])
        avg_niqe = np.mean([r['niqe'] for r in results])
        
        print("="*50)
        print(f"Average PSNR (Y): {avg_psnr:.2f} dB")
        print(f"Average SSIM (Y): {avg_ssim:.4f}")
        print(f"Average LPIPS (Y): {avg_lpips:.4f}")
        print(f"FID:             {fid_score:.4f}")
        print(f"Average NIQE:    {avg_niqe:.4f}")
        print("="*50)
    
    return results

if __name__ == "__main__":
    datasetname = "Set14"
    lr_folder = f"/root/media/quoc-huy/eval-wir/srbenchmarks/{datasetname}/LR-bicubic-matlab-v2"
    hr_folder = f"/root/media/quoc-huy/eval-wir/srbenchmarks/{datasetname}/HR"
    sr_folder = f"/root/quoc-huy/all-tested-results/Bicubic/{datasetname}LR-bicubic-matlab"
    
    results = process_image_folders(lr_folder, hr_folder, sr_folder)
    