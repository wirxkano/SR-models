import cv2
import torch
import numpy as np
from pathlib import Path
import math
import os
from PIL import Image

def calculate_metrics_y_channel(hr_img, lr_img, scale):
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
    lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
    
    h, w = hr_img.shape[:2]
    lr_upsampled = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_CUBIC)
    
    hr_ycrcb = cv2.cvtColor(hr_img, cv2.COLOR_RGB2YCrCb)
    lr_ycrcb = cv2.cvtColor(lr_upsampled, cv2.COLOR_RGB2YCrCb)
    
    hr_y = hr_ycrcb[:, :, 0].astype(np.float64)
    lr_y = lr_ycrcb[:, :, 0].astype(np.float64)
    
    shave = scale
    if shave > 0:
        hr_y = hr_y[shave:-shave, shave:-shave]
        lr_y = lr_y[shave:-shave, shave:-shave]
    
    psnr_value = psnr(hr_y, lr_y, data_range=255.0)
    ssim_value = ssim(hr_y, lr_y, data_range=255.0)
    
    return psnr_value, ssim_value, lr_upsampled

def process_image_folders(lr_folder, hr_folder, save_folder, calc_psnr=True, scale=4):
    lr_path = Path(lr_folder)
    if calc_psnr:
        hr_path = Path(hr_folder)
    else:
        hr_path = lr_path
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    lr_images = sorted([f for f in lr_path.iterdir() if f.suffix.lower() in image_extensions])
    hr_images = sorted([f for f in hr_path.iterdir() if f.suffix.lower() in image_extensions])
    
    if len(lr_images) != len(hr_images):
        print(f"Warning: Number of LR images ({len(lr_images)}) != Number of HR images ({len(hr_images)})")
    
    results = []
    
    print(f"Processing {len(lr_images)} image pairs with scale={scale}x...\n")
    
    for lr_file, hr_file in zip(lr_images, hr_images):
        lr_img = cv2.imread(str(lr_file))
        hr_img = cv2.imread(str(hr_file))
        
        if lr_img is None or hr_img is None:
            print(f"Error reading: {lr_file.name} or {hr_file.name}")
            continue
        
        if calc_psnr:
            psnr_lr, ssim_lr, lr_upsampled = calculate_metrics_y_channel(hr_img, lr_img, scale)
        else:
            psnr_lr, ssim_lr = 0, 0
            h, w = lr_img.shape[:2]
            lr_upsampled = cv2.resize(lr_img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
            
        lr_upsampled = Image.fromarray(lr_upsampled)
        lr_upsampled.save(os.path.join(save_folder, lr_file.name))
        
        if calc_psnr:
            lr_h, lr_w = lr_img.shape[:2]
            hr_downsampled = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
            psnr_bicubic, ssim_bicubic, _ = calculate_metrics_y_channel(hr_img, hr_downsampled, scale)
        else:
            psnr_bicubic, ssim_bicubic = 0, 0
        
        results.append({
            'lr_name': lr_file.name,
            'hr_name': hr_file.name,
            'psnr_lr': psnr_lr,
            'ssim_lr': ssim_lr,
            'psnr_bicubic': psnr_bicubic,
            'ssim_bicubic': ssim_bicubic
        })
        
        print(f"{lr_file.name} <-> {hr_file.name}")
        print(f"  LR Bicubic Upsampling:")
        print(f"    PSNR (Y): {psnr_lr:.2f} dB")
        print(f"    SSIM (Y): {ssim_lr:.4f}")
        print(f"  HR Downsampled then Upsampled (Bicubic Baseline):")
        print(f"    PSNR (Y): {psnr_bicubic:.2f} dB")
        print(f"    SSIM (Y): {ssim_bicubic:.4f}\n")
    
    if results:
        avg_psnr_lr = np.mean([r['psnr_lr'] for r in results])
        avg_ssim_lr = np.mean([r['ssim_lr'] for r in results])
        avg_psnr_bicubic = np.mean([r['psnr_bicubic'] for r in results])
        avg_ssim_bicubic = np.mean([r['ssim_bicubic'] for r in results])
        
        print("="*60)
        print("LR Bicubic Upsampling (Original LR):")
        print(f"  Average PSNR (Y): {avg_psnr_lr:.2f} dB")
        print(f"  Average SSIM (Y): {avg_ssim_lr:.4f}")
        print()
        print("HR Downsampled->Upsampled Bicubic Baseline:")
        print(f"  Average PSNR (Y): {avg_psnr_bicubic:.2f} dB")
        print(f"  Average SSIM (Y): {avg_ssim_bicubic:.4f}")
        print("="*60)
    
    return results

if __name__ == "__main__":
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    datasetname = "RealSR"
    # lr_folder = f"/root/media/quoc-huy/blur-DAN/{datasetname}/LRbic"
    lr_folder = f"/root/quoc-huy/BSRGAN/testsets/RealSRSet"
    hr_folder = f"/root/media/quoc-huy/eval-wir/srbenchmarks/{datasetname}/HR"
    save_folder = f"/root/quoc-huy/all-tested-results/Bicubic/{datasetname}"
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    
    results = process_image_folders(lr_folder, hr_folder, save_folder, calc_psnr=False, scale=4)